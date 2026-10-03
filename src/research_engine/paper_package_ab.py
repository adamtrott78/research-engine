from __future__ import annotations

import copy
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .artifact_store import ContentAddressedArtifactStore
from .codex_executor import CodexExecExecutor
from .orchestration import ExecutorTier, JobType, OrchestratorStore, ResearchJob
from .paper_package_context import HybridPaperPackagePageTextExtractor
from .postgres import PostgresStore
from .semantic_pipeline import ResearchContextHydrator


def _without_execution(result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in dict(result or {}).items()
        if key not in {"_execution", "_materialization"}
    }


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(value).lower())


def token_f1(left: str, right: str) -> float:
    a = Counter(_tokens(left))
    b = Counter(_tokens(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    precision = overlap / sum(b.values())
    recall = overlap / sum(a.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def quote_token_recall(quote: str, context: str) -> float:
    q = Counter(_tokens(quote))
    c = Counter(_tokens(context))
    if not q:
        return 1.0
    return sum((q & c).values()) / sum(q.values())


def _code_sha() -> str | None:
    explicit = os.getenv("RESEARCH_ENGINE_CODE_SHA")
    if explicit:
        return explicit.strip() or None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _usage_from_result(result: dict[str, Any]) -> dict[str, Any]:
    execution = dict(result.get("_execution") or {})
    usage = dict(execution.get("usage") or {})
    return {
        "provider": execution.get("provider"),
        "model": execution.get("model"),
        "reasoning_effort": execution.get("reasoning_effort"),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
    }


class PaperPackageABPlanner:
    def __init__(self, *, store: PostgresStore, artifact_root: str | Path):
        self.store = store
        self.queue = OrchestratorStore(store)
        self.artifact_store = ContentAddressedArtifactStore(artifact_root)

    def select_run(self, *, expected_papers: int = 9) -> str:
        if expected_papers <= 0:
            raise ValueError("expected_papers must be positive")
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id,
                           count(*) FILTER (
                             WHERE j.job_type='EXTRACT' AND j.status='COMPLETED'
                           )::int AS extracts,
                           count(*) FILTER (
                             WHERE j.job_type='VERIFY' AND j.status='COMPLETED'
                           )::int AS verifies,
                           max(j.updated_at) AS latest_job
                    FROM research_runs r
                    JOIN research_jobs j ON j.run_id=r.id
                    GROUP BY r.id
                    HAVING count(*) FILTER (
                             WHERE j.job_type='EXTRACT' AND j.status='COMPLETED'
                           )=%s
                       AND count(*) FILTER (
                             WHERE j.job_type='VERIFY' AND j.status='COMPLETED'
                           )=%s
                    ORDER BY max(j.updated_at) DESC,r.id
                    """,
                    (expected_papers, expected_papers),
                )
                rows = cur.fetchall()
        if not rows:
            raise ValueError(
                f"no research run has exactly {expected_papers} completed "
                "EXTRACT and VERIFY jobs"
            )
        return str(rows[0][0])

    def _artifact_record(self, artifact_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,work_version_id,sha256,byte_size,media_type,format
                    FROM source_artifacts WHERE id=%s
                    """,
                    (artifact_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"source artifact not found: {artifact_id}")
                cols = [d.name for d in cur.description]
        record = dict(zip(cols, row, strict=True))
        path = self.artifact_store.path_for_hash(str(record["sha256"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        record["source_path"] = str(path)
        return record

    def usage(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT provider,model,reasoning_effort,
                           input_tokens,cached_input_tokens,
                           output_tokens,reasoning_output_tokens,
                           wall_time_ms
                    FROM research_usage_events WHERE job_id=%s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"job has no usage event: {job_id}")
        return {
            "provider": row[0],
            "model": row[1],
            "reasoning_effort": row[2],
            "input_tokens": int(row[3]),
            "cached_input_tokens": int(row[4]),
            "output_tokens": int(row[5]),
            "reasoning_output_tokens": int(row[6]),
            "wall_time_ms": None if row[7] is None else int(row[7]),
        }

    @staticmethod
    def _artifact_id(job: ResearchJob) -> str:
        if job.job_type == JobType.EXTRACT:
            value = (job.payload.get("acquisition") or {}).get("artifact_id")
        elif job.job_type == JobType.VERIFY:
            value = (
                (job.payload.get("extraction") or {})
                .get("_materialization", {})
                .get("artifact_id")
            )
        else:
            raise ValueError(f"unsupported A/B job type: {job.job_type}")
        if not value:
            raise ValueError(f"{job.id} lacks source artifact identity")
        return str(value)

    def completed_jobs(self, run_id: str, job_type: JobType) -> list[ResearchJob]:
        if job_type not in {JobType.EXTRACT, JobType.VERIFY}:
            raise ValueError("A/B supports only EXTRACT and VERIFY")
        return [
            job
            for job in self.queue.jobs(run_id, job_type)
            if job.status == "COMPLETED"
        ]

    def representative_jobs(
        self, run_id: str, job_type: JobType
    ) -> list[ResearchJob]:
        jobs = self.completed_jobs(run_id, job_type)
        ranked = sorted(
            (
                self.usage(job.id)["input_tokens"],
                job.id,
                job,
            )
            for job in jobs
        )
        if not ranked:
            return []
        indices = sorted({0, len(ranked) // 2, len(ranked) - 1})
        return [ranked[index][2] for index in indices]

    def plan(
        self,
        *,
        run_id: str | None = None,
        expected_papers: int = 9,
    ) -> dict[str, Any]:
        run_id = run_id or self.select_run(expected_papers=expected_papers)
        extract_jobs = self.completed_jobs(run_id, JobType.EXTRACT)
        verify_jobs = self.completed_jobs(run_id, JobType.VERIFY)
        if len(extract_jobs) != expected_papers or len(verify_jobs) != expected_papers:
            raise ValueError(
                f"run {run_id} has {len(extract_jobs)} EXTRACT and "
                f"{len(verify_jobs)} VERIFY jobs; expected {expected_papers} each"
            )

        sources: dict[str, dict[str, Any]] = {}
        for job in extract_jobs + verify_jobs:
            artifact_id = self._artifact_id(job)
            row = sources.setdefault(
                artifact_id,
                {
                    **self._artifact_record(artifact_id),
                    "extract_job_ids": [],
                    "verify_job_ids": [],
                },
            )
            key = (
                "extract_job_ids"
                if job.job_type == JobType.EXTRACT
                else "verify_job_ids"
            )
            row[key].append(job.id)

        if len(sources) != expected_papers:
            raise ValueError(
                f"run {run_id} resolves to {len(sources)} unique source artifacts; "
                f"expected {expected_papers}"
            )

        extract_usage = [self.usage(job.id) for job in extract_jobs]
        verify_usage = [self.usage(job.id) for job in verify_jobs]
        reps = {
            "extract_job_ids": [
                job.id for job in self.representative_jobs(run_id, JobType.EXTRACT)
            ],
            "verify_job_ids": [
                job.id for job in self.representative_jobs(run_id, JobType.VERIFY)
            ],
        }

        def totals(rows: list[dict[str, Any]]) -> dict[str, int]:
            return {
                key: sum(int(row.get(key) or 0) for row in rows)
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                )
            }

        return {
            "schema_version": 1,
            "plan_type": "EXTRACT_VERIFY_PAPER_PACKAGE_AB",
            "source_run_id": run_id,
            "expected_papers": expected_papers,
            "source_artifacts": sorted(
                sources.values(),
                key=lambda row: str(row["sha256"]),
            ),
            "historical_usage": {
                "extract": totals(extract_usage),
                "verify": totals(verify_usage),
            },
            "representative_jobs": reps,
        }


class PaperPackageABPreflight:
    def __init__(
        self,
        *,
        store: PostgresStore,
        artifact_root: str | Path,
        package_extractor: HybridPaperPackagePageTextExtractor,
    ):
        self.store = store
        self.planner = PaperPackageABPlanner(
            store=store,
            artifact_root=artifact_root,
        )
        self.pdf_hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
        )
        self.package_hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
            text_extractor=package_extractor,
        )
        self.package_extractor = package_extractor

    @staticmethod
    def _context(hydrated: ResearchJob) -> dict[str, Any]:
        return dict(hydrated.payload.get("_research_context") or {})

    @staticmethod
    def _quote_rows(
        *,
        pages: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_page = {
            int(page["page_number"]): str(page.get("text") or "")
            for page in pages
        }
        rows: list[dict[str, Any]] = []
        for item in evidence:
            page = int(item["page_number"])
            quote = str(item.get("raw_source_form") or "")
            context = by_page.get(page)
            rows.append(
                {
                    "evidence_id": str(item.get("id") or ""),
                    "page_number": page,
                    "page_present": context is not None,
                    "quote_token_recall": (
                        quote_token_recall(quote, context or "")
                        if context is not None
                        else 0.0
                    ),
                }
            )
        return rows

    def _historical_extract_evidence(self, job: ResearchJob) -> list[dict[str, Any]]:
        clean = _without_execution(job.result)
        return [dict(item) for item in clean.get("evidence", [])]

    def run(
        self,
        *,
        run_id: str,
        expected_papers: int = 9,
        quote_threshold: float = 0.8,
        required_quote_fraction: float = 0.95,
    ) -> dict[str, Any]:
        if not 0 <= quote_threshold <= 1:
            raise ValueError("quote_threshold must be in [0,1]")
        if not 0 <= required_quote_fraction <= 1:
            raise ValueError("required_quote_fraction must be in [0,1]")

        plan = self.planner.plan(
            run_id=run_id,
            expected_papers=expected_papers,
        )
        source_shas = [
            str(row["sha256"]) for row in plan["source_artifacts"]
        ]
        coverage = self.package_extractor.coverage(source_shas)
        if coverage["missing_native"] or coverage["missing_standard"]:
            raise ValueError(
                "paper-package coverage incomplete: "
                f"{coverage}"
            )

        rows: list[dict[str, Any]] = []
        for job_type in (JobType.EXTRACT, JobType.VERIFY):
            for job in self.planner.completed_jobs(run_id, job_type):
                pdf_job = self.pdf_hydrator.hydrate(job)
                package_job = self.package_hydrator.hydrate(job)
                pdf_context = self._context(pdf_job)
                package_context = self._context(package_job)
                pdf_pages = list(pdf_context.get("pages") or [])
                package_pages = list(package_context.get("pages") or [])
                if job_type == JobType.EXTRACT:
                    evidence = self._historical_extract_evidence(job)
                else:
                    evidence = [
                        dict(item)
                        for item in package_context.get(
                            "authoritative_evidence", []
                        )
                    ]
                quote_rows = self._quote_rows(
                    pages=package_pages,
                    evidence=evidence,
                )
                source = dict(package_context.get("source_artifact") or {})
                source_path = self.package_hydrator.artifact_store.path_for_hash(
                    str(source["sha256"])
                )
                rows.append(
                    {
                        "job_id": job.id,
                        "job_type": str(job_type),
                        "artifact_id": str(source.get("id") or ""),
                        "source_sha256": str(source.get("sha256") or ""),
                        "package": self.package_extractor.describe(source_path),
                        "pdf_context_pages": len(pdf_pages),
                        "package_context_pages": len(package_pages),
                        "pdf_context_chars": sum(
                            len(str(page.get("text") or ""))
                            for page in pdf_pages
                        ),
                        "package_context_chars": sum(
                            len(str(page.get("text") or ""))
                            for page in package_pages
                        ),
                        "evidence_count": len(evidence),
                        "quote_checks": quote_rows,
                    }
                )

        all_quotes = [
            check
            for row in rows
            for check in row["quote_checks"]
        ]
        passing_quotes = sum(
            check["page_present"]
            and float(check["quote_token_recall"]) >= quote_threshold
            for check in all_quotes
        )
        quote_fraction = (
            passing_quotes / len(all_quotes)
            if all_quotes
            else 1.0
        )
        pdf_chars = sum(int(row["pdf_context_chars"]) for row in rows)
        package_chars = sum(int(row["package_context_chars"]) for row in rows)
        reduction = (
            (pdf_chars - package_chars) / pdf_chars
            if pdf_chars > 0
            else None
        )
        decision = (
            "PASS"
            if quote_fraction >= required_quote_fraction
            else "FAIL_QUOTE_RECOVERY"
        )
        return {
            "schema_version": 1,
            "preflight_type": "PAPER_PACKAGE_CONTEXT_ZERO_MODEL",
            "source_run_id": run_id,
            "decision": decision,
            "quote_threshold": quote_threshold,
            "required_quote_fraction": required_quote_fraction,
            "package_coverage": coverage,
            "historical_evidence_quotes": len(all_quotes),
            "quotes_passing": passing_quotes,
            "quote_recovery_fraction": quote_fraction,
            "pdf_context_chars_total": pdf_chars,
            "package_context_chars_total": package_chars,
            "context_char_reduction_fraction": reduction,
            "rows": rows,
        }


def _greedy_extract_matches(
    historical: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, int]] = []
    for i, old in enumerate(historical):
        for j, new in enumerate(replay):
            score = token_f1(
                str(old.get("normalized_form") or ""),
                str(new.get("normalized_form") or ""),
            )
            scored.append((score, i, j))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_old: set[int] = set()
    used_new: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, i, j in scored:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        old = historical[i]
        new = replay[j]
        matches.append(
            {
                "historical_index": i,
                "replay_index": j,
                "normalized_f1": score,
                "raw_quote_f1": token_f1(
                    str(old.get("raw_source_form") or ""),
                    str(new.get("raw_source_form") or ""),
                ),
                "page_equal": int(old.get("page_number") or 0)
                == int(new.get("page_number") or 0),
                "type_equal": str(old.get("type") or "")
                == str(new.get("type") or ""),
            }
        )
    return matches


def extract_agreement(
    historical_result: dict[str, Any] | None,
    replay_result: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    historical = [
        dict(item)
        for item in _without_execution(historical_result).get("evidence", [])
    ]
    replay = [
        dict(item)
        for item in _without_execution(replay_result).get("evidence", [])
    ]
    matches = _greedy_extract_matches(historical, replay)
    passing = [m for m in matches if m["normalized_f1"] >= threshold]
    historical_count = len(historical)
    replay_count = len(replay)
    matched_f1 = (
        sum(float(m["normalized_f1"]) for m in matches) / len(matches)
        if matches
        else (1.0 if not historical and not replay else 0.0)
    )
    page_agreement = (
        sum(bool(m["page_equal"]) for m in passing) / len(passing)
        if passing
        else (1.0 if not historical and not replay else 0.0)
    )
    return {
        "historical_evidence_count": historical_count,
        "replay_evidence_count": replay_count,
        "matched_pairs": len(matches),
        "matched_normalized_f1": matched_f1,
        "historical_coverage_at_f1_0_5": (
            len(passing) / historical_count if historical_count else 1.0
        ),
        "replay_precision_at_f1_0_5": (
            len(passing) / replay_count if replay_count else (1.0 if not historical else 0.0)
        ),
        "page_agreement_on_passing_matches": page_agreement,
        "type_agreement_on_passing_matches": (
            sum(bool(m["type_equal"]) for m in passing) / len(passing)
            if passing
            else (1.0 if not historical and not replay else 0.0)
        ),
        "matches": matches,
    }


def verify_agreement(
    historical_result: dict[str, Any] | None,
    replay_result: dict[str, Any],
) -> dict[str, Any]:
    old = _without_execution(historical_result)
    new = _without_execution(replay_result)
    old_ids = sorted(str(x) for x in old.get("accepted_evidence_ids", []))
    new_ids = sorted(str(x) for x in new.get("accepted_evidence_ids", []))
    return {
        "verified_equal": bool(old.get("verified")) == bool(new.get("verified")),
        "accepted_evidence_ids_equal": old_ids == new_ids,
        "historical_accepted_count": len(old_ids),
        "replay_accepted_count": len(new_ids),
        "issues_equal": sorted(str(x) for x in old.get("issues", []))
        == sorted(str(x) for x in new.get("issues", [])),
    }


def agreement_pass(job_type: JobType, agreement: dict[str, Any]) -> bool:
    if job_type == JobType.EXTRACT:
        return (
            float(agreement["historical_coverage_at_f1_0_5"]) >= 0.9
            and float(agreement["replay_precision_at_f1_0_5"]) >= 0.9
            and float(agreement["page_agreement_on_passing_matches"]) >= 0.9
        )
    if job_type == JobType.VERIFY:
        return bool(
            agreement["verified_equal"]
            and agreement["accepted_evidence_ids_equal"]
        )
    raise ValueError(job_type)


class PaperPackageABReplay:
    def __init__(
        self,
        *,
        store: PostgresStore,
        artifact_root: str | Path,
        package_extractor: HybridPaperPackagePageTextExtractor,
        delegate: Any | None = None,
    ):
        self.store = store
        self.planner = PaperPackageABPlanner(
            store=store,
            artifact_root=artifact_root,
        )
        self.package_hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
            text_extractor=package_extractor,
        )
        self.pdf_hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
        )
        self.delegate = delegate or CodexExecExecutor(
            tier=ExecutorTier.STANDARD
        )

    def _record(
        self,
        *,
        calibration_type: str,
        source: ResearchJob,
        hydrated: ResearchJob,
        result: dict[str, Any],
        agreement: dict[str, Any],
        wall_time_ms: int,
        configuration: dict[str, Any],
    ) -> int:
        from psycopg.types.json import Jsonb

        historical = self.planner.usage(source.id)
        replay = _usage_from_result(result)
        context = dict(hydrated.payload.get("_research_context") or {})
        pages = list(context.get("pages") or [])
        evidence = list(context.get("authoritative_evidence") or [])
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_calibration_events(
                      calibration_type,source_run_id,source_job_id,
                      executor_tier,provider,model,reasoning_effort,code_sha,
                      configuration,context_page_count,context_char_count,
                      evidence_count,
                      historical_input_tokens,historical_cached_input_tokens,
                      historical_output_tokens,
                      replay_input_tokens,replay_cached_input_tokens,
                      replay_output_tokens,replay_reasoning_output_tokens,
                      wall_time_ms,outcome_agreement,result_summary
                    )
                    VALUES (
                      %s,%s,%s,'STANDARD',%s,%s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s
                    )
                    RETURNING id
                    """,
                    (
                        calibration_type,
                        source.run_id,
                        source.id,
                        str(replay.get("provider") or "unknown"),
                        replay.get("model"),
                        replay.get("reasoning_effort"),
                        _code_sha(),
                        Jsonb(configuration),
                        len(pages),
                        sum(len(str(page.get("text") or "")) for page in pages),
                        len(evidence),
                        int(historical["input_tokens"]),
                        int(historical["cached_input_tokens"]),
                        int(historical["output_tokens"]),
                        int(replay["input_tokens"]),
                        int(replay["cached_input_tokens"]),
                        int(replay["output_tokens"]),
                        int(replay["reasoning_output_tokens"]),
                        wall_time_ms,
                        Jsonb(agreement),
                        Jsonb(_without_execution(result)),
                    ),
                )
                event_id = int(cur.fetchone()[0])
            conn.commit()
        return event_id

    def _execute(
        self,
        *,
        source: ResearchJob,
        hydrator: ResearchContextHydrator,
        calibration_type: str,
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        hydrated = hydrator.hydrate(source)
        started = time.monotonic()
        result = self.delegate.execute(hydrated)
        wall_time_ms = max(0, int((time.monotonic() - started) * 1000))
        agreement = (
            extract_agreement(source.result, result)
            if source.job_type == JobType.EXTRACT
            else verify_agreement(source.result, result)
        )
        event_id = self._record(
            calibration_type=calibration_type,
            source=source,
            hydrated=hydrated,
            result=result,
            agreement=agreement,
            wall_time_ms=wall_time_ms,
            configuration=configuration,
        )
        context = dict(hydrated.payload.get("_research_context") or {})
        return {
            "calibration_event_id": event_id,
            "usage": _usage_from_result(result),
            "wall_time_ms": wall_time_ms,
            "context_page_count": len(context.get("pages") or []),
            "context_char_count": sum(
                len(str(page.get("text") or ""))
                for page in context.get("pages") or []
            ),
            "agreement": agreement,
            "agreement_pass": agreement_pass(source.job_type, agreement),
        }

    def run_representative(
        self,
        *,
        run_id: str,
        control_on_disagreement: bool = True,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for job_type in (JobType.EXTRACT, JobType.VERIFY):
            for source in self.planner.representative_jobs(run_id, job_type):
                historical = self.planner.usage(source.id)
                package = self._execute(
                    source=source,
                    hydrator=self.package_hydrator,
                    calibration_type=f"PAPER_PACKAGE_{str(job_type)}_AB",
                    configuration={
                        "arm": "paper_package",
                        "selection": "historical_input_low_median_high",
                    },
                )
                control = None
                if control_on_disagreement and not package["agreement_pass"]:
                    control = self._execute(
                        source=source,
                        hydrator=self.pdf_hydrator,
                        calibration_type=f"PAPER_PACKAGE_{str(job_type)}_AB_CONTROL_PDF",
                        configuration={
                            "arm": "current_pdf_control",
                            "trigger": "package_disagreement",
                        },
                    )
                if package["agreement_pass"]:
                    classification = "PACKAGE_MATCHES_HISTORICAL"
                elif control is not None and control["agreement_pass"]:
                    classification = "PACKAGE_CONTEXT_REGRESSION_SIGNAL"
                else:
                    classification = "MODEL_VARIANCE_AMBIGUOUS"

                package_input = int(package["usage"]["input_tokens"])
                old_input = int(historical["input_tokens"])
                rows.append(
                    {
                        "source_job_id": source.id,
                        "job_type": str(job_type),
                        "historical": historical,
                        "paper_package": package,
                        "pdf_control": control,
                        "classification": classification,
                        "input_reduction_fraction": (
                            (old_input - package_input) / old_input
                            if old_input > 0
                            else None
                        ),
                    }
                )

        old_input = sum(int(row["historical"]["input_tokens"]) for row in rows)
        package_input = sum(
            int(row["paper_package"]["usage"]["input_tokens"])
            for row in rows
        )
        package_pass = sum(
            bool(row["paper_package"]["agreement_pass"]) for row in rows
        )
        regression_signals = sum(
            row["classification"] == "PACKAGE_CONTEXT_REGRESSION_SIGNAL"
            for row in rows
        )
        return {
            "schema_version": 1,
            "replay_type": "PAPER_PACKAGE_REPRESENTATIVE_AB",
            "source_run_id": run_id,
            "jobs": len(rows),
            "package_agreement_passes": package_pass,
            "package_agreement_rate": package_pass / len(rows) if rows else 1.0,
            "package_context_regression_signals": regression_signals,
            "historical_input_tokens": old_input,
            "paper_package_input_tokens": package_input,
            "input_reduction_fraction": (
                (old_input - package_input) / old_input
                if old_input > 0
                else None
            ),
            "rows": rows,
        }


def planner_from_env() -> PaperPackageABPlanner:
    return PaperPackageABPlanner(
        store=PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"]),
        artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"],
    )


def package_extractor_from_paths(
    *,
    native_package_root: str | Path,
    standard_package_root: str | Path,
) -> HybridPaperPackagePageTextExtractor:
    return HybridPaperPackagePageTextExtractor(
        native_package_root=native_package_root,
        standard_package_root=standard_package_root,
    )
