from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .codex_executor import CodexExecExecutor
from .orchestration import ExecutorTier, JobType, OrchestratorStore, ResearchJob
from .postgres import PostgresStore
from .semantic_pipeline import ResearchContextHydrator


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
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _without_execution(result: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(result or {}).items()
        if key != "_execution"
    }


class VerifyReplayCalibration:
    """Operational A/B replay for completed VERIFY jobs.

    The source scientific job remains immutable. The replay invokes the semantic
    delegate directly and records only an operational calibration event.
    """

    def __init__(
        self,
        *,
        store: PostgresStore,
        artifact_root: str | Path,
        delegate: Any | None = None,
        verify_page_radius: int | None = None,
    ):
        self.store = store
        self.queue = OrchestratorStore(store)
        radius = (
            int(os.getenv("RESEARCH_ENGINE_VERIFY_PAGE_RADIUS", "0"))
            if verify_page_radius is None
            else int(verify_page_radius)
        )
        self.verify_page_radius = radius
        self.hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
            verify_page_radius=radius,
        )
        self.delegate = delegate or CodexExecExecutor(tier=ExecutorTier.STANDARD)

    def _verify_jobs(self, run_id: str) -> list[ResearchJob]:
        return [
            job
            for job in self.queue.jobs(run_id, JobType.VERIFY)
            if job.status == "COMPLETED"
        ]

    def _usage(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT provider,model,reasoning_effort,
                           input_tokens,cached_input_tokens,
                           output_tokens,reasoning_output_tokens
                    FROM research_usage_events
                    WHERE job_id=%s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"source VERIFY job has no usage event: {job_id}")
        return {
            "provider": row[0],
            "model": row[1],
            "reasoning_effort": row[2],
            "input_tokens": int(row[3]),
            "cached_input_tokens": int(row[4]),
            "output_tokens": int(row[5]),
            "reasoning_output_tokens": int(row[6]),
        }

    def select_source_job(
        self,
        *,
        run_id: str,
        source_job_id: str | None = None,
    ) -> ResearchJob:
        jobs = self._verify_jobs(run_id)
        if not jobs:
            raise ValueError(f"run has no completed VERIFY jobs: {run_id}")
        by_id = {job.id: job for job in jobs}
        if source_job_id is not None:
            try:
                return by_id[source_job_id]
            except KeyError as exc:
                raise ValueError(
                    f"completed VERIFY job not found in run {run_id}: {source_job_id}"
                ) from exc

        ranked = sorted(
            ((self._usage(job.id)["input_tokens"], job.id, job) for job in jobs),
            key=lambda item: (item[0], item[1]),
        )
        return ranked[len(ranked) // 2][2]

    @staticmethod
    def _agreement(
        historical: dict[str, Any] | None,
        replay: dict[str, Any],
    ) -> dict[str, Any]:
        historical = _without_execution(historical)
        replay_clean = _without_execution(replay)
        old_ids = sorted(
            str(value)
            for value in historical.get("accepted_evidence_ids", [])
        )
        new_ids = sorted(
            str(value)
            for value in replay_clean.get("accepted_evidence_ids", [])
        )
        return {
            "verified_equal": (
                bool(historical.get("verified"))
                == bool(replay_clean.get("verified"))
            ),
            "accepted_evidence_ids_equal": old_ids == new_ids,
            "historical_accepted_count": len(old_ids),
            "replay_accepted_count": len(new_ids),
            "historical_verified": bool(historical.get("verified")),
            "replay_verified": bool(replay_clean.get("verified")),
        }

    def run(
        self,
        *,
        run_id: str,
        source_job_id: str | None = None,
    ) -> dict[str, Any]:
        source = self.select_source_job(
            run_id=run_id,
            source_job_id=source_job_id,
        )
        historical_usage = self._usage(source.id)
        hydrated = self.hydrator.hydrate(source)
        context = dict(hydrated.payload.get("_research_context") or {})
        pages = list(context.get("pages") or [])
        evidence = list(context.get("authoritative_evidence") or [])
        context_chars = sum(len(str(page.get("text") or "")) for page in pages)

        started = time.monotonic()
        replay = self.delegate.execute(hydrated)
        wall_time_ms = max(0, int((time.monotonic() - started) * 1000))

        execution = dict(replay.get("_execution") or {})
        replay_usage = dict(execution.get("usage") or {})
        agreement = self._agreement(source.result, replay)

        historical_input = int(historical_usage["input_tokens"])
        replay_input = int(replay_usage.get("input_tokens") or 0)
        historical_cached = int(historical_usage["cached_input_tokens"])
        replay_cached = int(replay_usage.get("cached_input_tokens") or 0)
        historical_uncached = max(0, historical_input - historical_cached)
        replay_uncached = max(0, replay_input - replay_cached)

        def reduction(old: int, new: int) -> float | None:
            return None if old <= 0 else (old - new) / old

        summary = {
            "calibration_type": "VERIFY_CONTEXT_WINDOW",
            "source_run_id": run_id,
            "source_job_id": source.id,
            "verify_page_radius": self.verify_page_radius,
            "context_page_count": len(pages),
            "context_page_numbers": [
                int(page["page_number"]) for page in pages
            ],
            "context_char_count": context_chars,
            "evidence_count": len(evidence),
            "historical": historical_usage,
            "replay": {
                "provider": execution.get("provider"),
                "model": execution.get("model"),
                "reasoning_effort": execution.get("reasoning_effort"),
                "input_tokens": replay_input,
                "cached_input_tokens": replay_cached,
                "output_tokens": int(replay_usage.get("output_tokens") or 0),
                "reasoning_output_tokens": int(
                    replay_usage.get("reasoning_output_tokens") or 0
                ),
                "wall_time_ms": wall_time_ms,
            },
            "input_reduction_fraction": reduction(
                historical_input,
                replay_input,
            ),
            "uncached_input_reduction_fraction": reduction(
                historical_uncached,
                replay_uncached,
            ),
            "agreement": agreement,
        }

        from psycopg.types.json import Jsonb

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
                      'VERIFY_CONTEXT_WINDOW',%s,%s,
                      'STANDARD',%s,%s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s
                    )
                    RETURNING id,created_at
                    """,
                    (
                        run_id,
                        source.id,
                        str(execution.get("provider") or "unknown"),
                        execution.get("model"),
                        execution.get("reasoning_effort"),
                        _code_sha(),
                        Jsonb(
                            {
                                "verify_page_radius": self.verify_page_radius,
                                "context_page_numbers": summary[
                                    "context_page_numbers"
                                ],
                            }
                        ),
                        len(pages),
                        context_chars,
                        len(evidence),
                        historical_input,
                        historical_cached,
                        int(historical_usage["output_tokens"]),
                        replay_input,
                        replay_cached,
                        int(replay_usage.get("output_tokens") or 0),
                        int(
                            replay_usage.get("reasoning_output_tokens") or 0
                        ),
                        wall_time_ms,
                        Jsonb(agreement),
                        Jsonb(
                            {
                                "verified": bool(
                                    _without_execution(replay).get("verified")
                                ),
                                "accepted_evidence_count": len(
                                    _without_execution(replay).get(
                                        "accepted_evidence_ids",
                                        [],
                                    )
                                ),
                                "issues": list(
                                    _without_execution(replay).get("issues", [])
                                ),
                            }
                        ),
                    ),
                )
                event_id, created_at = cur.fetchone()
            conn.commit()

        summary["calibration_event_id"] = int(event_id)
        summary["created_at"] = created_at.isoformat()
        return summary


def service_from_env(*, delegate: Any | None = None) -> VerifyReplayCalibration:
    return VerifyReplayCalibration(
        store=PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"]),
        artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"],
        delegate=delegate,
    )
