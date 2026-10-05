from __future__ import annotations

import argparse
import copy
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from .codex_executor import CodexExecExecutor
from .orchestration import ExecutorTier, JobExecutionError, JobType, OrchestratorStore, ResearchJob
from .openai_executor import result_schema
from .paper_package_ab import (
    PaperPackageABPlanner,
    agreement_pass,
    extract_agreement,
    verify_agreement,
)
from .postgres import PostgresStore
from .semantic_pipeline import ResearchContextHydrator


CONTROL_TYPES = {
    JobType.EXTRACT: "PAPER_PACKAGE_EXTRACT_AB_CONTROL_PDF",
    JobType.VERIFY: "PAPER_PACKAGE_VERIFY_AB_CONTROL_PDF",
}


class SelectedPdfPageTextExtractor:
    """Extract only explicitly authorized 1-based PDF pages."""

    def __init__(self, page_numbers: list[int] | set[int]):
        normalized = sorted({int(x) for x in page_numbers})
        if not normalized or any(x <= 0 for x in normalized):
            raise ValueError("page_numbers must contain positive 1-based pages")
        self.page_numbers = normalized

    def __call__(self, path: Path) -> list[dict[str, Any]]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
        except Exception as exc:
            raise JobExecutionError(
                f"failed to parse selected PDF pages: {exc}",
                code="SOURCE_PARSE_FAILED",
                retryable=False,
                detail={"path": str(path), "page_numbers": self.page_numbers},
            ) from exc

        pages: list[dict[str, Any]] = []
        for page_number in self.page_numbers:
            if page_number > len(reader.pages):
                raise JobExecutionError(
                    "selected page is outside source PDF",
                    code="MISSING_VERIFICATION_SOURCE_PAGE",
                    retryable=False,
                    detail={
                        "path": str(path),
                        "page_number": page_number,
                        "pdf_page_count": len(reader.pages),
                    },
                )
            try:
                text = reader.pages[page_number - 1].extract_text() or ""
            except Exception as exc:
                raise JobExecutionError(
                    f"failed to extract selected PDF page {page_number}: {exc}",
                    code="SOURCE_PARSE_FAILED",
                    retryable=False,
                    detail={"path": str(path), "page_number": page_number},
                ) from exc
            pages.append({"page_number": page_number, "text": text})
        return pages


def _flexible_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    quote = str(quote or "").strip()
    if not quote:
        return None
    direct = text.find(quote)
    if direct >= 0:
        return direct, direct + len(quote)
    tokens = quote.split()
    if not tokens:
        return None
    match = re.search(
        r"\s+".join(re.escape(token) for token in tokens),
        text,
        flags=re.DOTALL,
    )
    return None if match is None else match.span()


def bounded_verification_page(
    *,
    page_text: str,
    authoritative_evidence: list[dict[str, Any]],
    max_chars: int,
) -> tuple[str | None, dict[str, Any]]:
    """Return a deterministic quote-preserving window for one VERIFY page."""

    text = str(page_text or "")
    if len(text) <= max_chars:
        return text, {
            "windowed": False,
            "full_page_char_count": len(text),
            "window_start_char": 0,
            "window_end_char": len(text),
            "window_char_count": len(text),
            "reason": "FULL_PAGE_WITHIN_BOUND",
        }

    spans: list[tuple[int, int]] = []
    missing: list[str] = []
    for item in authoritative_evidence:
        quote = str(item.get("raw_source_form") or "")
        span = _flexible_quote_span(text, quote)
        if span is None:
            missing.append(str(item.get("id") or "unknown"))
        else:
            spans.append(span)

    if missing or not spans:
        return None, {
            "windowed": True,
            "full_page_char_count": len(text),
            "reason": "AUTHORITATIVE_QUOTE_NOT_LOCATABLE",
            "missing_evidence_ids": missing,
        }

    anchor_start = min(start for start, _ in spans)
    anchor_end = max(end for _, end in spans)
    if anchor_end - anchor_start > max_chars:
        return None, {
            "windowed": True,
            "full_page_char_count": len(text),
            "reason": "AUTHORITATIVE_QUOTE_SPAN_EXCEEDS_BOUND",
            "anchor_span_chars": anchor_end - anchor_start,
        }

    spare = max_chars - (anchor_end - anchor_start)
    start = max(0, anchor_start - spare // 2)
    end = min(len(text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    window = text[start:end]

    missing_after = [
        str(item.get("id") or "unknown")
        for item in authoritative_evidence
        if _norm(str(item.get("raw_source_form") or "")) not in _norm(window)
    ]
    if missing_after:
        return None, {
            "windowed": True,
            "full_page_char_count": len(text),
            "reason": "WINDOW_DROPPED_AUTHORITATIVE_QUOTE",
            "missing_evidence_ids": missing_after,
        }

    return window, {
        "windowed": True,
        "full_page_char_count": len(text),
        "window_start_char": start,
        "window_end_char": end,
        "window_char_count": len(window),
        "reason": "AUTHORITATIVE_QUOTE_WINDOW",
    }


def _norm(value: str) -> str:
    return " ".join(str(value).split())


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def job_to_record(job: ResearchJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "run_id": job.run_id,
        "job_type": str(job.job_type),
        "payload": _json_safe(job.payload),
    }


def job_from_record(record: dict[str, Any]) -> ResearchJob:
    return ResearchJob(
        id=str(record["id"]),
        run_id=str(record["run_id"]),
        job_type=JobType(str(record["job_type"])),
        executor_tier=ExecutorTier.STANDARD,
        status="CALIBRATION",
        priority=0,
        idempotency_key=f"qwen35-guarded-calibration:{record['id']}",
        payload=copy.deepcopy(dict(record["payload"])),
        result=None,
        depends_on=[],
        attempt_count=0,
        max_attempts=1,
        lease_owner=None,
        lease_expires_at=None,
    )


def _guard_failure(job: ResearchJob, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": "FALLTHROUGH_CODEX",
        "job_id": job.id,
        "job_type": str(job.job_type),
        "reasons": reasons,
        "canonical_result": None,
        "authoritative_pdf_fallthrough_required": True,
    }


def guard_qwen_semantic_result(
    job: ResearchJob,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail-closed deterministic guard for bounded Qwen EXTRACT/VERIFY output."""

    if job.job_type not in {JobType.EXTRACT, JobType.VERIFY}:
        return _guard_failure(job, ["UNSUPPORTED_JOB_TYPE"])
    if not isinstance(result, dict):
        return _guard_failure(job, ["STRUCTURED_RESULT_MISSING"])

    try:
        validate(instance=result, schema=result_schema(job.job_type))
    except ValidationError as exc:
        return _guard_failure(
            job,
            [f"SCHEMA_INVALID:{exc.json_path or '$'}:{exc.message}"],
        )

    context = dict(job.payload.get("_research_context") or {})
    pages = [dict(page) for page in context.get("pages") or []]
    by_page = {
        int(page["page_number"]): str(page.get("text") or "")
        for page in pages
        if page.get("page_number") is not None
    }
    reasons: list[str] = []

    if job.job_type == JobType.EXTRACT:
        if not (1 <= len(pages) <= 4):
            reasons.append(f"EXTRACT_PAGE_BOUND_VIOLATION:{len(pages)}")

        evidence = [dict(item) for item in result.get("evidence") or []]
        ids = [str(item.get("id") or "") for item in evidence]
        if len(ids) != len(set(ids)):
            reasons.append("DUPLICATE_EVIDENCE_ID")

        for index, item in enumerate(evidence):
            page_number = int(item.get("page_number") or 0)
            raw = str(item.get("raw_source_form") or "").strip()
            normalized = str(item.get("normalized_form") or "").strip()
            confidence = item.get("confidence")
            page_text = by_page.get(page_number)

            if page_text is None:
                reasons.append(f"EVIDENCE_{index}:PAGE_OUTSIDE_CONTEXT:{page_number}")
                continue
            if not raw:
                reasons.append(f"EVIDENCE_{index}:EMPTY_RAW_SOURCE_FORM")
            elif _norm(raw) not in _norm(page_text):
                reasons.append(f"EVIDENCE_{index}:RAW_QUOTE_NOT_ON_CITED_PAGE")
            if not normalized:
                reasons.append(f"EVIDENCE_{index}:EMPTY_NORMALIZED_FORM")
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                reasons.append(f"EVIDENCE_{index}:INVALID_CONFIDENCE")

        canonical = {"evidence": evidence}

    else:
        scope = dict(context.get("verification_scope") or {})
        if int(scope.get("page_radius", -1)) != 0:
            reasons.append("VERIFY_RADIUS_NOT_ZERO")
        supplied = [int(x) for x in scope.get("supplied_page_numbers") or []]
        if len(supplied) != 1 or len(pages) != 1:
            reasons.append(
                f"VERIFY_EXACT_ONE_PAGE_REQUIRED:supplied={supplied}:pages={len(pages)}"
            )

        authoritative = [
            dict(item) for item in context.get("authoritative_evidence") or []
        ]
        known_ids = {str(item.get("id")) for item in authoritative}
        accepted = [str(x) for x in result.get("accepted_evidence_ids") or []]
        unknown = sorted(set(accepted) - known_ids)
        if unknown:
            reasons.append("UNKNOWN_ACCEPTED_EVIDENCE_IDS:" + ",".join(unknown))

        canonical = {
            "verified": bool(result["verified"]),
            "item_id": str(result["item_id"]),
            "accepted_evidence_ids": list(dict.fromkeys(accepted)),
            "issues": [str(x) for x in result.get("issues") or []],
        }

    if reasons:
        return _guard_failure(job, reasons)
    return {
        "status": "PASS_GUARDED",
        "job_id": job.id,
        "job_type": str(job.job_type),
        "reasons": [],
        "canonical_result": canonical,
        "authoritative_pdf_fallthrough_required": False,
    }


class GuardedQwenCalibrationPlanner:
    def __init__(
        self,
        *,
        store: PostgresStore,
        artifact_root: str | Path,
        max_context_chars: int = 20_000,
    ):
        self.store = store
        self.artifact_root = Path(artifact_root)
        self.queue = OrchestratorStore(store)
        self.planner = PaperPackageABPlanner(
            store=store,
            artifact_root=artifact_root,
        )
        self.hydrator = ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
            verify_page_radius=0,
        )
        self.max_context_chars = max_context_chars

    def _controlled_job_ids(
        self,
        run_id: str,
        job_type: JobType,
    ) -> list[str]:
        control_type = CONTROL_TYPES[job_type]
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT source_job_id
                    FROM research_calibration_events
                    WHERE source_run_id=%s
                      AND calibration_type=%s
                    ORDER BY source_job_id
                    """,
                    (run_id, control_type),
                )
                return [str(row[0]) for row in cur.fetchall()]

    @staticmethod
    def _context_chars(job: ResearchJob) -> int:
        context = dict(job.payload.get("_research_context") or {})
        return sum(
            len(str(page.get("text") or ""))
            for page in context.get("pages") or []
        )

    def _hydrator_for_pages(
        self,
        page_numbers: list[int] | set[int],
    ) -> ResearchContextHydrator:
        return ResearchContextHydrator(
            store=self.store,
            artifact_root=self.artifact_root,
            text_extractor=SelectedPdfPageTextExtractor(page_numbers),
            verify_page_radius=0,
        )

    def _bounded_extract(
        self, job: ResearchJob
    ) -> tuple[ResearchJob | None, str]:
        historical = dict(job.result or {})
        evidence = [dict(item) for item in historical.get("evidence") or []]
        pages = sorted(
            {
                int(item["page_number"])
                for item in evidence
                if item.get("page_number") is not None
            }
        )
        if not (1 <= len(pages) <= 4):
            return None, f"EXTRACT_PAGE_COUNT:{len(pages)}"

        hydrated = self._hydrator_for_pages(pages).hydrate(job)
        context = dict(hydrated.payload.get("_research_context") or {})
        selected = [dict(page) for page in context.get("pages") or []]
        if {int(page["page_number"]) for page in selected} != set(pages):
            return None, "EXTRACT_SELECTED_PAGE_MISMATCH"
        context["bounded_calibration_scope"] = {
            "selection_basis": "historical_evidence_page_numbers_not_answer_content",
            "page_numbers": pages,
            "max_pages": 4,
            "source_extraction": "SELECTED_PAGES_ONLY",
        }
        payload = copy.deepcopy(hydrated.payload)
        payload["_research_context"] = context
        bounded = replace(hydrated, payload=payload)
        chars = self._context_chars(bounded)
        if chars > self.max_context_chars:
            return None, f"EXTRACT_CONTEXT_CHARS:{chars}"
        return bounded, "PASS"

    def _bounded_verify(
        self, job: ResearchJob
    ) -> tuple[ResearchJob | None, str]:
        materialized = (
            (job.payload.get("extraction") or {})
            .get("_materialization", {})
        )
        evidence_ids = list(materialized.get("evidence_ids") or [])
        if not evidence_ids:
            return None, "VERIFY_NO_EVIDENCE_IDS"

        authoritative = self.hydrator._evidence(evidence_ids)
        cited_pages = sorted(
            {
                int(item["page_number"])
                for item in authoritative
                if item.get("page_number") is not None
            }
        )
        if len(cited_pages) != 1:
            return None, f"VERIFY_CITED_PAGE_COUNT:{len(cited_pages)}"

        hydrated = self._hydrator_for_pages(cited_pages).hydrate(job)
        payload = copy.deepcopy(hydrated.payload)
        context = dict(payload.get("_research_context") or {})
        pages = [dict(page) for page in context.get("pages") or []]
        scope = dict(context.get("verification_scope") or {})
        if int(scope.get("page_radius", -1)) != 0:
            return None, "VERIFY_RADIUS_NOT_ZERO"
        if len(pages) != 1 or len(scope.get("supplied_page_numbers") or []) != 1:
            return None, (
                "VERIFY_SINGLE_PAGE_REQUIRED:"
                f"pages={len(pages)}:"
                f"supplied={scope.get('supplied_page_numbers') or []}"
            )

        window, window_meta = bounded_verification_page(
            page_text=str(pages[0].get("text") or ""),
            authoritative_evidence=[
                dict(item)
                for item in context.get("authoritative_evidence") or []
            ],
            max_chars=self.max_context_chars,
        )
        if window is None:
            return None, f"VERIFY_WINDOW:{window_meta.get('reason')}"

        pages[0]["text"] = window
        context["pages"] = pages
        scope["text_window"] = window_meta
        scope["source_extraction"] = "SELECTED_PAGE_ONLY"
        context["verification_scope"] = scope
        payload["_research_context"] = context
        bounded = replace(hydrated, payload=payload)
        chars = self._context_chars(bounded)
        if chars > self.max_context_chars:
            return None, f"VERIFY_CONTEXT_CHARS:{chars}"
        return bounded, "PASS"

    def build(self, *, run_id: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for job_type in (JobType.EXTRACT, JobType.VERIFY):
            controlled = self._controlled_job_ids(run_id, job_type)
            if not controlled:
                raise ValueError(
                    f"no preserved current-PDF control jobs for {job_type}"
                )
            by_id = {
                job.id: job
                for job in self.planner.completed_jobs(run_id, job_type)
            }
            selected: ResearchJob | None = None
            source_id: str | None = None
            rejections: list[dict[str, str]] = []
            for job_id in controlled:
                source = by_id.get(job_id)
                if source is None:
                    rejections.append(
                        {"job_id": job_id, "reason": "SOURCE_JOB_NOT_COMPLETED"}
                    )
                    continue
                candidate, reason = (
                    self._bounded_extract(source)
                    if job_type == JobType.EXTRACT
                    else self._bounded_verify(source)
                )
                if candidate is not None:
                    selected = candidate
                    source_id = job_id
                    break
                rejections.append({"job_id": job_id, "reason": reason})
            if selected is None:
                raise ValueError(
                    f"no preserved {job_type} control fits bounded Qwen context; "
                    f"rejections={json.dumps(rejections, sort_keys=True)}"
                )
            context = dict(selected.payload.get("_research_context") or {})
            records.append(
                {
                    "job": job_to_record(selected),
                    "source_job_id": source_id,
                    "context_page_numbers": [
                        int(page["page_number"])
                        for page in context.get("pages") or []
                    ],
                    "context_chars": self._context_chars(selected),
                    "selection": "preserved_current_pdf_control_ambiguity",
                }
            )

        return {
            "schema_version": 1,
            "operation": "QWEN35_GUARDED_EXTRACT_VERIFY_CALIBRATION",
            "source_run_id": run_id,
            "jobs": records,
            "qwen_call_budget": 2,
            "job_types": [str(JobType.EXTRACT), str(JobType.VERIFY)],
            "production_routing_changed": False,
            "scientific_state_materialized": False,
        }


def run_codex_controls(plan: dict[str, Any]) -> dict[str, Any]:
    delegate = CodexExecExecutor(tier=ExecutorTier.STANDARD)
    rows: list[dict[str, Any]] = []
    for row in plan["jobs"]:
        job = job_from_record(dict(row["job"]))
        result = delegate.execute(job)
        rows.append(
            {
                "job_id": job.id,
                "job_type": str(job.job_type),
                "result": _json_safe(result),
            }
        )
    return {
        "schema_version": 1,
        "control": "CONTEMPORANEOUS_STANDARD_CODEX_AUTHORITATIVE_PDF_CONTEXT",
        "source_run_id": plan["source_run_id"],
        "jobs": rows,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
    }


def evaluate_calibration(
    plan: dict[str, Any],
    qwen: dict[str, Any],
    codex: dict[str, Any],
) -> dict[str, Any]:
    qwen_by_id = {str(row["job_id"]): row for row in qwen.get("jobs") or []}
    codex_by_id = {str(row["job_id"]): row for row in codex.get("jobs") or []}
    rows: list[dict[str, Any]] = []

    for planned in plan["jobs"]:
        job = job_from_record(dict(planned["job"]))
        qrow = qwen_by_id.get(job.id)
        crow = codex_by_id.get(job.id)
        raw_qwen = (
            dict(qrow.get("parsed") or {})
            if isinstance(qrow, dict)
            else None
        )
        guard = guard_qwen_semantic_result(job, raw_qwen)
        control_result = (
            dict(crow.get("result") or {})
            if isinstance(crow, dict)
            else None
        )

        agreement = None
        agrees = False
        if guard["status"] == "PASS_GUARDED" and control_result is not None:
            local = dict(guard["canonical_result"])
            agreement = (
                extract_agreement(control_result, local)
                if job.job_type == JobType.EXTRACT
                else verify_agreement(control_result, local)
            )
            agrees = agreement_pass(job.job_type, agreement)

        rows.append(
            {
                "job_id": job.id,
                "job_type": str(job.job_type),
                "context_page_numbers": planned["context_page_numbers"],
                "context_chars": planned["context_chars"],
                "guard": guard,
                "codex_control_present": control_result is not None,
                "qwen_vs_codex_agreement": agreement,
                "qwen_vs_codex_agreement_pass": agrees,
                "fallthrough_required": (
                    guard["status"] != "PASS_GUARDED" or not agrees
                ),
            }
        )

    all_pass = (
        len(rows) == 2
        and {row["job_type"] for row in rows}
        == {str(JobType.EXTRACT), str(JobType.VERIFY)}
        and all(
            row["guard"]["status"] == "PASS_GUARDED"
            and row["qwen_vs_codex_agreement_pass"]
            for row in rows
        )
    )
    return {
        "schema_version": 1,
        "operation": "QWEN35_GUARDED_EXTRACT_VERIFY_CALIBRATION",
        "source_run_id": plan["source_run_id"],
        "decision": (
            "PROMOTION_CANDIDATE_GUARDED_BOUNDED_SEMANTICS"
            if all_pass
            else "FAIL_CLOSED_RETAIN_CODEX"
        ),
        "jobs": rows,
        "guarded_jobs_passed": sum(
            row["guard"]["status"] == "PASS_GUARDED" for row in rows
        ),
        "codex_agreement_jobs_passed": sum(
            bool(row["qwen_vs_codex_agreement_pass"]) for row in rows
        ),
        "production_routing_changed": False,
        "scientific_state_materialized": False,
    }


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan")
    plan_p.add_argument("run_id")
    plan_p.add_argument("--output", required=True)
    plan_p.add_argument("--max-context-chars", type=int, default=20_000)

    control_p = sub.add_parser("control")
    control_p.add_argument("--plan", required=True)
    control_p.add_argument("--output", required=True)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--plan", required=True)
    eval_p.add_argument("--qwen", required=True)
    eval_p.add_argument("--codex", required=True)
    eval_p.add_argument("--output", required=True)

    args = p.parse_args()

    if args.command == "plan":
        planner = GuardedQwenCalibrationPlanner(
            store=PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"]),
            artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"],
            max_context_chars=args.max_context_chars,
        )
        payload = planner.build(run_id=args.run_id)
    elif args.command == "control":
        payload = run_codex_controls(_read(args.plan))
    else:
        payload = evaluate_calibration(
            _read(args.plan),
            _read(args.qwen),
            _read(args.codex),
        )

    _write(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
