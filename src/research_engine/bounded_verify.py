from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from typing import Any

from .local_inference import (
    LocalInferenceBackpressureError,
    LocalInferenceError,
    LocalInferenceService,
)
from .openai_executor import COMMON_INSTRUCTIONS, JOB_INSTRUCTIONS, result_schema
from .orchestration import ExecutorTier, JobType, ResearchJob
from .qwen_guarded_semantic import bounded_verification_page, guard_qwen_semantic_result


QWEN_VERIFY_MODEL = "Qwen/Qwen3.5-4B"
QWEN_VERIFY_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_VERIFY_RUNTIME_CONTRACT = "qwen35-vllm-cu129-image-native-sampler-v3"
DEFAULT_MAX_CONTEXT_CHARS = 20_000


def qwen_verify_enabled_from_env() -> bool:
    value = os.getenv("RESEARCH_ENGINE_QWEN_VERIFY_ENABLED", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("RESEARCH_ENGINE_QWEN_VERIFY_ENABLED must be 0 or 1")
    return value == "1"


def qwen_verify_max_context_chars_from_env() -> int:
    value = int(
        os.getenv(
            "RESEARCH_ENGINE_QWEN_VERIFY_MAX_CONTEXT_CHARS",
            str(DEFAULT_MAX_CONTEXT_CHARS),
        )
    )
    if value <= 0:
        raise ValueError(
            "RESEARCH_ENGINE_QWEN_VERIFY_MAX_CONTEXT_CHARS must be positive"
        )
    return value


def prepare_bounded_verify_pages(
    job: ResearchJob,
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[list[tuple[int, ResearchJob, list[str]]] | None, str]:
    """Split one hydrated radius-0 VERIFY job into calibrated page-scoped units."""

    if job.job_type != JobType.VERIFY:
        return None, "NOT_VERIFY"

    context = copy.deepcopy(dict(job.payload.get("_research_context") or {}))
    scope = dict(context.get("verification_scope") or {})
    if int(scope.get("page_radius", -1)) != 0:
        return None, "VERIFY_RADIUS_NOT_ZERO"

    pages = [dict(page) for page in context.get("pages") or []]
    supplied = [int(x) for x in scope.get("supplied_page_numbers") or []]
    if not pages or not supplied:
        return None, "VERIFY_NO_SUPPLIED_PAGES"

    by_page_text: dict[int, dict[str, Any]] = {}
    for page in pages:
        page_number = int(page.get("page_number") or 0)
        if page_number <= 0 or page_number in by_page_text:
            return None, "VERIFY_INVALID_OR_DUPLICATE_PAGE"
        by_page_text[page_number] = page

    authoritative = [
        dict(item) for item in context.get("authoritative_evidence") or []
    ]
    if not authoritative:
        return None, "VERIFY_NO_AUTHORITATIVE_EVIDENCE"

    by_page_evidence: dict[int, list[dict[str, Any]]] = {}
    evidence_ids_seen: set[str] = set()
    for item in authoritative:
        evidence_id = str(item.get("id") or "")
        cited = item.get("page_number")
        if not evidence_id:
            return None, "VERIFY_EVIDENCE_MISSING_ID"
        if evidence_id in evidence_ids_seen:
            return None, "VERIFY_DUPLICATE_EVIDENCE_ID"
        evidence_ids_seen.add(evidence_id)
        if cited is None:
            return None, "VERIFY_EVIDENCE_MISSING_PAGE"
        by_page_evidence.setdefault(int(cited), []).append(item)

    expected_pages = set(by_page_evidence)
    if set(supplied) != expected_pages or set(by_page_text) != expected_pages:
        return None, (
            "VERIFY_PAGE_SCOPE_MISMATCH:"
            f"supplied={sorted(set(supplied))}:"
            f"text={sorted(by_page_text)}:"
            f"evidence={sorted(expected_pages)}"
        )

    bounded_units: list[tuple[int, ResearchJob, list[str]]] = []
    page_count = len(expected_pages)
    for page_number in sorted(expected_pages):
        page_evidence = by_page_evidence[page_number]
        window, window_meta = bounded_verification_page(
            page_text=str(by_page_text[page_number].get("text") or ""),
            authoritative_evidence=page_evidence,
            max_chars=max_context_chars,
        )
        if window is None:
            return None, (
                f"VERIFY_PAGE_{page_number}_WINDOW:"
                + str(window_meta.get("reason") or "UNKNOWN")
            )

        page = copy.deepcopy(by_page_text[page_number])
        page["text"] = window
        unit_context = copy.deepcopy(context)
        unit_context["pages"] = [page]
        unit_context["authoritative_evidence"] = copy.deepcopy(page_evidence)
        unit_context["verification_scope"] = {
            "cited_page_numbers": [page_number],
            "page_radius": 0,
            "supplied_page_numbers": [page_number],
            "text_window": window_meta,
            "bounded_local_verify": {
                "model": QWEN_VERIFY_MODEL,
                "revision": QWEN_VERIFY_REVISION,
                "runtime_contract": QWEN_VERIFY_RUNTIME_CONTRACT,
                "max_context_chars": max_context_chars,
                "eligibility": "RADIUS_0_PAGE_SCOPED_AUTHORITATIVE_EVIDENCE",
                "source_job_page_count": page_count,
                "selected_page_number": page_number,
            },
        }
        payload = copy.deepcopy(job.payload)
        payload["_research_context"] = unit_context
        evidence_ids = [str(item["id"]) for item in page_evidence]
        bounded_units.append(
            (page_number, replace(job, payload=payload), evidence_ids)
        )

    return bounded_units, "PASS"


def prepare_bounded_verify_job(
    job: ResearchJob,
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[ResearchJob | None, str]:
    """Backward-compatible helper for callers requiring exactly one page unit."""

    units, reason = prepare_bounded_verify_pages(
        job,
        max_context_chars=max_context_chars,
    )
    if units is None:
        return None, reason
    if len(units) != 1:
        return None, (
            "VERIFY_EXACT_ONE_PAGE_REQUIRED:"
            + ",".join(str(page) for page, _job, _ids in units)
        )
    return units[0][1], "PASS"


def qwen_verify_messages(job: ResearchJob) -> list[dict[str, str]]:
    if job.job_type != JobType.VERIFY:
        raise ValueError("Qwen bounded VERIFY prompt requires a VERIFY job")
    system = (
        COMMON_INSTRUCTIONS
        + "\n"
        + JOB_INSTRUCTIONS[JobType.VERIFY]
        + "\nThis is a bounded local VERIFY operation. Use ONLY "
        "payload._research_context. accepted_evidence_ids must come only from "
        "authoritative_evidence. If uncertain, return no accepted IDs and "
        "explicit issues rather than guessing. Return exactly one JSON object "
        "matching the supplied schema."
    )
    user = json.dumps(
        {
            "job_id": job.id,
            "job_type": str(job.job_type),
            "payload": job.payload,
            "output_schema": result_schema(JobType.VERIFY),
        },
        sort_keys=True,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class BoundedVerifySemanticDelegate:
    """Page-scoped Qwen-primary VERIFY with whole-job fail-closed fallthrough."""

    def __init__(
        self,
        *,
        tier: ExecutorTier,
        fallback,
        local_inference: LocalInferenceService | None,
        enabled: bool | None = None,
        max_context_chars: int | None = None,
    ):
        self.tier = tier
        self.fallback = fallback
        self.local_inference = local_inference
        self.enabled = (
            qwen_verify_enabled_from_env() if enabled is None else bool(enabled)
        )
        self.max_context_chars = (
            qwen_verify_max_context_chars_from_env()
            if max_context_chars is None
            else int(max_context_chars)
        )
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")

    def _fallback(
        self,
        job: ResearchJob,
        *,
        reason: str,
        local_attempted: bool,
        local_detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.fallback.execute(job)
        execution = dict(result.get("_execution") or {})
        execution["routing"] = {
            "policy": "QWEN35_PAGE_SCOPED_VERIFY_PRIMARY_CODEX_FALLTHROUGH",
            "decision": "FALLTHROUGH",
            "reason": reason,
            "local_attempted": bool(local_attempted),
            "local_detail": copy.deepcopy(local_detail or {}),
        }
        result["_execution"] = execution
        return result

    def execute(self, job: ResearchJob) -> dict[str, Any]:
        if job.executor_tier != self.tier:
            return self._fallback(
                job,
                reason="TIER_MISMATCH_FOR_LOCAL_VERIFY",
                local_attempted=False,
            )
        if not self.enabled:
            return self._fallback(
                job,
                reason="QWEN_VERIFY_DISABLED",
                local_attempted=False,
            )
        if job.job_type != JobType.VERIFY:
            return self._fallback(
                job,
                reason="JOB_TYPE_NOT_BOUNDED_VERIFY",
                local_attempted=False,
            )
        if self.tier != ExecutorTier.STANDARD:
            return self._fallback(
                job,
                reason="VERIFY_TIER_NOT_STANDARD",
                local_attempted=False,
            )
        if self.local_inference is None or self.local_inference.qwen_verify is None:
            return self._fallback(
                job,
                reason="QWEN_VERIFY_OWNER_UNAVAILABLE",
                local_attempted=False,
            )

        units, reason = prepare_bounded_verify_pages(
            job,
            max_context_chars=self.max_context_chars,
        )
        if units is None:
            return self._fallback(
                job,
                reason=reason,
                local_attempted=False,
            )

        all_authoritative = [
            dict(item)
            for item in (
                job.payload.get("_research_context") or {}
            ).get("authoritative_evidence") or []
        ]
        authoritative_order = [str(item["id"]) for item in all_authoritative]
        expected_item_id = str((job.payload.get("item") or {}).get("id") or "")
        if not expected_item_id:
            return self._fallback(
                job,
                reason="VERIFY_ITEM_ID_MISSING",
                local_attempted=False,
            )
        total_input_tokens = 0
        total_output_tokens = 0
        page_receipts: list[dict[str, Any]] = []

        for page_number, bounded, expected_ids in units:
            try:
                response = self.local_inference.qwen_verify_request(
                    messages=qwen_verify_messages(bounded),
                    job_id=f"{job.id}:page:{page_number}",
                )
            except LocalInferenceBackpressureError as exc:
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_BACKPRESSURE",
                    local_attempted=True,
                    local_detail={
                        "error": str(exc),
                        "completed_pages": page_receipts,
                    },
                )
            except LocalInferenceError as exc:
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_RUNTIME_FAILURE",
                    local_attempted=True,
                    local_detail={
                        "error": str(exc),
                        "completed_pages": page_receipts,
                    },
                )

            parsed = response.get("result")
            guard = guard_qwen_semantic_result(
                bounded,
                parsed if isinstance(parsed, dict) else None,
            )
            if guard["status"] != "PASS_GUARDED":
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_GUARD_FAILURE",
                    local_attempted=True,
                    local_detail={
                        "guard_reasons": list(guard.get("reasons") or []),
                        "completed_pages": page_receipts,
                        "server_pid": response.get("server_pid"),
                        "request_seq": response.get("request_seq"),
                    },
                )

            canonical = copy.deepcopy(dict(guard["canonical_result"]))
            accepted = [str(x) for x in canonical["accepted_evidence_ids"]]
            issues = [str(x) for x in canonical["issues"]]
            local_item_id = str(canonical["item_id"])
            if expected_item_id and local_item_id != expected_item_id:
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_ITEM_ID_MISMATCH",
                    local_attempted=True,
                    local_detail={
                        "expected_item_id": expected_item_id,
                        "actual_item_id": local_item_id,
                        "completed_pages": page_receipts,
                    },
                )
            if not bool(canonical["verified"]):
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_NOT_VERIFIED",
                    local_attempted=True,
                    local_detail={
                        "issues": issues,
                        "accepted_evidence_ids": accepted,
                        "completed_pages": page_receipts,
                    },
                )
            if issues:
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_ISSUES_PRESENT",
                    local_attempted=True,
                    local_detail={
                        "issues": issues,
                        "accepted_evidence_ids": accepted,
                        "completed_pages": page_receipts,
                    },
                )
            if set(accepted) != set(expected_ids):
                return self._fallback(
                    job,
                    reason=f"QWEN_VERIFY_PAGE_{page_number}_INCOMPLETE_ACCEPTANCE",
                    local_attempted=True,
                    local_detail={
                        "expected_evidence_ids": expected_ids,
                        "accepted_evidence_ids": accepted,
                        "completed_pages": page_receipts,
                    },
                )

            usage = dict(response.get("usage") or {})
            total_input_tokens += int(usage.get("input_tokens") or 0)
            total_output_tokens += int(usage.get("output_tokens") or 0)
            page_receipts.append(
                {
                    "page_number": page_number,
                    "accepted_evidence_ids": expected_ids,
                    "server_pid": response.get("server_pid"),
                    "request_seq": response.get("request_seq"),
                    "wall_time_ms": response.get("wall_time_ms"),
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                }
            )

        result = {
            "verified": True,
            "item_id": expected_item_id,
            "accepted_evidence_ids": authoritative_order,
            "issues": [],
        }
        result["_execution"] = {
            "provider": "local-qwen-vllm",
            "model": f"{QWEN_VERIFY_MODEL}@{QWEN_VERIFY_REVISION}",
            "response_id": f"page-scoped:{job.id}",
            "executor_tier": str(self.tier),
            "reasoning_effort": "none",
            "usage": {
                "input_tokens": total_input_tokens,
                "cached_input_tokens": 0,
                "output_tokens": total_output_tokens,
                "reasoning_output_tokens": 0,
            },
            "routing": {
                "policy": "QWEN35_PAGE_SCOPED_VERIFY_PRIMARY_CODEX_FALLTHROUGH",
                "decision": "LOCAL_QWEN_ACCEPTED",
                "reason": "ALL_PAGE_UNITS_PASS_GUARDED_FULL_ACCEPTANCE",
                "local_attempted": True,
                "page_count": len(units),
                "qwen_calls": len(units),
                "guard_reasons": [],
            },
            "local_inference": {
                "family": "qwen_verify",
                "runtime_contract": QWEN_VERIFY_RUNTIME_CONTRACT,
                "model_revision": QWEN_VERIFY_REVISION,
                "page_receipts": page_receipts,
            },
        }
        return result
