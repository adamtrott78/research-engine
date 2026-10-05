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


def prepare_bounded_verify_job(
    job: ResearchJob,
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[ResearchJob | None, str]:
    """Return the exact calibrated one-page VERIFY unit or a fallthrough reason."""

    if job.job_type != JobType.VERIFY:
        return None, "NOT_VERIFY"

    context = copy.deepcopy(dict(job.payload.get("_research_context") or {}))
    scope = dict(context.get("verification_scope") or {})
    if int(scope.get("page_radius", -1)) != 0:
        return None, "VERIFY_RADIUS_NOT_ZERO"

    pages = [dict(page) for page in context.get("pages") or []]
    supplied = [int(x) for x in scope.get("supplied_page_numbers") or []]
    if len(pages) != 1 or len(supplied) != 1:
        return None, (
            "VERIFY_EXACT_ONE_PAGE_REQUIRED:"
            f"supplied={supplied}:pages={len(pages)}"
        )

    page_number = int(pages[0].get("page_number") or 0)
    if page_number <= 0 or supplied != [page_number]:
        return None, "VERIFY_SUPPLIED_PAGE_MISMATCH"

    authoritative = [
        dict(item) for item in context.get("authoritative_evidence") or []
    ]
    if not authoritative:
        return None, "VERIFY_NO_AUTHORITATIVE_EVIDENCE"

    evidence_pages: set[int] = set()
    for item in authoritative:
        cited = item.get("page_number")
        if cited is None:
            return None, "VERIFY_EVIDENCE_MISSING_PAGE"
        evidence_pages.add(int(cited))
    if evidence_pages != {page_number}:
        return None, (
            "VERIFY_EVIDENCE_SPANS_MULTIPLE_PAGES:"
            + ",".join(str(x) for x in sorted(evidence_pages))
        )

    window, window_meta = bounded_verification_page(
        page_text=str(pages[0].get("text") or ""),
        authoritative_evidence=authoritative,
        max_chars=max_context_chars,
    )
    if window is None:
        return None, "VERIFY_WINDOW:" + str(window_meta.get("reason") or "UNKNOWN")

    pages[0]["text"] = window
    context["pages"] = pages
    scope["text_window"] = window_meta
    scope["bounded_local_verify"] = {
        "model": QWEN_VERIFY_MODEL,
        "revision": QWEN_VERIFY_REVISION,
        "runtime_contract": QWEN_VERIFY_RUNTIME_CONTRACT,
        "max_context_chars": max_context_chars,
        "eligibility": "RADIUS_0_EXACT_ONE_AUTHORITATIVE_PAGE",
    }
    context["verification_scope"] = scope
    payload = copy.deepcopy(job.payload)
    payload["_research_context"] = context
    return replace(job, payload=payload), "PASS"


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
    """Qwen-primary semantic delegate for the calibrated bounded VERIFY class."""

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
            "policy": "QWEN35_BOUNDED_VERIFY_PRIMARY_CODEX_FALLTHROUGH",
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

        bounded, reason = prepare_bounded_verify_job(
            job,
            max_context_chars=self.max_context_chars,
        )
        if bounded is None:
            return self._fallback(
                job,
                reason=reason,
                local_attempted=False,
            )

        try:
            response = self.local_inference.qwen_verify_request(
                messages=qwen_verify_messages(bounded),
                job_id=job.id,
            )
        except LocalInferenceBackpressureError as exc:
            return self._fallback(
                job,
                reason="QWEN_VERIFY_BACKPRESSURE",
                local_attempted=True,
                local_detail={"error": str(exc)},
            )
        except LocalInferenceError as exc:
            return self._fallback(
                job,
                reason="QWEN_VERIFY_RUNTIME_FAILURE",
                local_attempted=True,
                local_detail={"error": str(exc)},
            )

        parsed = response.get("result")
        guard = guard_qwen_semantic_result(
            bounded,
            parsed if isinstance(parsed, dict) else None,
        )
        if guard["status"] != "PASS_GUARDED":
            return self._fallback(
                job,
                reason="QWEN_VERIFY_GUARD_FAILURE",
                local_attempted=True,
                local_detail={
                    "guard_reasons": list(guard.get("reasons") or []),
                    "server_pid": response.get("server_pid"),
                    "request_seq": response.get("request_seq"),
                },
            )

        result = copy.deepcopy(dict(guard["canonical_result"]))
        usage = dict(response.get("usage") or {})
        result["_execution"] = {
            "provider": "local-qwen-vllm",
            "model": f"{QWEN_VERIFY_MODEL}@{QWEN_VERIFY_REVISION}",
            "response_id": response.get("request_id"),
            "executor_tier": str(self.tier),
            "reasoning_effort": "none",
            "usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "cached_input_tokens": 0,
                "output_tokens": int(usage.get("output_tokens") or 0),
                "reasoning_output_tokens": 0,
            },
            "routing": {
                "policy": "QWEN35_BOUNDED_VERIFY_PRIMARY_CODEX_FALLTHROUGH",
                "decision": "LOCAL_QWEN_ACCEPTED",
                "reason": "PASS_GUARDED",
                "local_attempted": True,
                "guard_reasons": [],
            },
            "local_inference": {
                "family": response.get("family"),
                "server_pid": response.get("server_pid"),
                "model_load_count": response.get("model_load_count"),
                "request_seq": response.get("request_seq"),
                "wall_time_ms": response.get("wall_time_ms"),
                "runtime_contract": response.get("runtime_contract"),
                "model_revision": response.get("model_revision"),
            },
        }
        return result
