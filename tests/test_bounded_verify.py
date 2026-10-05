from __future__ import annotations

import json
from pathlib import Path

from research_engine.bounded_verify import (
    BoundedVerifySemanticDelegate,
    prepare_bounded_verify_job,
    prepare_bounded_verify_pages,
)
from research_engine.local_inference import LocalInferenceError
from research_engine.orchestration import ExecutorTier, JobType, ResearchJob


def _job(*, pages=None, evidence=None, supplied=None) -> ResearchJob:
    pages = pages or [
        {
            "page_number": 2,
            "text": "Methods text. The detector achieved 94.2 percent accuracy. "
            "The threshold was fixed before evaluation.",
        }
    ]
    evidence = evidence or [
        {
            "id": "evidence:1",
            "page_number": 2,
            "raw_source_form": "The detector achieved 94.2 percent accuracy.",
            "normalized_form": "detector accuracy 94.2 percent",
        },
        {
            "id": "evidence:2",
            "page_number": 2,
            "raw_source_form": "The threshold was fixed before evaluation.",
            "normalized_form": "threshold fixed before evaluation",
        },
    ]
    supplied = supplied or [2]
    return ResearchJob(
        id="job:verify",
        run_id="run:test",
        job_type=JobType.VERIFY,
        executor_tier=ExecutorTier.STANDARD,
        status="RUNNING",
        priority=0,
        idempotency_key="verify:test",
        payload={
            "item": {"id": "item:1"},
            "_research_context": {
                "pages": pages,
                "authoritative_evidence": evidence,
                "verification_scope": {
                    "page_radius": 0,
                    "cited_page_numbers": sorted(
                        {int(x["page_number"]) for x in evidence}
                    ),
                    "supplied_page_numbers": supplied,
                },
            },
        },
        result=None,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker:test",
        lease_expires_at=None,
    )


class _Fallback:
    def __init__(self):
        self.calls = 0

    def execute(self, _job):
        self.calls += 1
        return {
            "verified": False,
            "item_id": "item:1",
            "accepted_evidence_ids": [],
            "issues": ["codex fallback"],
            "_execution": {
                "provider": "codex",
                "model": "test",
                "reasoning_effort": "medium",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "reasoning_output_tokens": 0,
                },
            },
        }


class _Local:
    def __init__(self, result=None, error=None, results_by_page=None):
        self.qwen_verify = object()
        self.calls = 0
        self.result = result or {
            "verified": True,
            "item_id": "item:1",
            "accepted_evidence_ids": ["evidence:1", "evidence:2"],
            "issues": [],
        }
        self.results_by_page = dict(results_by_page or {})
        self.error = error

    def qwen_verify_request(self, *, messages, job_id, timeout_seconds=None):
        self.calls += 1
        assert job_id.startswith("job:verify:page:")
        assert messages
        if self.error is not None:
            raise self.error
        page_number = int(job_id.rsplit(":", 1)[1])
        result = self.results_by_page.get(page_number, self.result)
        return {
            "result": result,
            "usage": {"input_tokens": 321, "output_tokens": 17},
            "family": "qwen_verify",
            "server_pid": 123,
            "model_load_count": 1,
            "request_seq": self.calls,
            "wall_time_ms": 250,
            "runtime_contract": "qwen35-vllm-cu129-image-native-sampler-v3",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        }


def test_prepare_bounded_verify_accepts_exact_one_page_radius_zero():
    bounded, reason = prepare_bounded_verify_job(_job())
    assert reason == "PASS"
    assert bounded is not None
    context = bounded.payload["_research_context"]
    assert len(context["pages"]) == 1
    assert context["verification_scope"]["page_radius"] == 0
    assert context["verification_scope"]["bounded_local_verify"]["eligibility"] == (
        "RADIUS_0_PAGE_SCOPED_AUTHORITATIVE_EVIDENCE"
    )


def test_prepare_bounded_verify_splits_multi_page_context_into_calibrated_units():
    job = _job(
        pages=[
            {"page_number": 2, "text": "quote two"},
            {"page_number": 3, "text": "quote three"},
        ],
        evidence=[
            {
                "id": "evidence:1",
                "page_number": 2,
                "raw_source_form": "quote two",
                "normalized_form": "two",
            },
            {
                "id": "evidence:2",
                "page_number": 3,
                "raw_source_form": "quote three",
                "normalized_form": "three",
            },
        ],
        supplied=[2, 3],
    )
    units, reason = prepare_bounded_verify_pages(job)
    assert reason == "PASS"
    assert units is not None
    assert [page for page, _bounded, _ids in units] == [2, 3]
    assert [ids for _page, _bounded, ids in units] == [
        ["evidence:1"],
        ["evidence:2"],
    ]
    for page, bounded, ids in units:
        context = bounded.payload["_research_context"]
        assert [row["page_number"] for row in context["pages"]] == [page]
        assert [row["id"] for row in context["authoritative_evidence"]] == ids
        assert context["verification_scope"]["supplied_page_numbers"] == [page]

    bounded, one_reason = prepare_bounded_verify_job(job)
    assert bounded is None
    assert one_reason.startswith("VERIFY_EXACT_ONE_PAGE_REQUIRED")


def test_bounded_verify_uses_qwen_without_codex_on_guarded_success():
    fallback = _Fallback()
    local = _Local()
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )

    result = delegate.execute(_job())

    assert fallback.calls == 0
    assert local.calls == 1
    assert result["verified"] is True
    assert result["accepted_evidence_ids"] == ["evidence:1", "evidence:2"]
    assert result["_execution"]["provider"] == "local-qwen-vllm"
    assert result["_execution"]["routing"]["decision"] == "LOCAL_QWEN_ACCEPTED"
    assert result["_execution"]["usage"]["input_tokens"] == 321


def test_bounded_verify_falls_through_on_guard_failure():
    fallback = _Fallback()
    local = _Local(
        result={
            "verified": True,
            "item_id": "item:1",
            "accepted_evidence_ids": ["invented:evidence"],
            "issues": [],
        }
    )
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )

    result = delegate.execute(_job())

    assert local.calls == 1
    assert fallback.calls == 1
    routing = result["_execution"]["routing"]
    assert routing["decision"] == "FALLTHROUGH"
    assert routing["reason"] == "QWEN_VERIFY_PAGE_2_GUARD_FAILURE"
    assert "UNKNOWN_ACCEPTED_EVIDENCE_IDS:invented:evidence" in (
        routing["local_detail"]["guard_reasons"]
    )


def test_bounded_verify_falls_through_on_local_runtime_failure():
    fallback = _Fallback()
    local = _Local(error=LocalInferenceError("boom"))
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )

    result = delegate.execute(_job())

    assert local.calls == 1
    assert fallback.calls == 1
    assert result["_execution"]["routing"]["reason"] == (
        "QWEN_VERIFY_PAGE_2_RUNTIME_FAILURE"
    )


def test_bounded_verify_aggregates_multi_page_full_acceptance_without_codex():
    fallback = _Fallback()
    local = _Local(
        results_by_page={
            2: {
                "verified": True,
                "item_id": "item:1",
                "accepted_evidence_ids": ["evidence:1"],
                "issues": [],
            },
            3: {
                "verified": True,
                "item_id": "item:1",
                "accepted_evidence_ids": ["evidence:2"],
                "issues": [],
            },
        }
    )
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )
    job = _job(
        pages=[
            {"page_number": 2, "text": "quote two"},
            {"page_number": 3, "text": "quote three"},
        ],
        evidence=[
            {
                "id": "evidence:1",
                "page_number": 2,
                "raw_source_form": "quote two",
                "normalized_form": "two",
            },
            {
                "id": "evidence:2",
                "page_number": 3,
                "raw_source_form": "quote three",
                "normalized_form": "three",
            },
        ],
        supplied=[2, 3],
    )

    result = delegate.execute(job)

    assert fallback.calls == 0
    assert local.calls == 2
    assert result["verified"] is True
    assert result["accepted_evidence_ids"] == ["evidence:1", "evidence:2"]
    assert result["issues"] == []
    routing = result["_execution"]["routing"]
    assert routing["decision"] == "LOCAL_QWEN_ACCEPTED"
    assert routing["page_count"] == 2
    assert routing["qwen_calls"] == 2
    assert result["_execution"]["usage"]["input_tokens"] == 642


def test_bounded_verify_falls_through_whole_job_on_partial_page_acceptance():
    fallback = _Fallback()
    local = _Local(
        results_by_page={
            2: {
                "verified": True,
                "item_id": "item:1",
                "accepted_evidence_ids": ["evidence:1"],
                "issues": [],
            },
            3: {
                "verified": False,
                "item_id": "item:1",
                "accepted_evidence_ids": [],
                "issues": ["not supported"],
            },
        }
    )
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )
    job = _job(
        pages=[
            {"page_number": 2, "text": "quote two"},
            {"page_number": 3, "text": "quote three"},
        ],
        evidence=[
            {
                "id": "evidence:1",
                "page_number": 2,
                "raw_source_form": "quote two",
                "normalized_form": "two",
            },
            {
                "id": "evidence:2",
                "page_number": 3,
                "raw_source_form": "quote three",
                "normalized_form": "three",
            },
        ],
        supplied=[2, 3],
    )

    result = delegate.execute(job)

    assert local.calls == 2
    assert fallback.calls == 1
    assert result["_execution"]["routing"]["decision"] == "FALLTHROUGH"
    assert result["_execution"]["routing"]["reason"] == (
        "QWEN_VERIFY_PAGE_3_NOT_VERIFIED"
    )


def test_bounded_verify_falls_through_without_local_call_on_scope_mismatch():
    fallback = _Fallback()
    local = _Local()
    delegate = BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=local,
        enabled=True,
    )
    job = _job(supplied=[99])

    result = delegate.execute(job)

    assert local.calls == 0
    assert fallback.calls == 1
    assert result["_execution"]["routing"]["local_attempted"] is False
    assert result["_execution"]["routing"]["reason"].startswith(
        "VERIFY_PAGE_SCOPE_MISMATCH"
    )


def test_production_gate_is_non_materializing_and_forbids_codex():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-qwen35-bounded-verify-production-gate.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "class NoCodexFallback" in text
    assert "forbidden-fallback-sentinel" in text
    assert "LOCAL_PAGE_SCOPED_VERIFY_FELL_THROUGH_WITHOUT_CODEX_CALL" in text
    assert "codex_calls_performed" in text
    assert 'SCIENTIFIC_STATE_MATERIALIZED=false' in text
    assert 'PRODUCTION_ENV_MUTATED=false' in text
    assert "SemanticResultMaterializer" not in text
    assert "CodexExecExecutor" not in text
    assert "pip install" not in text
    assert "prepare_bounded_verify_pages" in text
    assert "page_scoped_units" in text
    assert "QWEN_BOUNDED_VERIFY_PRODUCTION_GATE=PASS" in text
    assert "workspace-control-tmp" in text


def test_production_config_and_acceptance_require_bounded_qwen_verify():
    root = Path(__file__).parents[1]
    config = (
        root / "deploy" / "configure-production-local-inference.sh"
    ).read_text(encoding="utf-8")
    acceptance = (
        root / "deploy" / "run-lambda-production-acceptance.sh"
    ).read_text(encoding="utf-8")

    for line in (
        "RESEARCH_ENGINE_QWEN_VERIFY_ENABLED=1",
        "RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX=1",
        "RESEARCH_ENGINE_QWEN_VERIFY_MAX_CONTEXT_CHARS=20000",
        "RESEARCH_ENGINE_QWEN_VERIFY_TIMEOUT_SECONDS=900",
    ):
        assert line in config
    assert "QWEN_VERIFY_OWNER_SPAWNED_DURING_BOOTSTRAP" in acceptance
    assert "qwen_verify_owner_zero_spawn" in acceptance
    assert "BOUNDED_LOCAL_QWEN_PRIMARY_WITH_REMOTE_CODEX_FALLTHROUGH" in acceptance


def test_localization_matrix_promotes_only_bounded_verify():
    root = Path(__file__).parents[1]
    matrix = json.loads(
        (
            root
            / "benchmarks"
            / "document_intelligence"
            / "localization-matrix.v2.json"
        ).read_text(encoding="utf-8")
    )
    rows = {row["job_type"]: row for row in matrix["job_type_decisions"]}
    verify = rows["VERIFY"]
    extract = rows["EXTRACT"]

    assert verify["production_placement"] == (
        "BOUNDED_LOCAL_QWEN_PRIMARY_WITH_REMOTE_CODEX_FALLTHROUGH"
    )
    assert verify["localization_decision"] == "PROMOTE_BOUNDED_LOCAL_VERIFY"
    assert "Qwen3.5-4B" in " ".join(verify["local_support"])
    assert extract["localization_decision"] != "PROMOTE_BOUNDED_LOCAL_VERIFY"
