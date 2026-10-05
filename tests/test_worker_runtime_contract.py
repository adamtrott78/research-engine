from pathlib import Path

from research_engine.orchestration import (
    ExecutorTier,
    JobExecutionError,
)


def test_job_execution_error_carries_retry_semantics():
    err = JobExecutionError(
        "refused",
        code="MODEL_REFUSAL",
        retryable=False,
        detail={"model": "x"},
    )
    assert err.code == "MODEL_REFUSAL"
    assert err.retryable is False
    assert err.detail["model"] == "x"


def test_standard_semantic_worker_can_wrap_codex_with_bounded_qwen_verify():
    root = Path(__file__).parents[1]
    worker = (
        root / "src" / "research_engine" / "worker_runtime.py"
    ).read_text(encoding="utf-8")
    daemon = (
        root / "src" / "research_engine" / "daemon.py"
    ).read_text(encoding="utf-8")

    assert "BoundedVerifySemanticDelegate" in worker
    assert "qwen_verify_enabled_from_env" in worker
    assert "tier == ExecutorTier.STANDARD" in worker
    assert "local_inference=local_inference" in worker
    assert 'metadata_backend = "codex+qwen-bounded-verify"' in worker

    assert "local_inference_service_from_env()" in daemon
    assert "semantic_worker_from_env(" in daemon
    assert "local_inference=local_inference" in daemon
