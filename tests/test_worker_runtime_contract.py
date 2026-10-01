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
