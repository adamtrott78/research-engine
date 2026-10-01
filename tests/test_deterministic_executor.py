from dataclasses import dataclass
from pathlib import Path

import pytest

from research_engine.artifact_store import ContentAddressedArtifactStore
from research_engine.deterministic_executor import (
    DeterministicResearchExecutor,
    arxiv_id_from_uri,
)
from research_engine.orchestration import (
    ExecutorTier,
    JobExecutionError,
    JobType,
    ResearchJob,
)


def job(job_type, payload):
    return ResearchJob(
        id="job:test",
        run_id="run:test",
        job_type=job_type,
        executor_tier=ExecutorTier.DETERMINISTIC,
        status="RUNNING",
        priority=0,
        idempotency_key="test",
        payload=payload,
        result=None,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker:test",
        lease_expires_at=None,
    )


def test_arxiv_id_from_uri_accepts_abs_and_pdf_urls():
    assert arxiv_id_from_uri("https://arxiv.org/abs/2605.15593v1") == "2605.15593v1"
    assert arxiv_id_from_uri("https://arxiv.org/pdf/2605.15593v1") == "2605.15593v1"
    assert arxiv_id_from_uri("https://arxiv.org/pdf/2605.15593v1.pdf") == "2605.15593v1"
    assert arxiv_id_from_uri("https://example.com/paper.pdf") is None


def test_unsupported_provider_is_non_retryable(tmp_path):
    class Store:
        pass

    executor = DeterministicResearchExecutor(
        store=Store(),
        artifact_root=tmp_path,
        arxiv_provider=object(),
    )
    with pytest.raises(JobExecutionError) as exc:
        executor.execute(
            job(
                JobType.ACQUIRE,
                {
                    "item": {
                        "id": "work:x",
                        "title": "X",
                        "source_uri": "https://example.com/x.pdf",
                    }
                },
            )
        )
    assert exc.value.code == "UNSUPPORTED_SOURCE_PROVIDER"
    assert exc.value.retryable is False
