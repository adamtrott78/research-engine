from pathlib import Path

import pytest

from research_engine.orchestration import (
    ExecutorTier,
    JobExecutionError,
    JobType,
    ResearchJob,
)
from research_engine.semantic_pipeline import ResearchContextHydrator


def _job() -> ResearchJob:
    return ResearchJob(
        id="job:verify-window",
        run_id="run:verify-window",
        job_type=JobType.VERIFY,
        executor_tier=ExecutorTier.STANDARD,
        status="RUNNING",
        priority=0,
        idempotency_key="verify-window",
        payload={
            "item": {"id": "paper:1"},
            "extraction": {
                "_materialization": {
                    "artifact_id": "artifact:1",
                    "evidence_ids": ["evidence:1", "evidence:2"],
                }
            },
        },
        result=None,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker:1",
        lease_expires_at=None,
    )


def test_verification_pages_are_only_cited_pages(tmp_path: Path):
    hydrator = ResearchContextHydrator(
        store=None,
        artifact_root=tmp_path,
        verify_page_radius=0,
    )
    pages = [
        {"page_number": 1, "text": "one"},
        {"page_number": 2, "text": "two"},
        {"page_number": 3, "text": "three"},
        {"page_number": 4, "text": "four"},
    ]
    evidence = [
        {"id": "e1", "page_number": 2},
        {"id": "e2", "page_number": 4},
    ]

    selected, cited = hydrator._verification_pages(pages, evidence)

    assert cited == [2, 4]
    assert [page["page_number"] for page in selected] == [2, 4]


def test_verification_page_radius_is_configurable(tmp_path: Path):
    hydrator = ResearchContextHydrator(
        store=None,
        artifact_root=tmp_path,
        verify_page_radius=1,
    )
    pages = [
        {"page_number": 1, "text": "one"},
        {"page_number": 2, "text": "two"},
        {"page_number": 3, "text": "three"},
        {"page_number": 4, "text": "four"},
    ]

    selected, cited = hydrator._verification_pages(
        pages,
        [{"id": "e1", "page_number": 2}],
    )

    assert cited == [2]
    assert [page["page_number"] for page in selected] == [1, 2, 3]


def test_verification_fails_closed_for_missing_locator(tmp_path: Path):
    hydrator = ResearchContextHydrator(
        store=None,
        artifact_root=tmp_path,
    )

    with pytest.raises(JobExecutionError) as exc:
        hydrator._verification_pages(
            [{"page_number": 1, "text": "one"}],
            [{"id": "e1", "page_number": None}],
        )

    assert exc.value.code == "MISSING_VERIFICATION_LOCATOR"


def test_verification_fails_closed_for_unavailable_page(tmp_path: Path):
    hydrator = ResearchContextHydrator(
        store=None,
        artifact_root=tmp_path,
    )

    with pytest.raises(JobExecutionError) as exc:
        hydrator._verification_pages(
            [{"page_number": 1, "text": "one"}],
            [{"id": "e1", "page_number": 3}],
        )

    assert exc.value.code == "MISSING_VERIFICATION_SOURCE_PAGE"


def test_verify_hydration_excludes_uncited_pages(tmp_path: Path):
    hydrator = ResearchContextHydrator(
        store=None,
        artifact_root=tmp_path,
        verify_page_radius=0,
    )
    pages = [
        {"page_number": 1, "text": "irrelevant"},
        {"page_number": 2, "text": "quoted evidence"},
        {"page_number": 3, "text": "irrelevant"},
    ]
    evidence = [
        {
            "id": "evidence:1",
            "page_number": 2,
            "raw_source_form": "quoted evidence",
            "normalized_form": "claim",
        },
        {
            "id": "evidence:2",
            "page_number": 2,
            "raw_source_form": "quoted evidence",
            "normalized_form": "second claim",
        },
    ]

    hydrator._artifact_context = lambda artifact_id: {
        "source_artifact": {"id": artifact_id},
        "pages": pages,
    }
    hydrator._evidence = lambda ids: evidence

    hydrated = hydrator.hydrate(_job())
    context = hydrated.payload["_research_context"]

    assert [page["page_number"] for page in context["pages"]] == [2]
    assert context["authoritative_evidence"] == evidence
    assert context["verification_scope"] == {
        "cited_page_numbers": [2],
        "page_radius": 0,
        "supplied_page_numbers": [2],
    }
