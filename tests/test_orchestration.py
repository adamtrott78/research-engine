from research_engine.orchestration import (
    DEFAULT_EXECUTOR_PROFILES,
    ExecutorTier,
    JOB_TIERS,
    JobType,
    ResearchJob,
    _job_id,
    acquisition_candidate_supported,
    bounded_discovery_candidates,
    canonical_acquisition_candidate,
    bounded_gap_candidates,
    canonical_screened_candidates,
)


def test_job_tier_routing_separates_gruntwork_reasoning_and_human():
    assert JOB_TIERS[JobType.ACQUIRE] == ExecutorTier.DETERMINISTIC
    assert JOB_TIERS[JobType.DISCOVERY] == ExecutorTier.CHEAP
    assert JOB_TIERS[JobType.EXTRACT] == ExecutorTier.STANDARD
    assert JOB_TIERS[JobType.SYNTHESIZE] == ExecutorTier.STRONG
    assert JOB_TIERS[JobType.HUMAN_REVIEW] == ExecutorTier.HUMAN
    assert DEFAULT_EXECUTOR_PROFILES[ExecutorTier.CHEAP].model_alias == "cheap"
    assert DEFAULT_EXECUTOR_PROFILES[ExecutorTier.STRONG].reasoning_class == "high"


def test_job_ids_are_deterministic_and_idempotency_scoped_to_run():
    a = _job_id("run:a", "acquire:paper-1")
    b = _job_id("run:a", "acquire:paper-1")
    c = _job_id("run:b", "acquire:paper-1")
    assert a == b
    assert a != c


def test_bounded_gap_candidates_caps_gaps_and_queries_without_mutating_source():
    result = {
        "gaps": [
            {"id": "g1", "search_queries": ["a", "b", "c"]},
            {"id": "g2", "search_queries": ["d", "e", "f"]},
            {"id": "g3", "search_queries": ["g", "h", "i"]},
        ]
    }
    bounded = bounded_gap_candidates(result, {})
    assert [g["id"] for g in bounded] == ["g1", "g2"]
    assert bounded[0]["search_queries"] == ["a", "b"]
    assert bounded[1]["search_queries"] == ["d", "e"]
    assert result["gaps"][0]["search_queries"] == ["a", "b", "c"]



def _research_job(*, job_type, payload, result):
    return ResearchJob(
        id=f"job:{str(job_type).lower()}",
        run_id="run:test",
        job_type=job_type,
        executor_tier=JOB_TIERS[job_type],
        status="COMPLETED",
        priority=1,
        idempotency_key=f"test:{str(job_type).lower()}",
        payload=payload,
        result=result,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
    )


def test_discovery_cap_and_screening_preserve_original_candidate_metadata():
    discovered = [
        {
            "id": "paper:1",
            "title": "Original One",
            "source_uri": "https://arxiv.org/abs/2601.00001",
            "rationale": "one",
        },
        {
            "id": "paper:2",
            "title": "Original Two",
            "source_uri": "https://arxiv.org/abs/2601.00002",
            "rationale": "two",
        },
        {
            "id": "paper:3",
            "title": "Outside Cap",
            "source_uri": "https://arxiv.org/abs/2601.00003",
            "rationale": "three",
        },
    ]
    discovery = _research_job(
        job_type=JobType.DISCOVERY,
        payload={"candidate_limit": 2},
        result={"candidates": discovered},
    )
    screen = _research_job(
        job_type=JobType.SCREEN,
        payload={"candidates": discovered[:2]},
        result={
            "retained": [
                {
                    "id": "paper:2",
                    "title": "Mutated title",
                    "source_uri": "https://publisher.example/not-arxiv",
                    "rationale": "mutated",
                },
                {
                    "id": "paper:invented",
                    "title": "Invented",
                    "source_uri": "https://arxiv.org/abs/2601.99999",
                    "rationale": "invented",
                },
            ]
        },
    )

    assert [x["id"] for x in bounded_discovery_candidates(discovery)] == [
        "paper:1",
        "paper:2",
    ]
    canonical = canonical_screened_candidates(discovery, screen)
    assert canonical == [discovered[1]]
    assert acquisition_candidate_supported(canonical[0]) is True
    assert acquisition_candidate_supported(
        {
            "id": "publisher:1",
            "source_uri": "https://publisher.example/paper",
        }
    ) is False



def test_canonical_acquisition_candidate_derives_stable_arxiv_work_id():
    item = {
        "id": "model:anything",
        "title": "Versioned arXiv paper",
        "source_uri": "https://arxiv.org/abs/2108.01656v3",
        "rationale": "fixture",
    }
    canonical = canonical_acquisition_candidate(item)
    assert canonical is not None
    assert canonical["id"] == "arxiv:2108.01656"
    assert canonical["discovery_id"] == "model:anything"
    assert canonical["source_uri"] == item["source_uri"]
