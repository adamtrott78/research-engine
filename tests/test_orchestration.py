from research_engine.orchestration import (
    DEFAULT_EXECUTOR_PROFILES,
    ExecutorTier,
    JOB_TIERS,
    JobType,
    _job_id,
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
