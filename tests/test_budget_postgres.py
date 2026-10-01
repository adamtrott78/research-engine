import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.mark.integration
def test_usage_ledger_pauses_and_explicitly_resumes_budgeted_run():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.daemon import DaemonLimits, ResearchDaemon
    from research_engine.orchestration import ExecutorTier, JobType, OrchestratorStore, ResearchPlanner
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE research_usage_events, human_review_requests,
                         semantic_materializations, research_job_attempts,
                         research_jobs, research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "run:budget"
    queue.create_run(
        run_id=run_id,
        topic="budgeted RF research",
        policy={
            "budgets": {
                "max_input_tokens": 10,
                "max_output_tokens": 5,
                "max_semantic_jobs": 1,
                "max_jobs_by_tier": {"CHEAP": 1},
            }
        },
    )
    queue.register_worker("worker:cheap-budget", [ExecutorTier.CHEAP])
    job = queue.lease_next(
        worker_id="worker:cheap-budget",
        supported_tiers=[ExecutorTier.CHEAP],
    )
    assert job is not None
    assert job.job_type == JobType.DISCOVERY
    queue.complete(
        job,
        "worker:cheap-budget",
        {
            "candidates": [],
            "_execution": {
                "provider": "openai",
                "model": "cheap-test",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        },
    )
    planner.reconcile(run_id)

    daemon = ResearchDaemon(
        queue=queue,
        planner=planner,
        workers=[],
        poll_seconds=0,
        limits=DaemonLimits(),
    )
    tick = daemon.tick()
    assert tick.budget_paused_runs == 1

    state = queue.run_state(run_id)
    assert state["status"] == "PAUSED_BUDGET"
    assert state["human_gate_state"] == "BUDGET_REVIEW"

    budget = daemon.budget_status(run_id)
    assert budget["exhausted"] is True
    assert "MAX_INPUT_TOKENS" in budget["reasons"]
    assert "MAX_SEMANTIC_JOBS" in budget["reasons"]
    assert "MAX_CHEAP_JOBS" in budget["reasons"]
    assert budget["usage"]["input_tokens"] == 10
    assert budget["usage"]["semantic_jobs"] == 1

    # Paused runs cannot lease queued downstream work.
    assert queue.lease_next(
        worker_id="worker:cheap-budget",
        supported_tiers=[ExecutorTier.CHEAP],
    ) is None

    resumed = daemon.resume_budget(
        run_id,
        budgets={
            "max_input_tokens": 100,
            "max_semantic_jobs": 10,
            "max_jobs_by_tier": {"CHEAP": 10},
        },
    )
    assert resumed["status"] == "RUNNING"
    assert resumed["exhausted"] is False

    screen = queue.lease_next(
        worker_id="worker:cheap-budget",
        supported_tiers=[ExecutorTier.CHEAP],
    )
    assert screen is not None
    assert screen.job_type == JobType.SCREEN

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT provider,model,input_tokens,output_tokens
                FROM research_usage_events
                WHERE run_id=%s
                """,
                (run_id,),
            )
            assert cur.fetchone() == ("openai", "cheap-test", 10, 2)


@pytest.mark.integration
def test_saturated_unused_tier_does_not_pause_other_runnable_tier():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.daemon import DaemonLimits, ResearchDaemon
    from research_engine.orchestration import (
        ExecutorTier,
        JobType,
        OrchestratorStore,
        ResearchPlanner,
    )
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE research_usage_events, human_review_requests,
                         semantic_materializations, research_job_attempts,
                         research_jobs, research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "run:tier-budget-routing"
    queue.create_run(
        run_id=run_id,
        topic="tier budget routing",
        policy={
            "budgets": {
                "max_semantic_jobs": 10,
                "max_jobs_by_tier": {
                    "CHEAP": 1,
                    "STANDARD": 5,
                    "STRONG": 5,
                },
            }
        },
    )
    queue.register_worker("worker:cheap-routing", [ExecutorTier.CHEAP])
    cheap = queue.lease_next(
        worker_id="worker:cheap-routing",
        supported_tiers=[ExecutorTier.CHEAP],
    )
    assert cheap is not None
    queue.complete(
        cheap,
        "worker:cheap-routing",
        {
            "candidates": [],
            "_execution": {
                "provider": "codex-chatgpt-plan",
                "model": "cheap-test",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )

    queue.enqueue(
        run_id=run_id,
        job_type=JobType.SYNTHESIZE,
        idempotency_key="test:strong-after-cheap-cap",
        payload={},
        priority=200,
    )

    daemon = ResearchDaemon(
        queue=queue,
        planner=planner,
        workers=[],
        poll_seconds=0,
        limits=DaemonLimits(),
    )
    budget = daemon.budget_status(run_id)
    assert "MAX_CHEAP_JOBS" in budget["saturated_reasons"]
    assert budget["exhausted"] is False
    assert budget["reasons"] == []

    tick = daemon.tick()
    assert tick.budget_paused_runs == 0
    assert queue.run_state(run_id)["status"] == "RUNNING"

    queue.register_worker("worker:strong-routing", [ExecutorTier.STRONG])
    strong = queue.lease_next(
        worker_id="worker:strong-routing",
        supported_tiers=[ExecutorTier.STRONG],
    )
    assert strong is not None
    assert strong.job_type == JobType.SYNTHESIZE

    # The saturated CHEAP tier remains admission-blocked.
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.SCREEN,
        idempotency_key="test:cheap-blocked",
        payload={},
        priority=300,
    )
    assert queue.lease_next(
        worker_id="worker:cheap-routing",
        supported_tiers=[ExecutorTier.CHEAP],
    ) is None


@pytest.mark.integration
def test_concurrent_workers_cannot_overlease_semantic_budget():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.orchestration import (
        ExecutorTier,
        JobType,
        OrchestratorStore,
    )
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE research_usage_events, human_review_requests,
                         semantic_materializations, research_job_attempts,
                         research_jobs, research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "run:concurrent-budget-admission"
    queue.create_run(
        run_id=run_id,
        topic="concurrent semantic budget",
        policy={
            "budgets": {
                "max_semantic_jobs": 1,
                "max_jobs_by_tier": {"CHEAP": 1},
            }
        },
    )
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.SCREEN,
        idempotency_key="test:second-cheap-job",
        payload={},
        priority=99,
    )

    queue.register_worker("worker:race-a", [ExecutorTier.CHEAP])
    queue.register_worker("worker:race-b", [ExecutorTier.CHEAP])

    def lease(worker_id):
        return queue.lease_next(
            worker_id=worker_id,
            supported_tiers=[ExecutorTier.CHEAP],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lease, ["worker:race-a", "worker:race-b"]))

    leased = [job for job in jobs if job is not None]
    assert len(leased) == 1

    all_jobs = queue.jobs(run_id)
    assert sum(job.status == "RUNNING" for job in all_jobs) == 1
    assert sum(job.status == "QUEUED" for job in all_jobs) == 1
