import os
from pathlib import Path

import pytest


class ScriptedExecutor:
    def __init__(self, tier):
        self.tier = tier
        self.fail_novelty_once = True

    def execute(self, job):
        jt = str(job.job_type)
        if jt == "DISCOVERY":
            return {"candidates": [{"id": "paper:a"}, {"id": "paper:b"}]}
        if jt == "SCREEN":
            return {"retained": job.payload["candidates"]}
        if jt == "ACQUIRE":
            return {"artifact_id": f"artifact:{job.payload['item']['id']}"}
        if jt == "EXTRACT":
            return {"evidence": [f"evidence:{job.payload['item']['id']}"]}
        if jt == "VERIFY":
            return {"verified": True, "item_id": job.payload["item"]["id"]}
        if jt == "CHECKPOINT":
            return {"knowledge_checkpoint_id": "knowledge-checkpoint:dry-run-v1"}
        if jt == "SYNTHESIZE":
            return {"themes": ["open-set RF"], "claim_count": 2}
        if jt == "GAP_GENERATE":
            return {
                "gaps": [
                    {"id": "gap:a", "search_queries": ["query a1", "query a2", "query a3"]},
                    {"id": "gap:b", "search_queries": ["query b1", "query b2", "query b3"]},
                    {"id": "gap:c", "search_queries": ["query c1", "query c2", "query c3"]},
                ]
            }
        if jt == "NOVELTY_SEARCH":
            if self.fail_novelty_once:
                self.fail_novelty_once = False
                raise RuntimeError("transient scholarly provider failure")
            return {"counterevidence": [], "query": job.payload["query"]}
        if jt == "NOVELTY_ARBITRATE":
            return {
                "status": "NEEDS_HUMAN_REVIEW",
                "search_count": len(job.payload["search_results"]),
            }
        raise AssertionError(f"unexpected job type {jt}")


@pytest.mark.integration
def test_autonomous_cycle_fanout_retry_crash_recovery_and_human_gate():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.orchestration import (
        AgentWorker,
        ExecutorTier,
        JobType,
        OrchestratorStore,
        ResearchPlanner,
    )
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE human_review_requests, research_job_attempts, research_jobs,
                         research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "research-run:agentic-dry-v1"
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_checkpoints(id, fingerprint, scope, manifest, record_counts)
                VALUES (%s, %s, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    "knowledge-checkpoint:dry-run-v1",
                    "0" * 64,
                ),
            )
        conn.commit()

    queue.create_run(
        run_id=run_id,
        topic="open-set RF fingerprinting",
        policy={
            "executor_profiles": {
                "CHEAP": "cheap",
                "STANDARD": "standard",
                "STRONG": "strong",
            }
        },
    )
    planner = ResearchPlanner(queue)

    cheap_exec = ScriptedExecutor(ExecutorTier.CHEAP)
    standard_exec = ScriptedExecutor(ExecutorTier.STANDARD)
    strong_exec = ScriptedExecutor(ExecutorTier.STRONG)
    deterministic_exec = ScriptedExecutor(ExecutorTier.DETERMINISTIC)

    workers = [
        AgentWorker(
            worker_id="worker:cheap",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.CHEAP],
            executors={ExecutorTier.CHEAP: cheap_exec},
        ),
        AgentWorker(
            worker_id="worker:deterministic",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.DETERMINISTIC],
            executors={ExecutorTier.DETERMINISTIC: deterministic_exec},
        ),
        AgentWorker(
            worker_id="worker:standard",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STANDARD],
            executors={ExecutorTier.STANDARD: standard_exec},
        ),
        AgentWorker(
            worker_id="worker:strong",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STRONG],
            executors={ExecutorTier.STRONG: strong_exec},
        ),
    ]

    # Drive only discovery/screening/acquisition first, then crash a STANDARD
    # worker before the normal STANDARD worker can consume extraction.
    workers[0].run_until_idle()
    planner.reconcile(run_id)
    workers[1].run_until_idle()
    planner.reconcile(run_id)

    crash_worker = "worker:crashed"
    queue.register_worker(crash_worker, [ExecutorTier.STANDARD])
    leased = queue.lease_next(
        worker_id=crash_worker,
        supported_tiers=[ExecutorTier.STANDARD],
        lease_seconds=0,
    )
    assert leased is not None
    assert leased.job_type == JobType.EXTRACT
    recovered = queue.recover_expired_leases()
    assert leased.id in recovered

    # Continue autonomously. One novelty search intentionally fails once and retries.
    for _ in range(40):
        queue.recover_expired_leases()
        for worker in workers:
            worker.run_until_idle()
        planner.reconcile(run_id)
        if queue.run_state(run_id)["status"] == "WAITING_HUMAN":
            break

    state = queue.run_state(run_id)
    assert state["status"] == "WAITING_HUMAN"
    assert state["current_phase"] == "HUMAN_REVIEW"
    assert state["human_gate_state"] == "PENDING"
    assert state["knowledge_checkpoint_id"] == "knowledge-checkpoint:dry-run-v1"

    jobs = queue.jobs(run_id)
    by_type = {}
    for job in jobs:
        by_type.setdefault(job.job_type, []).append(job)

    assert len(by_type[JobType.ACQUIRE]) == 2
    assert len(by_type[JobType.EXTRACT]) == 2
    assert len(by_type[JobType.VERIFY]) == 2
    assert len(by_type[JobType.NOVELTY_SEARCH]) == 4
    assert len(by_type[JobType.NOVELTY_ARBITRATE]) == 2
    assert len(by_type[JobType.HUMAN_REVIEW]) == 1
    assert by_type[JobType.HUMAN_REVIEW][0].status == "WAITING_HUMAN"
    assert all(job.executor_tier != ExecutorTier.HUMAN for job in jobs if job.status == "RUNNING")

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM research_job_attempts WHERE status='EXPIRED'"
            )
            expired_attempts = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM research_job_attempts WHERE status='RETRY'"
            )
            retry_attempts = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM human_review_requests WHERE run_id=%s AND status='PENDING'",
                (run_id,),
            )
            pending_human = cur.fetchone()[0]

    assert expired_attempts >= 1
    assert retry_attempts >= 1
    assert pending_human == 1

    # Reconciliation is idempotent: rerunning it creates no duplicate downstream jobs.
    before = len(queue.jobs(run_id))
    planner.reconcile(run_id)
    after = len(queue.jobs(run_id))
    assert before == after


@pytest.mark.integration
def test_cancel_queued_except_salvages_legacy_novelty_fanout():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.orchestration import JobType, OrchestratorStore
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE human_review_requests, research_job_attempts, research_jobs,
                         research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "research-run:legacy-fanout"
    queue.create_run(run_id=run_id, topic="legacy fanout salvage")
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        idempotency_key="novelty-search:g1:0",
        payload={"gap": {"id": "g1"}, "query": "q0"},
    )
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        idempotency_key="novelty-search:g1:1",
        payload={"gap": {"id": "g1"}, "query": "q1"},
    )
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        idempotency_key="novelty-search:g1:2",
        payload={"gap": {"id": "g1"}, "query": "q2"},
    )

    cancelled = queue.cancel_queued_except(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        keep_idempotency_keys={
            "novelty-search:g1:0",
            "novelty-search:g1:1",
        },
    )
    assert cancelled == 1

    jobs = queue.jobs(run_id, JobType.NOVELTY_SEARCH)
    states = {job.idempotency_key: job.status for job in jobs}
    assert states["novelty-search:g1:0"] == "QUEUED"
    assert states["novelty-search:g1:1"] == "QUEUED"
    assert states["novelty-search:g1:2"] == "CANCELLED"


@pytest.mark.integration
def test_reconcile_reuses_legacy_novelty_job_payload_by_idempotency_key():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from psycopg.types.json import Jsonb

    from research_engine.orchestration import (
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
                TRUNCATE human_review_requests, research_job_attempts, research_jobs,
                         research_workers, research_runs CASCADE
                """
            )
        conn.commit()

    run_id = "research-run:legacy-idempotent-reconcile"
    queue.create_run(run_id=run_id, topic="legacy payload reconciliation")

    full_gap = {
        "id": "gap:legacy",
        "statement": "legacy gap",
        "search_queries": ["q0", "q1", "q2"],
    }
    gap_job_id = queue.enqueue(
        run_id=run_id,
        job_type=JobType.GAP_GENERATE,
        idempotency_key="stage:gap-generate",
        payload={"synthesis": {"legacy": True}},
        status="COMPLETED",
    )
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE research_jobs SET result=%s WHERE id=%s",
                (Jsonb({"gaps": [full_gap]}), gap_job_id),
            )
        conn.commit()

    legacy_payload = {"gap": full_gap, "query": "q0"}
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        idempotency_key="novelty-search:gap:legacy:0",
        payload=legacy_payload,
        depends_on=[gap_job_id],
        priority=30,
    )
    queue.enqueue(
        run_id=run_id,
        job_type=JobType.NOVELTY_SEARCH,
        idempotency_key="novelty-search:gap:legacy:2",
        payload={"gap": full_gap, "query": "q2"},
        depends_on=[gap_job_id],
        priority=30,
    )

    # Must not raise even though the bounded representation would truncate
    # search_queries and therefore differ from the immutable legacy payload.
    planner.reconcile(run_id)

    jobs = {
        job.idempotency_key: job
        for job in queue.jobs(run_id, JobType.NOVELTY_SEARCH)
    }
    assert jobs["novelty-search:gap:legacy:0"].status == "QUEUED"
    assert jobs["novelty-search:gap:legacy:0"].payload == legacy_payload
    assert jobs["novelty-search:gap:legacy:1"].status == "QUEUED"
    assert jobs["novelty-search:gap:legacy:2"].status == "CANCELLED"


@pytest.mark.integration
def test_codex_plan_limit_defers_without_consuming_failure_budget():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.orchestration import (
        AgentWorker,
        ExecutorTier,
        JobExecutionError,
        JobType,
        OrchestratorStore,
        ResearchPlanner,
    )
    from research_engine.postgres import PostgresStore

    class PlanLimited:
        def execute(self, job):
            raise JobExecutionError(
                "Usage limit reached. Try again in 1h.",
                code="CODEX_PLAN_LIMIT",
                retryable=True,
                detail={"retry_after_seconds": 3600},
            )

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
            cur.execute("DELETE FROM research_provider_backoffs")
        conn.commit()

    run_id = "run:plan-limit-backoff"
    queue.create_run(run_id=run_id, topic="plan limit durability")
    worker = AgentWorker(
        worker_id="worker:plan-limit",
        queue=queue,
        planner=planner,
        supported_tiers=[ExecutorTier.CHEAP],
        executors={ExecutorTier.CHEAP: PlanLimited()},
        backoff_provider="codex-chatgpt-plan",
    )

    leased = worker.run_once()
    assert leased is not None
    assert leased.job_type == JobType.DISCOVERY

    job = queue.jobs(run_id, JobType.DISCOVERY)[0]
    assert job.status == "QUEUED"
    assert job.attempt_count == 1

    backoff = queue.provider_backoff("codex-chatgpt-plan")
    assert backoff is not None
    assert backoff["reason"] == "CODEX_PLAN_LIMIT"

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status
                FROM research_job_attempts
                WHERE job_id=%s AND attempt_no=1
                """,
                (job.id,),
            )
            assert cur.fetchone()[0] == "DEFERRED_PLAN_LIMIT"

            # Make the deferred job available again and clear provider backoff
            # to prove three genuine failures are still required for terminal
            # failure even though attempt_no will advance past max_attempts.
            cur.execute(
                "UPDATE research_jobs SET available_at=now() WHERE id=%s",
                (job.id,),
            )
            cur.execute("DELETE FROM research_provider_backoffs")
        conn.commit()

    queue.register_worker("worker:genuine-failures", [ExecutorTier.CHEAP])
    for expected_status in ("QUEUED", "QUEUED", "FAILED"):
        current = queue.lease_next(
            worker_id="worker:genuine-failures",
            supported_tiers=[ExecutorTier.CHEAP],
        )
        assert current is not None
        actual = queue.fail(
            current,
            "worker:genuine-failures",
            {"type": "GENUINE_FAILURE"},
            retryable=True,
        )
        assert actual == expected_status

    final = queue.jobs(run_id, JobType.DISCOVERY)[0]
    assert final.status == "FAILED"
    assert final.attempt_count == 4

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)::int
                FROM research_job_attempts
                WHERE job_id=%s AND status='DEFERRED_PLAN_LIMIT'
                """,
                (final.id,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                """
                SELECT count(*)::int
                FROM research_job_attempts
                WHERE job_id=%s AND status IN ('FAILED','RETRY','EXPIRED')
                """,
                (final.id,),
            )
            assert cur.fetchone()[0] == 3
