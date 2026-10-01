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
                    {"id": "gap:a", "search_queries": ["query a1", "query a2"]},
                    {"id": "gap:b", "search_queries": ["query b1", "query b2"]},
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

    # Discovery + screening + acquisition fan-out.
    for _ in range(6):
        for worker in workers:
            worker.run_until_idle()
        planner.reconcile(run_id)

    # Simulate a standard worker dying after leasing one extraction job.
    crash_worker = "worker:crashed"
    queue.register_worker(crash_worker, [ExecutorTier.STANDARD])
    leased = queue.lease_next(
        worker_id=crash_worker,
        supported_tiers=[ExecutorTier.STANDARD],
        lease_seconds=0,
    )
    if leased is not None:
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
