import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_daemon_backpressure_health_and_overnight_report():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.daemon import BackpressureError, DaemonLimits, ResearchDaemon
    from research_engine.orchestration import ExecutorTier, OrchestratorStore, ResearchPlanner
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE human_review_requests, semantic_materializations,
                         research_job_attempts, research_jobs, research_workers,
                         research_runs CASCADE
                """
            )
        conn.commit()

    daemon = ResearchDaemon(
        queue=queue,
        planner=planner,
        workers=[],
        poll_seconds=0,
        limits=DaemonLimits(
            max_active_runs=1,
            max_pending_human=1,
            stale_worker_seconds=1,
        ),
    )

    assert daemon.submit_run(topic="topic one", run_id="run:one") == "run:one"
    with pytest.raises(BackpressureError, match="MAX_ACTIVE_RUNS"):
        daemon.submit_run(topic="topic two", run_id="run:two")

    queue.ensure_human_review("run:one", {"reason": "needs review"})
    with pytest.raises(BackpressureError, match="MAX_PENDING_HUMAN"):
        daemon.submit_run(topic="topic two", run_id="run:two")

    queue.resolve_human_review(
        "human-review:run:one",
        approved=True,
        decision={"reviewer": "test", "decision": "continue"},
    )
    assert daemon.submit_run(topic="topic two", run_id="run:two") == "run:two"

    queue.register_worker("worker:stale", [ExecutorTier.CHEAP])
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE research_workers SET last_heartbeat_at=now()-interval '1 hour' WHERE id='worker:stale'"
            )
        conn.commit()

    tick = daemon.tick()
    assert tick.stale_workers_marked == 1

    health = daemon.health_snapshot()
    assert health["runs"]["COMPLETED"] == 1
    assert health["runs"]["RUNNING"] == 1
    assert health["pending_human"] == 0
    stale = next(w for w in health["workers"] if w["id"] == "worker:stale")
    assert stale["state"] == "STALE"

    report = daemon.overnight_report(hours=24)
    assert {run["id"] for run in report["runs"]} == {"run:one", "run:two"}
    assert report["model_usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "by_tier_model": {},
    }
    assert report["human_reviews"][0]["run_id"] == "run:one"
    assert report["health"]["runs"]["RUNNING"] == 1
