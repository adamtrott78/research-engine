import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.mark.integration
def test_ops_snapshot_attributes_usage_and_forecasts_allowance():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.ops import OpsService
    from research_engine.orchestration import ExecutorTier, OrchestratorStore
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)
    ops = OpsService(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE research_allowance_observations,
                         research_usage_events,
                         human_review_requests,
                         semantic_materializations,
                         research_job_attempts,
                         research_jobs,
                         research_workers,
                         research_runs CASCADE
                """
            )
            cur.execute("DELETE FROM research_provider_backoffs")
        conn.commit()

    run_id = "run:ops"
    queue.create_run(
        run_id=run_id,
        topic="ops telemetry",
        discovery_payload={"candidate_limit": 2},
    )
    queue.register_worker(
        "daemon:test:123:cheap:0",
        [ExecutorTier.CHEAP],
        metadata={
            "logical_agent_id": "semantic:cheap:0",
            "host_name": "test-host",
            "process_id": 123,
            "code_sha": "abc123",
            "backend": "codex",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
        },
    )
    job = queue.lease_next(
        worker_id="daemon:test:123:cheap:0",
        supported_tiers=[ExecutorTier.CHEAP],
    )
    assert job is not None
    queue.complete(
        job,
        "daemon:test:123:cheap:0",
        {
            "candidates": [
                {
                    "id": "paper:1",
                    "title": "One",
                    "source_uri": "https://arxiv.org/abs/2601.00001",
                    "rationale": "one",
                },
                {
                    "id": "paper:2",
                    "title": "Two",
                    "source_uri": "https://arxiv.org/abs/2601.00002",
                    "rationale": "two",
                },
            ],
            "_execution": {
                "provider": "codex-chatgpt-plan",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 3,
                },
                "web_search": {
                    "mode": "codex-live",
                    "events": [{"query": "q", "action": {"type": "search"}}],
                },
            },
        },
    )

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_type,worker_id,logical_agent_id,attempt_no,
                       reasoning_effort,input_tokens,cached_input_tokens,
                       output_tokens,reasoning_output_tokens,
                       wall_time_ms,queue_wait_ms,web_search_count
                FROM research_usage_events
                WHERE job_id=%s
                """,
                (job.id,),
            )
            row = cur.fetchone()

    assert row[0] == "DISCOVERY"
    assert row[1] == "daemon:test:123:cheap:0"
    assert row[2] == "semantic:cheap:0"
    assert row[3] == 1
    assert row[4] == "low"
    assert row[5:9] == (100, 20, 10, 3)
    assert row[9] >= 0
    assert row[10] >= 0
    assert row[11] == 1

    start = datetime.now(timezone.utc) - timedelta(hours=1)
    ops.observe_allowance(
        window="5h",
        percent_used=16,
        approximate=True,
        note="test seed",
        observed_at=start,
    )
    ops.observe_allowance(
        window="5h",
        percent_used=20,
        approximate=False,
        observed_at=start + timedelta(hours=1),
    )

    snapshot = ops.snapshot(run_id=run_id, windows=(1, 5))
    assert snapshot["run"]["id"] == run_id
    assert snapshot["pipeline"]["discovered"] == 2
    assert snapshot["usage"]["totals"]["input_tokens"] == 100
    assert snapshot["usage"]["totals"]["cached_input_tokens"] == 20
    assert snapshot["usage"]["totals"]["reasoning_output_tokens"] == 3
    assert snapshot["usage"]["by_agent"][0]["agent"] == "semantic:cheap:0"
    assert snapshot["allowance"]["FIVE_HOUR"]["burn_percent_per_hour"] == pytest.approx(4.0)
    assert snapshot["allowance"]["FIVE_HOUR"]["eta_hours"] == pytest.approx(20.0)
    cheap = next(
        row for row in snapshot["agents"]
        if row["logical_agent_id"] == "semantic:cheap:0"
    )
    assert cheap["model"] == "gpt-5.6-luna"
    assert cheap["calls"] == 1
