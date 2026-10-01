import os
from dataclasses import replace
from pathlib import Path

import pytest


class FakeHydrator:
    def hydrate(self, job):
        payload = dict(job.payload)
        payload["_research_context"] = {
            "pages": [
                {"page_number": 3, "text": "grounded source page"},
            ],
            "authoritative_evidence": [
                {
                    "id": "evidence:1",
                    "page_number": 3,
                    "raw_source_form": "grounded source page",
                    "normalized_form": "grounded claim",
                }
            ],
            "verification_scope": {
                "cited_page_numbers": [3],
                "page_radius": 0,
                "supplied_page_numbers": [3],
            },
        }
        return replace(job, payload=payload)


class FakeDelegate:
    def execute(self, job):
        assert [p["page_number"] for p in job.payload["_research_context"]["pages"]] == [3]
        return {
            "verified": True,
            "item_id": "paper:1",
            "accepted_evidence_ids": ["evidence:1"],
            "issues": [],
            "_execution": {
                "provider": "codex-chatgpt-plan",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "usage": {
                    "input_tokens": 12000,
                    "cached_input_tokens": 2000,
                    "output_tokens": 400,
                    "reasoning_output_tokens": 50,
                },
            },
        }


@pytest.mark.integration
def test_verify_replay_calibration_is_operational_only(tmp_path: Path):
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from psycopg.types.json import Jsonb

    from research_engine.calibration import VerifyReplayCalibration
    from research_engine.ops import OpsService
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")

    run_id = "run:calibration"
    job_id = "job:verify-calibration-source"
    source_result = {
        "verified": True,
        "item_id": "paper:1",
        "accepted_evidence_ids": ["evidence:1"],
        "issues": [],
        "_execution": {
            "provider": "codex-chatgpt-plan",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "usage": {
                "input_tokens": 30000,
                "cached_input_tokens": 5000,
                "output_tokens": 900,
                "reasoning_output_tokens": 100,
            },
        },
    }

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE research_calibration_events,
                         research_allowance_observations,
                         research_usage_events,
                         human_review_requests,
                         semantic_materializations,
                         research_job_attempts,
                         research_jobs,
                         research_workers,
                         research_runs CASCADE
                """
            )
            cur.execute(
                """
                INSERT INTO research_runs(
                  id,topic,status,current_phase,policy,human_gate_state
                )
                VALUES (
                  %s,'calibration source','WAITING_HUMAN','HUMAN_REVIEW',
                  '{}'::jsonb,'PENDING'
                )
                """,
                (run_id,),
            )
            cur.execute(
                """
                INSERT INTO research_jobs(
                  id,run_id,job_type,executor_tier,status,priority,
                  idempotency_key,payload,result,depends_on,
                  attempt_count,max_attempts
                )
                VALUES (
                  %s,%s,'VERIFY','STANDARD','COMPLETED',0,
                  'verify-source',%s,%s,'{}',1,3
                )
                """,
                (
                    job_id,
                    run_id,
                    Jsonb({"item": {"id": "paper:1"}}),
                    Jsonb(source_result),
                ),
            )
            cur.execute(
                """
                INSERT INTO research_usage_events(
                  run_id,job_id,job_type,executor_tier,
                  provider,model,reasoning_effort,
                  input_tokens,cached_input_tokens,
                  output_tokens,reasoning_output_tokens
                )
                VALUES (
                  %s,%s,'VERIFY','STANDARD',
                  'codex-chatgpt-plan','gpt-5.6-terra','medium',
                  30000,5000,900,100
                )
                """,
                (run_id, job_id),
            )
            for extra_id, tokens in (
                ("job:verify-low", 10000),
                ("job:verify-high", 50000),
            ):
                cur.execute(
                    """
                    INSERT INTO research_jobs(
                      id,run_id,job_type,executor_tier,status,priority,
                      idempotency_key,payload,result,depends_on,
                      attempt_count,max_attempts
                    )
                    VALUES (
                      %s,%s,'VERIFY','STANDARD','COMPLETED',0,
                      %s,%s,%s,'{}',1,3
                    )
                    """,
                    (
                        extra_id,
                        run_id,
                        extra_id,
                        Jsonb({"item": {"id": "paper:1"}}),
                        Jsonb(source_result),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO research_usage_events(
                      run_id,job_id,job_type,executor_tier,
                      provider,model,reasoning_effort,
                      input_tokens,cached_input_tokens,
                      output_tokens,reasoning_output_tokens
                    )
                    VALUES (
                      %s,%s,'VERIFY','STANDARD',
                      'codex-chatgpt-plan','gpt-5.6-terra','medium',
                      %s,0,100,10
                    )
                    """,
                    (run_id, extra_id, tokens),
                )
            cur.execute(
                """
                SELECT result,status,payload
                FROM research_jobs WHERE id=%s
                """,
                (job_id,),
            )
            before_job = cur.fetchone()
            cur.execute("SELECT count(*)::int FROM evidence_objects")
            before_evidence = cur.fetchone()[0]
            cur.execute("SELECT count(*)::int FROM verification_events")
            before_verification = cur.fetchone()[0]
            cur.execute("SELECT count(*)::int FROM knowledge_checkpoints")
            before_checkpoints = cur.fetchone()[0]
        conn.commit()

    service = VerifyReplayCalibration(
        store=store,
        artifact_root=tmp_path,
        delegate=FakeDelegate(),
        verify_page_radius=0,
    )
    service.hydrator = FakeHydrator()

    assert service.select_source_job(
        run_id=run_id,
        selector="lowest",
    ).id == "job:verify-low"
    assert service.select_source_job(
        run_id=run_id,
        selector="median",
    ).id == job_id
    assert service.select_source_job(
        run_id=run_id,
        selector="highest",
    ).id == "job:verify-high"

    result = service.run(run_id=run_id)

    assert result["source_job_id"] == job_id
    assert result["selector"] == "median"
    assert result["context_page_count"] == 1
    assert result["context_page_numbers"] == [3]
    assert result["evidence_count"] == 1
    assert result["historical"]["input_tokens"] == 30000
    assert result["replay"]["input_tokens"] == 12000
    assert result["input_reduction_fraction"] == pytest.approx(0.6)
    assert result["uncached_input_reduction_fraction"] == pytest.approx(0.6)
    assert result["agreement"]["verified_equal"] is True
    assert result["agreement"]["accepted_evidence_ids_equal"] is True

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT result,status,payload
                FROM research_jobs WHERE id=%s
                """,
                (job_id,),
            )
            after_job = cur.fetchone()
            cur.execute("SELECT count(*)::int FROM evidence_objects")
            after_evidence = cur.fetchone()[0]
            cur.execute("SELECT count(*)::int FROM verification_events")
            after_verification = cur.fetchone()[0]
            cur.execute("SELECT count(*)::int FROM knowledge_checkpoints")
            after_checkpoints = cur.fetchone()[0]
            cur.execute(
                """
                SELECT calibration_type,source_run_id,source_job_id,
                       historical_input_tokens,replay_input_tokens,
                       context_page_count,evidence_count,outcome_agreement
                FROM research_calibration_events
                """
            )
            calibration = cur.fetchone()

    assert after_job == before_job
    assert after_evidence == before_evidence
    assert after_verification == before_verification
    assert after_checkpoints == before_checkpoints
    assert calibration[0] == "VERIFY_CONTEXT_WINDOW"
    assert calibration[1] == run_id
    assert calibration[2] == job_id
    assert calibration[3] == 30000
    assert calibration[4] == 12000
    assert calibration[5] == 1
    assert calibration[6] == 1
    assert calibration[7]["verified_equal"] is True
    assert calibration[7]["accepted_evidence_ids_equal"] is True

    snapshot = OpsService(store).snapshot(run_id=run_id)
    latest = snapshot["calibrations"][0]
    assert latest["calibration_type"] == "VERIFY_CONTEXT_WINDOW"
    assert latest["historical_input_tokens"] == 30000
    assert latest["replay_input_tokens"] == 12000
    assert latest["input_reduction_fraction"] == pytest.approx(0.6)
