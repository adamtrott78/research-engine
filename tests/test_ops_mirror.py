import json

from research_engine.ops_mirror import (
    build_chat_ops_mirror,
    render_chat_ops_markdown,
)


def _snapshot():
    return {
        "captured_at": "2026-10-01T22:30:00+00:00",
        "systemd": {
            "available": True,
            "ActiveState": "inactive",
            "SubState": "dead",
            "MainPID": "12345",
        },
        "run": {
            "id": "run:test",
            "topic": "SECRET_SOURCE_TOPIC",
            "status": "RUNNING",
            "current_phase": "VERIFY",
            "policy": {"secret": "SECRET_POLICY"},
            "knowledge_checkpoint_id": "checkpoint:test",
            "human_gate_state": "PENDING",
            "updated_at": "2026-10-01T22:29:00+00:00",
        },
        "allowance": {
            "FIVE_HOUR": {
                "observations": 4,
                "percent_used": 17.0,
                "percent_used_bounds": [16.0, 18.0],
                "percent_quantum": 1.0,
                "burn_percent_per_hour": None,
                "burn_percent_per_hour_bounds": [0.0, 2.0],
                "eta_hours": None,
                "eta_hours_bounds": [10.0, None],
                "reset_at": "2026-10-01T23:45:00+00:00",
                "confidence": "CALIBRATING",
                "approximate": True,
                "censored": True,
                "observed_at": "2026-10-01T22:25:00+00:00",
            },
            "WEEKLY": {
                "observations": 2,
                "percent_used": 5.0,
                "confidence": "CALIBRATING",
                "censored": True,
            },
        },
        "agents": [
            {
                "logical_agent_id": "semantic:standard:0",
                "worker_id": "SECRET_WORKER_ID",
                "host_name": "SECRET_HOST",
                "process_id": 4321,
                "code_sha": "abc123",
                "effective_state": "OFFLINE",
                "backend": "codex-chatgpt-plan",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "current_job_id": "SECRET_JOB_ID",
                "current_job_type": "VERIFY",
                "heartbeat_age_seconds": 8,
                "calls": 9,
                "input_tokens": 261800,
                "output_tokens": 2800,
            }
        ],
        "pipeline": {
            "discovered": 48,
            "retained": 9,
            "acquired": 9,
            "extracted": 9,
            "verified": 9,
            "checkpointed": 1,
            "syntheses": 0,
            "evidence_objects": 71,
            "gaps": 5,
            "novelty_search": {"COMPLETED": 4},
            "novelty_arbitrations": 2,
            "novelty_challenges": 3,
            "human_reviews": {"PENDING": 1},
        },
        "queue": {
            "by_status": {"COMPLETED": 25, "WAITING_HUMAN": 1},
            "oldest_queue_age_seconds": None,
            "expired_leases": 0,
            "oldest_pending_review_age_seconds": 120,
        },
        "throughput": {
            "1h": {
                "window_hours": 1,
                "acquired_per_hour": 1,
                "extracted_per_hour": 1,
                "verified_per_hour": 1,
                "evidence_per_hour": 7,
                "gaps_per_hour": 1,
                "semantic_calls_per_hour": 3,
                "input_tokens_per_hour": 90000,
                "output_tokens_per_hour": 2000,
            }
        },
        "usage": {
            "totals": {
                "calls": 28,
                "input_tokens": 959229,
                "cached_input_tokens": 240300,
                "output_tokens": 22030,
                "reasoning_output_tokens": 3500,
                "wall_time_ms": 280000,
                "web_search_count": 20,
            },
            "by_agent": [
                {
                    "agent": "semantic:standard:0",
                    "calls": 18,
                    "input_tokens": 485100,
                    "cached_input_tokens": 70100,
                    "output_tokens": 10200,
                    "reasoning_output_tokens": 1200,
                    "wall_time_ms": 180000,
                    "input_token_share": 0.5057,
                    "latest_model": "gpt-5.6-terra",
                    "latest_provider": "codex-chatgpt-plan",
                    "latest_reasoning_effort": "medium",
                }
            ],
            "by_model": [
                {
                    "executor_tier": "STANDARD",
                    "provider": "codex-chatgpt-plan",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "medium",
                    "calls": 18,
                    "input_tokens": 485100,
                    "cached_input_tokens": 70100,
                    "output_tokens": 10200,
                    "reasoning_output_tokens": 1200,
                    "wall_time_ms": 180000,
                }
            ],
            "by_job_type": [
                {
                    "job_type": "VERIFY",
                    "executor_tier": "STANDARD",
                    "calls": 9,
                    "input_tokens": 261800,
                    "cached_input_tokens": 36100,
                    "output_tokens": 2800,
                    "reasoning_output_tokens": 400,
                    "wall_time_ms": 90000,
                    "web_search_count": 0,
                }
            ],
        },
        "efficiency": {
            "input_tokens_per_verified_source": 106581,
            "total_tokens_per_verified_source": 109028,
            "semantic_calls_per_verified_source": 3.1,
            "total_tokens_per_evidence_object": 13820,
            "total_tokens_per_challenged_gap": 320000,
            "wall_minutes_per_verified_source": 0.52,
        },
        "calibrations": [
            {
                "calibration_type": "VERIFY_CONTEXT_WINDOW",
                "created_at": "2026-10-01T22:26:00+00:00",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "historical_input_tokens": 40439,
                "replay_input_tokens": 32509,
                "replay_cached_input_tokens": 0,
                "context_page_count": 7,
                "evidence_count": 11,
                "input_reduction_fraction": 0.1961,
                "outcome_agreement": {
                    "verified_equal": True,
                    "accepted_evidence_ids_equal": True,
                },
                "result_summary": {"raw": "SECRET_SOURCE_TEXT"},
            }
        ],
        "provider_backoffs": [
            {
                "provider": "codex-chatgpt-plan",
                "blocked_until": "2026-10-01T23:00:00+00:00",
                "reason": "plan limit",
                "detail": {"raw": "SECRET_PROVIDER_DETAIL"},
            }
        ],
        "alerts": ["daemon is inactive; persisted run is paused operationally"],
    }


def test_chat_ops_mirror_whitelists_operational_fields_and_excludes_sensitive_data():
    mirror = build_chat_ops_mirror(_snapshot())
    encoded = json.dumps(mirror, sort_keys=True)
    rendered = render_chat_ops_markdown(mirror)

    assert mirror["authority"]["raw_truth"] == "PostgreSQL"
    assert mirror["authority"]["scientific_state_mutation"] is False
    assert mirror["run"]["id"] == "run:test"
    assert mirror["agents"][0]["logical_agent_id"] == "semantic:standard:0"
    assert mirror["usage"]["by_job_type"][0]["job_type"] == "VERIFY"
    assert mirror["calibrations"][0]["accepted_evidence_ids_equal"] is True

    for secret_value in (
        "SECRET_SOURCE_TOPIC",
        "SECRET_POLICY",
        "SECRET_WORKER_ID",
        "SECRET_HOST",
        "SECRET_JOB_ID",
        "SECRET_SOURCE_TEXT",
        "SECRET_PROVIDER_DETAIL",
    ):
        assert secret_value not in encoded
        assert secret_value not in rendered

    for forbidden_key in (
        "MainPID",
        "process_id",
        "host_name",
        "worker_id",
        "current_job_id",
        "result_summary",
        "detail",
    ):
        assert forbidden_key not in encoded

    assert "Research Engine Live Ops" in rendered
    assert "17% used (censored)" in rendered
    assert "gpt-5.6-terra" in rendered
    assert "VERIFY" in rendered
    assert "PostgreSQL authoritative" in rendered


def test_chat_ops_mirror_is_bounded():
    snapshot = _snapshot()
    snapshot["agents"] = [
        {
            "logical_agent_id": f"agent:{i}",
            "effective_state": "IDLE",
            "calls": i,
            "input_tokens": i * 100,
            "output_tokens": i,
        }
        for i in range(30)
    ]
    snapshot["usage"]["by_agent"] = [
        {"agent": f"agent:{i}", "calls": i, "input_tokens": i * 100}
        for i in range(30)
    ]
    snapshot["calibrations"] = snapshot["calibrations"] * 20

    mirror = build_chat_ops_mirror(snapshot)
    rendered = render_chat_ops_markdown(mirror)

    assert len(mirror["agents"]) == 12
    assert len(mirror["usage"]["by_agent"]) == 12
    assert len(mirror["calibrations"]) == 6
    assert mirror["truncated"]["agents"] == [30, 12]
    assert mirror["truncated"]["calibrations"] == [20, 6]
    assert "agents 12/30" in rendered
    assert "calibrations 6/20" in rendered
    assert len(rendered) <= 50_000
