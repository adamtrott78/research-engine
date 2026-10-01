from datetime import datetime, timedelta, timezone

from research_engine.ops import (
    _forecast_observations,
    parse_allowance_ui_paste,
    parse_relative_duration,
    relative_duration_precision_seconds,
    render_hud,
)




def test_parse_allowance_ui_paste_from_copied_product_block():
    parsed = parse_allowance_ui_paste(
        "5-hour limit\r\n"
        "Resets in 1h 53m\r\n"
        "84% left\r\n"
        "\r\n"
        "Weekly limit\r\n"
        "Resets in 5d 7h\r\n"
        "95% left\r\n"
    )
    assert parsed == {
        "5h": {"left_percent": 84.0, "resets_in": "1h 53m"},
        "weekly": {"left_percent": 95.0, "resets_in": "5d 7h"},
    }


def test_parse_allowance_ui_paste_accepts_markdown_wrapped_labels():
    parsed = parse_allowance_ui_paste(
        "**5-hour limit**\nResets in 1h 55m\n84% left\n"
        "**Weekly limit**\nResets in 5d 7h\n95% left\n"
    )
    assert parsed["5h"]["left_percent"] == 84.0
    assert parsed["weekly"]["resets_in"] == "5d 7h"


def test_parse_relative_allowance_reset_duration():
    assert parse_relative_duration("3h 20m") == timedelta(hours=3, minutes=20)
    assert parse_relative_duration("5d 8h") == timedelta(days=5, hours=8)
    assert parse_relative_duration("45m") == timedelta(minutes=45)
    assert parse_relative_duration("3h20m") == timedelta(hours=3, minutes=20)
    assert relative_duration_precision_seconds("3h 20m") == 60
    assert relative_duration_precision_seconds("5d 8h") == 3600
    assert relative_duration_precision_seconds("45m") == 60


def test_allowance_forecast_groups_reset_jitter_and_censors_whole_percent_ui():
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    rows = [
        {
            "window_type": "FIVE_HOUR",
            "percent_used": 14.0,
            "percent_quantum": 1.0,
            "observed_at": start,
            "reset_at": start + timedelta(hours=3, minutes=20),
            "reset_precision_seconds": 60,
            "approximate": True,
        },
        {
            "window_type": "FIVE_HOUR",
            "percent_used": 15.0,
            "percent_quantum": 1.0,
            "observed_at": start + timedelta(minutes=15),
            "reset_at": start + timedelta(hours=3, minutes=20, seconds=35),
            "reset_precision_seconds": 60,
            "approximate": True,
        },
    ]
    forecast = _forecast_observations(
        rows,
        now=start + timedelta(minutes=15),
        window_type="FIVE_HOUR",
    )
    assert forecast["observations"] == 2
    assert forecast["censored"] is True
    assert forecast["burn_percent_per_hour"] is None
    assert forecast["burn_percent_per_hour_bounds"] == [0.0, 12.0]
    assert forecast["percent_used_bounds"] == [14.0, 16.0]
    assert forecast["reset_window_tolerance_seconds"] == 120.0


def test_allowance_forecast_still_splits_distinct_reset_windows():
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    rows = [
        {
            "window_type": "FIVE_HOUR",
            "percent_used": 14.0,
            "observed_at": start,
            "reset_at": start + timedelta(hours=3),
            "reset_precision_seconds": 60,
            "approximate": True,
        },
        {
            "window_type": "FIVE_HOUR",
            "percent_used": 15.0,
            "observed_at": start + timedelta(minutes=15),
            "reset_at": start + timedelta(hours=8),
            "reset_precision_seconds": 60,
            "approximate": True,
        },
    ]
    forecast = _forecast_observations(
        rows,
        now=start + timedelta(minutes=15),
        window_type="FIVE_HOUR",
    )
    assert forecast["observations"] == 1
    assert forecast["confidence"] == "CALIBRATING"


def test_allowance_forecast_calibrates_then_estimates():
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    one = _forecast_observations(
        [
            {
                "percent_used": 16.0,
                "observed_at": start,
                "reset_at": None,
                "approximate": True,
            }
        ],
        now=start,
    )
    assert one["confidence"] == "CALIBRATING"
    assert one["burn_percent_per_hour"] is None
    assert one["eta_hours"] is None

    two = _forecast_observations(
        [
            {
                "percent_used": 16.0,
                "observed_at": start,
                "reset_at": None,
                "approximate": True,
            },
            {
                "percent_used": 20.0,
                "observed_at": start + timedelta(hours=1),
                "reset_at": None,
                "approximate": False,
            },
        ],
        now=start + timedelta(hours=1),
    )
    assert two["confidence"] == "LOW"
    assert two["burn_percent_per_hour"] == 4.0
    assert two["eta_hours"] == 20.0


def test_allowance_forecast_does_not_cross_reset_drop():
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    rows = [
        {
            "percent_used": 80.0,
            "observed_at": start,
            "reset_at": None,
            "approximate": False,
        },
        {
            "percent_used": 5.0,
            "observed_at": start + timedelta(hours=1),
            "reset_at": None,
            "approximate": False,
        },
        {
            "percent_used": 9.0,
            "observed_at": start + timedelta(hours=2),
            "reset_at": None,
            "approximate": False,
        },
    ]
    forecast = _forecast_observations(rows, now=start + timedelta(hours=2))
    assert forecast["observations"] == 2
    assert forecast["burn_percent_per_hour"] == 4.0


def test_hud_renders_core_panels():
    snapshot = {
        "captured_at": "2026-10-01T16:00:00+00:00",
        "systemd": {
            "available": True,
            "ActiveState": "inactive",
            "SubState": "dead",
        },
        "run": {
            "id": "run:test",
            "status": "RUNNING",
            "current_phase": "DISCOVERY",
            "human_gate_state": "NOT_REQUIRED",
        },
        "allowance": {
            "FIVE_HOUR": {
                "percent_used": 16.0,
                "confidence": "CALIBRATING",
                "burn_percent_per_hour": None,
                "eta_hours": None,
                "reset_at": None,
            },
            "WEEKLY": {
                "percent_used": None,
                "confidence": "NO_DATA",
            },
        },
        "agents": [
            {
                "logical_agent_id": "semantic:cheap:0",
                "state": "ACTIVE",
                "model": "gpt-5.6-luna",
                "current_job_type": "DISCOVERY",
                "current_job_elapsed_seconds": 5,
                "heartbeat_age_seconds": 2,
                "calls": 1,
                "input_tokens": 100,
                "output_tokens": 10,
            }
        ],
        "pipeline": {
            "discovered": 2,
            "retained": 0,
            "acquired": 0,
            "extracted": 0,
            "verified": 0,
            "checkpointed": 0,
            "syntheses": 0,
            "evidence_objects": 0,
            "gaps": 0,
            "novelty_search": {},
            "novelty_arbitrations": 0,
        },
        "throughput": {},
        "queue": {
            "by_status": {"RUNNING": 1},
            "oldest_queue_age_seconds": None,
            "expired_leases": 0,
        },
        "usage": {
            "totals": {
                "calls": 1,
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reasoning_output_tokens": 3,
            }
        },
        "efficiency": {},
        "alerts": [],
    }
    rendered = render_hud(snapshot, color=False)
    assert "RESEARCH ENGINE HUD" in rendered
    assert "ALLOWANCE" in rendered
    assert "16.0% used" in rendered
    assert "CALIBRATING" in rendered
    assert "semantic:cheap:0" in rendered
    assert "PIPELINE" in rendered
    assert "QUEUE / ALERTS" in rendered
