from __future__ import annotations

import json
from typing import Any

MIRROR_SCHEMA_VERSION = 1
MAX_MARKDOWN_CHARS = 50_000


def _text(value: Any, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _allowance(value: Any) -> dict[str, Any]:
    row = _mapping(value)
    keys = (
        "observations", "percent_used", "percent_used_bounds", "percent_quantum",
        "burn_percent_per_hour", "burn_percent_per_hour_bounds",
        "eta_hours", "eta_hours_bounds", "reset_at", "reset_precision_seconds",
        "confidence", "approximate", "censored", "observed_at",
    )
    return {key: row.get(key) for key in keys if key in row}


def _agent(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "logical_agent_id": _text(row.get("logical_agent_id"), 96),
        "state": _text(row.get("effective_state") or row.get("state"), 32),
        "backend": _text(row.get("backend"), 48),
        "model": _text(row.get("model"), 64),
        "reasoning_effort": _text(row.get("reasoning_effort"), 24),
        "current_job_type": _text(row.get("current_job_type"), 48),
        "current_job_elapsed_seconds": _float(row.get("current_job_elapsed_seconds")),
        "heartbeat_age_seconds": _float(row.get("heartbeat_age_seconds")),
        "calls": _int(row.get("calls")),
        "input_tokens": _int(row.get("input_tokens")),
        "output_tokens": _int(row.get("output_tokens")),
        "code_sha": _text(row.get("code_sha"), 64),
    }


def _usage_agent(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": _text(row.get("agent"), 96),
        "calls": _int(row.get("calls")),
        "input_tokens": _int(row.get("input_tokens")),
        "cached_input_tokens": _int(row.get("cached_input_tokens")),
        "output_tokens": _int(row.get("output_tokens")),
        "reasoning_output_tokens": _int(row.get("reasoning_output_tokens")),
        "wall_time_ms": _int(row.get("wall_time_ms")),
        "input_token_share": _float(row.get("input_token_share")),
        "latest_model": _text(row.get("latest_model"), 64),
        "latest_provider": _text(row.get("latest_provider"), 48),
        "latest_reasoning_effort": _text(row.get("latest_reasoning_effort"), 24),
    }


def _usage_model(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "executor_tier": _text(row.get("executor_tier"), 24),
        "provider": _text(row.get("provider"), 48),
        "model": _text(row.get("model"), 64),
        "reasoning_effort": _text(row.get("reasoning_effort"), 24),
        "calls": _int(row.get("calls")),
        "input_tokens": _int(row.get("input_tokens")),
        "cached_input_tokens": _int(row.get("cached_input_tokens")),
        "output_tokens": _int(row.get("output_tokens")),
        "reasoning_output_tokens": _int(row.get("reasoning_output_tokens")),
        "wall_time_ms": _int(row.get("wall_time_ms")),
    }


def _usage_stage(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_type": _text(row.get("job_type"), 48),
        "executor_tier": _text(row.get("executor_tier"), 24),
        "calls": _int(row.get("calls")),
        "input_tokens": _int(row.get("input_tokens")),
        "cached_input_tokens": _int(row.get("cached_input_tokens")),
        "output_tokens": _int(row.get("output_tokens")),
        "reasoning_output_tokens": _int(row.get("reasoning_output_tokens")),
        "wall_time_ms": _int(row.get("wall_time_ms")),
        "web_search_count": _int(row.get("web_search_count")),
    }


def _calibration(row: dict[str, Any]) -> dict[str, Any]:
    agreement = _mapping(row.get("outcome_agreement"))
    return {
        "calibration_type": _text(row.get("calibration_type"), 48),
        "created_at": _text(row.get("created_at"), 64),
        "model": _text(row.get("model"), 64),
        "reasoning_effort": _text(row.get("reasoning_effort"), 24),
        "historical_input_tokens": _int(row.get("historical_input_tokens")),
        "replay_input_tokens": _int(row.get("replay_input_tokens")),
        "replay_cached_input_tokens": _int(row.get("replay_cached_input_tokens")),
        "context_page_count": _int(row.get("context_page_count")),
        "evidence_count": _int(row.get("evidence_count")),
        "input_reduction_fraction": _float(row.get("input_reduction_fraction")),
        "verified_equal": bool(agreement.get("verified_equal")),
        "accepted_evidence_ids_equal": bool(
            agreement.get("accepted_evidence_ids_equal")
        ),
    }


def build_chat_ops_mirror(
    snapshot: dict[str, Any],
    *,
    max_agents: int = 12,
    max_models: int = 8,
    max_job_types: int = 24,
    max_calibrations: int = 6,
    max_alerts: int = 10,
) -> dict[str, Any]:
    run = _mapping(snapshot.get("run"))
    systemd = _mapping(snapshot.get("systemd"))
    usage = _mapping(snapshot.get("usage"))
    pipeline = _mapping(snapshot.get("pipeline"))
    queue = _mapping(snapshot.get("queue"))
    efficiency = _mapping(snapshot.get("efficiency"))
    totals = _mapping(usage.get("totals"))

    agents_all = [_agent(dict(x)) for x in snapshot.get("agents") or [] if isinstance(x, dict)]
    usage_agents_all = [_usage_agent(dict(x)) for x in usage.get("by_agent") or [] if isinstance(x, dict)]
    models_all = [_usage_model(dict(x)) for x in usage.get("by_model") or [] if isinstance(x, dict)]
    stages_all = [_usage_stage(dict(x)) for x in usage.get("by_job_type") or [] if isinstance(x, dict)]
    calibrations_all = [_calibration(dict(x)) for x in snapshot.get("calibrations") or [] if isinstance(x, dict)]
    alerts_all = [_text(x, 240) for x in snapshot.get("alerts") or []]
    alerts_all = [x for x in alerts_all if x]

    backoffs = []
    for raw in snapshot.get("provider_backoffs") or []:
        if isinstance(raw, dict):
            backoffs.append({
                "provider": _text(raw.get("provider"), 48),
                "blocked_until": _text(raw.get("blocked_until"), 64),
                "reason": _text(raw.get("reason"), 120),
            })

    throughput = {}
    for key, raw in _mapping(snapshot.get("throughput")).items():
        row = _mapping(raw)
        throughput[_text(key, 24) or "-"] = {
            "window_hours": _float(row.get("window_hours")),
            "acquired_per_hour": _float(row.get("acquired_per_hour")),
            "extracted_per_hour": _float(row.get("extracted_per_hour")),
            "verified_per_hour": _float(row.get("verified_per_hour")),
            "evidence_per_hour": _float(row.get("evidence_per_hour")),
            "gaps_per_hour": _float(row.get("gaps_per_hour")),
            "semantic_calls_per_hour": _float(row.get("semantic_calls_per_hour")),
            "input_tokens_per_hour": _float(row.get("input_tokens_per_hour")),
            "output_tokens_per_hour": _float(row.get("output_tokens_per_hour")),
        }

    return {
        "schema_version": MIRROR_SCHEMA_VERSION,
        "surface": "CHAT_VISIBLE_OPS_MIRROR",
        "authority": {
            "raw_truth": "PostgreSQL",
            "mirror_role": "derived read-only operational summary",
            "scientific_state_mutation": False,
        },
        "captured_at": _text(snapshot.get("captured_at"), 64),
        "daemon": {
            "available": bool(systemd.get("available")),
            "active_state": _text(systemd.get("ActiveState"), 32),
            "sub_state": _text(systemd.get("SubState"), 32),
        },
        "run": {
            "id": _text(run.get("id"), 128),
            "status": _text(run.get("status"), 48),
            "current_phase": _text(run.get("current_phase"), 64),
            "human_gate_state": _text(run.get("human_gate_state"), 48),
            "knowledge_checkpoint_id": _text(run.get("knowledge_checkpoint_id"), 160),
            "updated_at": _text(run.get("updated_at"), 64),
        },
        "allowance": {
            "FIVE_HOUR": _allowance(_mapping(snapshot.get("allowance")).get("FIVE_HOUR")),
            "WEEKLY": _allowance(_mapping(snapshot.get("allowance")).get("WEEKLY")),
        },
        "agents": agents_all[:max_agents],
        "pipeline": {
            "discovered": _int(pipeline.get("discovered")),
            "retained": _int(pipeline.get("retained")),
            "acquired": _int(pipeline.get("acquired")),
            "extracted": _int(pipeline.get("extracted")),
            "verified": _int(pipeline.get("verified")),
            "checkpointed": _int(pipeline.get("checkpointed")),
            "syntheses": _int(pipeline.get("syntheses")),
            "evidence_objects": _int(pipeline.get("evidence_objects")),
            "gaps": _int(pipeline.get("gaps")),
            "novelty_search": _mapping(pipeline.get("novelty_search")),
            "novelty_arbitrations": _int(pipeline.get("novelty_arbitrations")),
            "novelty_challenges": _int(pipeline.get("novelty_challenges")),
            "human_reviews": _mapping(pipeline.get("human_reviews")),
        },
        "queue": {
            "by_status": _mapping(queue.get("by_status")),
            "oldest_queue_age_seconds": _float(queue.get("oldest_queue_age_seconds")),
            "expired_leases": _int(queue.get("expired_leases")),
            "oldest_pending_review_age_seconds": _float(
                queue.get("oldest_pending_review_age_seconds")
            ),
        },
        "throughput": throughput,
        "usage": {
            "totals": {
                "calls": _int(totals.get("calls")),
                "input_tokens": _int(totals.get("input_tokens")),
                "cached_input_tokens": _int(totals.get("cached_input_tokens")),
                "output_tokens": _int(totals.get("output_tokens")),
                "reasoning_output_tokens": _int(totals.get("reasoning_output_tokens")),
                "wall_time_ms": _int(totals.get("wall_time_ms")),
                "web_search_count": _int(totals.get("web_search_count")),
            },
            "by_agent": usage_agents_all[:max_agents],
            "by_model": models_all[:max_models],
            "by_job_type": stages_all[:max_job_types],
        },
        "efficiency": {
            key: _float(efficiency.get(key))
            for key in (
                "input_tokens_per_verified_source",
                "total_tokens_per_verified_source",
                "semantic_calls_per_verified_source",
                "total_tokens_per_evidence_object",
                "total_tokens_per_challenged_gap",
                "wall_minutes_per_verified_source",
            )
        },
        "calibrations": calibrations_all[:max_calibrations],
        "provider_backoffs": backoffs[:8],
        "alerts": alerts_all[:max_alerts],
        "truncated": {
            "agents": [len(agents_all), min(len(agents_all), max_agents)],
            "usage_by_agent": [len(usage_agents_all), min(len(usage_agents_all), max_agents)],
            "usage_by_model": [len(models_all), min(len(models_all), max_models)],
            "usage_by_job_type": [len(stages_all), min(len(stages_all), max_job_types)],
            "calibrations": [len(calibrations_all), min(len(calibrations_all), max_calibrations)],
            "alerts": [len(alerts_all), min(len(alerts_all), max_alerts)],
        },
    }


def _md(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _pct(value: Any) -> str:
    number = _float(value)
    return "-" if number is None else f"{100 * number:.1f}%"


def _duration(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    seconds = max(0, int(number))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _allowance_line(label: str, row: dict[str, Any]) -> str:
    if row.get("percent_used") is None:
        return f"| {label} | no observation | - | - | - |"
    used = f"{float(row['percent_used']):.0f}% used"
    if row.get("censored"):
        used += " (censored)"
    burn = row.get("burn_percent_per_hour")
    if burn is not None:
        burn_text = f"{float(burn):.2f}%/h"
    elif row.get("burn_percent_per_hour_bounds") is not None:
        low, high = row["burn_percent_per_hour_bounds"]
        burn_text = f"{float(low):.2f}-{float(high):.2f}%/h censored"
    else:
        burn_text = "-"
    return f"| {label} | {_md(used)} | {_md(row.get('confidence'))} | {_md(burn_text)} | {_md(row.get('reset_at'))} |"


def render_chat_ops_markdown(mirror: dict[str, Any]) -> str:
    run = _mapping(mirror.get("run"))
    daemon = _mapping(mirror.get("daemon"))
    pipeline = _mapping(mirror.get("pipeline"))
    queue = _mapping(mirror.get("queue"))
    usage = _mapping(mirror.get("usage"))
    totals = _mapping(usage.get("totals"))

    lines = [
        "# Research Engine Live Ops",
        "",
        "> Derived read-only operational mirror. PostgreSQL remains authoritative. "
        "Prompts, source text, credentials, environment values, host/process "
        "identifiers, and provider error detail are intentionally excluded.",
        "",
        f"Captured: {_md(mirror.get('captured_at'))}",
        f"Daemon: {_md(daemon.get('active_state'))}/{_md(daemon.get('sub_state'))}",
        f"Run: {_md(run.get('id'))} — {_md(run.get('status'))} / {_md(run.get('current_phase'))} — gate {_md(run.get('human_gate_state'))}",
        "",
        "## Allowance",
        "",
        "| Window | Observation | Confidence | Burn estimate | Reset |",
        "|---|---:|---|---:|---|",
        _allowance_line("5-hour", _mapping(_mapping(mirror.get("allowance")).get("FIVE_HOUR"))),
        _allowance_line("Weekly", _mapping(_mapping(mirror.get("allowance")).get("WEEKLY"))),
        "",
        "## Pipeline",
        "",
        "| Discovered | Retained | Acquired | Extracted | Verified | Evidence | Checkpointed | Syntheses | Gaps | Novelty arbitrated |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_n(pipeline.get('discovered'))} | {_n(pipeline.get('retained'))} | {_n(pipeline.get('acquired'))} | {_n(pipeline.get('extracted'))} | {_n(pipeline.get('verified'))} | {_n(pipeline.get('evidence_objects'))} | {_n(pipeline.get('checkpointed'))} | {_n(pipeline.get('syntheses'))} | {_n(pipeline.get('gaps'))} | {_n(pipeline.get('novelty_arbitrations'))} |",
        "",
        "## Queue",
        "",
        f"- Status: {json.dumps(_mapping(queue.get('by_status')), sort_keys=True)}",
        f"- Oldest queued: {_duration(queue.get('oldest_queue_age_seconds'))}",
        f"- Expired leases: {_n(queue.get('expired_leases'))}",
        f"- Oldest pending human review: {_duration(queue.get('oldest_pending_review_age_seconds'))}",
        "",
        "## Agents",
        "",
        "| Agent | State | Model | Job | Heartbeat | Calls | Input | Output |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    agents = list(mirror.get("agents") or [])
    if not agents:
        lines.append("| none | - | - | - | - | 0 | 0 | 0 |")
    for raw in agents:
        row = _mapping(raw)
        lines.append(
            f"| {_md(row.get('logical_agent_id'))} | {_md(row.get('state'))} | "
            f"{_md(row.get('model') or row.get('backend'))} | {_md(row.get('current_job_type'))} | "
            f"{_duration(row.get('heartbeat_age_seconds'))} | {_n(row.get('calls'))} | "
            f"{_n(row.get('input_tokens'))} | {_n(row.get('output_tokens'))} |"
        )

    lines.extend([
        "",
        "## Usage",
        "",
        f"Totals: {_n(totals.get('calls'))} semantic calls · {_n(totals.get('input_tokens'))} input ({_n(totals.get('cached_input_tokens'))} cached) · {_n(totals.get('output_tokens'))} output · {_n(totals.get('reasoning_output_tokens'))} reasoning output",
        "",
        "### By stage",
        "",
        "| Stage | Tier | Calls | Input | Cached | Output | Reasoning | Web |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for raw in usage.get("by_job_type") or []:
        row = _mapping(raw)
        lines.append(
            f"| {_md(row.get('job_type'))} | {_md(row.get('executor_tier'))} | "
            f"{_n(row.get('calls'))} | {_n(row.get('input_tokens'))} | {_n(row.get('cached_input_tokens'))} | "
            f"{_n(row.get('output_tokens'))} | {_n(row.get('reasoning_output_tokens'))} | {_n(row.get('web_search_count'))} |"
        )

    lines.extend([
        "",
        "### By model",
        "",
        "| Tier | Provider | Model | Reasoning | Calls | Input | Cached | Output |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for raw in usage.get("by_model") or []:
        row = _mapping(raw)
        lines.append(
            f"| {_md(row.get('executor_tier'))} | {_md(row.get('provider'))} | {_md(row.get('model'))} | "
            f"{_md(row.get('reasoning_effort'))} | {_n(row.get('calls'))} | {_n(row.get('input_tokens'))} | "
            f"{_n(row.get('cached_input_tokens'))} | {_n(row.get('output_tokens'))} |"
        )

    lines.extend([
        "",
        "## Throughput",
        "",
        "| Window | Acquire/h | Extract/h | Verify/h | Evidence/h | Calls/h | Input/h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, raw in _mapping(mirror.get("throughput")).items():
        row = _mapping(raw)
        lines.append(
            f"| {_md(key)} | {(_float(row.get('acquired_per_hour')) or 0):.2f} | "
            f"{(_float(row.get('extracted_per_hour')) or 0):.2f} | {(_float(row.get('verified_per_hour')) or 0):.2f} | "
            f"{(_float(row.get('evidence_per_hour')) or 0):.2f} | {(_float(row.get('semantic_calls_per_hour')) or 0):.2f} | "
            f"{_n(_float(row.get('input_tokens_per_hour')) or 0)} |"
        )

    calibrations = list(mirror.get("calibrations") or [])
    if calibrations:
        lines.extend([
            "",
            "## Recent calibration",
            "",
            "| Type | Model | Historical input | Replay input | Saved | Pages | Evidence | Agreement |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for raw in calibrations:
            row = _mapping(raw)
            agree = "PASS" if row.get("verified_equal") and row.get("accepted_evidence_ids_equal") else "CHECK"
            lines.append(
                f"| {_md(row.get('calibration_type'))} | {_md(row.get('model'))} | {_n(row.get('historical_input_tokens'))} | "
                f"{_n(row.get('replay_input_tokens'))} | {_pct(row.get('input_reduction_fraction'))} | "
                f"{_n(row.get('context_page_count'))} | {_n(row.get('evidence_count'))} | {agree} |"
            )

    lines.extend(["", "## Alerts / backoffs", ""])
    alerts = list(mirror.get("alerts") or [])
    backoffs = list(mirror.get("provider_backoffs") or [])
    if not alerts and not backoffs:
        lines.append("- No active alerts.")
    else:
        lines.extend(f"- {_md(value)}" for value in alerts)
        for raw in backoffs:
            row = _mapping(raw)
            lines.append(
                f"- Provider backoff: {_md(row.get('provider'))} until {_md(row.get('blocked_until'))} ({_md(row.get('reason'))})"
            )

    clipped = []
    for key, counts in _mapping(mirror.get("truncated")).items():
        if isinstance(counts, list) and len(counts) == 2 and int(counts[0]) > int(counts[1]):
            clipped.append(f"{key} {counts[1]}/{counts[0]}")
    if clipped:
        lines.extend(["", "> Bounded mirror truncation: " + ", ".join(clipped) + "."])

    lines.extend(["", "---", "Mirror schema v1 · read-only derived surface · PostgreSQL authoritative."])
    rendered = "\n".join(lines)
    if len(rendered) > MAX_MARKDOWN_CHARS:
        rendered = rendered[: MAX_MARKDOWN_CHARS - 120] + "\n\n> Mirror hard-truncated at 50,000 characters.\n"
    return rendered
