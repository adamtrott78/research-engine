from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .postgres import PostgresStore


_RELATIVE_DURATION_PATTERN = re.compile(
    r"^\s*(?:(?P<days>\d+)\s*d(?:ays?)?\s*)?"
    r"(?:(?P<hours>\d+)\s*h(?:ours?)?\s*)?"
    r"(?:(?P<minutes>\d+)\s*m(?:in(?:ute)?s?)?\s*)?"
    r"(?:(?P<seconds>\d+)\s*s(?:ec(?:ond)?s?)?\s*)?$",
    re.IGNORECASE,
)


def parse_relative_duration(value: str) -> timedelta:
    match = _RELATIVE_DURATION_PATTERN.fullmatch(value or "")
    if match is None or not any(match.groupdict().values()):
        raise ValueError(
            "relative duration must look like '3h 20m', '5d 8h', or '45m'"
        )
    parts = {key: int(raw or 0) for key, raw in match.groupdict().items()}
    duration = timedelta(
        days=parts["days"],
        hours=parts["hours"],
        minutes=parts["minutes"],
        seconds=parts["seconds"],
    )
    if duration.total_seconds() <= 0:
        raise ValueError("relative duration must be positive")
    return duration


WINDOW_ALIASES = {
    "5h": "FIVE_HOUR",
    "five_hour": "FIVE_HOUR",
    "FIVE_HOUR": "FIVE_HOUR",
    "weekly": "WEEKLY",
    "week": "WEEKLY",
    "WEEKLY": "WEEKLY",
}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _dict_rows(cur) -> list[dict[str, Any]]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _one_dict(cur) -> dict[str, Any] | None:
    rows = _dict_rows(cur)
    return rows[0] if rows else None


def _seconds(delta) -> float:
    return max(0.0, float(delta.total_seconds()))


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _fmt_count(value: int | float | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}k"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _forecast_observations(
    rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    ordered = sorted(rows, key=lambda row: row["observed_at"])
    latest = ordered[-1] if ordered else None
    if latest is None:
        return {
            "observations": 0,
            "percent_used": None,
            "burn_percent_per_hour": None,
            "eta_hours": None,
            "reset_at": None,
            "confidence": "NO_DATA",
            "approximate": None,
        }

    segment = [latest]
    for previous in reversed(ordered[:-1]):
        newer = segment[-1]
        if previous["percent_used"] > newer["percent_used"]:
            break
        prev_reset = previous.get("reset_at")
        new_reset = newer.get("reset_at")
        if prev_reset and new_reset and prev_reset != new_reset:
            break
        segment.append(previous)
    segment.reverse()

    result = {
        "observations": len(segment),
        "percent_used": float(latest["percent_used"]),
        "burn_percent_per_hour": None,
        "eta_hours": None,
        "reset_at": _iso(latest.get("reset_at")),
        "confidence": "CALIBRATING",
        "approximate": bool(latest.get("approximate")),
        "observed_at": _iso(latest["observed_at"]),
    }
    if len(segment) < 2:
        return result

    elapsed_hours = (
        segment[-1]["observed_at"] - segment[0]["observed_at"]
    ).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        return result
    delta = float(segment[-1]["percent_used"]) - float(segment[0]["percent_used"])
    burn = delta / elapsed_hours
    if burn <= 0:
        result["confidence"] = "LOW"
        result["burn_percent_per_hour"] = max(0.0, burn)
        return result

    interval_rates = []
    for left, right in zip(segment, segment[1:]):
        hours = (right["observed_at"] - left["observed_at"]).total_seconds() / 3600.0
        if hours > 0:
            interval_rates.append(
                (float(right["percent_used"]) - float(left["percent_used"])) / hours
            )

    result["burn_percent_per_hour"] = burn
    result["eta_hours"] = max(0.0, (100.0 - float(latest["percent_used"])) / burn)

    confidence = "LOW"
    if len(segment) >= 3 and elapsed_hours >= 0.5:
        confidence = "MEDIUM"
    if len(segment) >= 5 and elapsed_hours >= 2 and interval_rates:
        mean = sum(interval_rates) / len(interval_rates)
        variance = sum((x - mean) ** 2 for x in interval_rates) / len(interval_rates)
        cv = math.sqrt(variance) / abs(mean) if mean else math.inf
        if cv <= 0.25:
            confidence = "HIGH"
    result["confidence"] = confidence
    return result


def _systemd_state() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "research-engine-daemon.service",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "Result",
                "-p",
                "MainPID",
                "-p",
                "NRestarts",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if proc.returncode != 0:
        return {"available": False, "error": (proc.stderr or "").strip()[:300]}
    result: dict[str, Any] = {"available": True}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


class OpsService:
    def __init__(self, store: PostgresStore):
        self.store = store

    def observe_allowance(
        self,
        *,
        window: str,
        percent_used: float,
        reset_at: datetime | None = None,
        note: str | None = None,
        approximate: bool = False,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        window_type = WINDOW_ALIASES.get(window)
        if window_type is None:
            raise ValueError(f"unsupported allowance window: {window}")
        if not 0 <= float(percent_used) <= 100:
            raise ValueError("percent_used must be in [0,100]")
        observed_at = observed_at or datetime.now(timezone.utc)

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT max(id) FROM research_usage_events")
                highwater = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO research_allowance_observations(
                      window_type,percent_used,observed_at,reset_at,
                      source,approximate,note,usage_event_highwater
                    )
                    VALUES (%s,%s,%s,%s,'MANUAL_UI',%s,%s,%s)
                    RETURNING id,window_type,percent_used,observed_at,reset_at,
                              source,approximate,note,usage_event_highwater
                    """,
                    (
                        window_type,
                        float(percent_used),
                        observed_at,
                        reset_at,
                        approximate,
                        note,
                        highwater,
                    ),
                )
                row = _one_dict(cur)
            conn.commit()
        assert row is not None
        return {key: _iso(value) if isinstance(value, datetime) else value for key, value in row.items()}

    def _selected_run(self, cur, run_id: str | None) -> dict[str, Any] | None:
        if run_id:
            cur.execute(
                """
                SELECT id,topic,status,current_phase,policy,knowledge_checkpoint_id,
                       human_gate_state,created_at,updated_at
                FROM research_runs WHERE id=%s
                """,
                (run_id,),
            )
            return _one_dict(cur)
        cur.execute(
            """
            SELECT id,topic,status,current_phase,policy,knowledge_checkpoint_id,
                   human_gate_state,created_at,updated_at
            FROM research_runs
            ORDER BY
              CASE status
                WHEN 'RUNNING' THEN 0
                WHEN 'WAITING_HUMAN' THEN 1
                WHEN 'PAUSED_BUDGET' THEN 2
                ELSE 3
              END,
              updated_at DESC,id
            LIMIT 1
            """
        )
        return _one_dict(cur)

    def snapshot(
        self,
        *,
        run_id: str | None = None,
        windows: Iterable[float] = (1.0, 5.0),
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        window_values = tuple(sorted({float(x) for x in windows if float(x) > 0}))
        if not window_values:
            window_values = (1.0, 5.0)

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                run = self._selected_run(cur, run_id)
                selected = run["id"] if run else None

                cur.execute(
                    """
                    SELECT window_type,percent_used,observed_at,reset_at,
                           approximate,note,usage_event_highwater
                    FROM research_allowance_observations
                    ORDER BY window_type,observed_at,id
                    """
                )
                allowance_rows = _dict_rows(cur)

                cur.execute(
                    """
                    SELECT provider,blocked_until,reason,detail
                    FROM research_provider_backoffs
                    WHERE blocked_until > now()
                    ORDER BY provider
                    """
                )
                backoffs = _dict_rows(cur)

                agents: list[dict[str, Any]] = []
                usage_by_agent: list[dict[str, Any]] = []
                usage_by_model: list[dict[str, Any]] = []
                pipeline: dict[str, Any] = {}
                queue: dict[str, Any] = {}
                throughputs: dict[str, Any] = {}
                usage_totals: dict[str, Any] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0,
                    "wall_time_ms": 0,
                    "web_search_count": 0,
                }

                if selected:
                    cur.execute(
                        """
                        SELECT status,count(*)::int
                        FROM research_jobs
                        WHERE run_id=%s
                        GROUP BY status
                        ORDER BY status
                        """,
                        (selected,),
                    )
                    status_counts = dict(cur.fetchall())
                    cur.execute(
                        """
                        SELECT EXTRACT(EPOCH FROM (now()-min(available_at)))
                        FROM research_jobs
                        WHERE run_id=%s AND status='QUEUED'
                        """,
                        (selected,),
                    )
                    oldest_queue = cur.fetchone()[0]
                    cur.execute(
                        """
                        SELECT count(*)::int
                        FROM research_jobs
                        WHERE run_id=%s AND status='RUNNING'
                          AND lease_expires_at <= now()
                        """,
                        (selected,),
                    )
                    expired = cur.fetchone()[0]
                    cur.execute(
                        """
                        SELECT EXTRACT(EPOCH FROM (now()-min(created_at)))
                        FROM human_review_requests
                        WHERE run_id=%s AND status='PENDING'
                        """,
                        (selected,),
                    )
                    oldest_review = cur.fetchone()[0]
                    queue = {
                        "by_status": status_counts,
                        "oldest_queue_age_seconds": float(oldest_queue) if oldest_queue is not None else None,
                        "expired_leases": int(expired),
                        "oldest_pending_review_age_seconds": (
                            float(oldest_review) if oldest_review is not None else None
                        ),
                    }

                    cur.execute(
                        """
                        SELECT payload,result,status
                        FROM research_jobs
                        WHERE run_id=%s AND job_type='DISCOVERY'
                        ORDER BY created_at,id LIMIT 1
                        """,
                        (selected,),
                    )
                    discovery = _one_dict(cur)
                    discovery_result = (discovery or {}).get("result") or {}
                    discovery_payload = (discovery or {}).get("payload") or {}
                    candidates = list(discovery_result.get("candidates") or [])
                    limit = discovery_payload.get("candidate_limit")
                    if isinstance(limit, int) and limit > 0:
                        candidates = candidates[:limit]
                    candidate_ids = {str(x.get("id")) for x in candidates if x.get("id") is not None}

                    cur.execute(
                        """
                        SELECT result,status
                        FROM research_jobs
                        WHERE run_id=%s AND job_type='SCREEN'
                        ORDER BY created_at,id LIMIT 1
                        """,
                        (selected,),
                    )
                    screen = _one_dict(cur)
                    screen_result = (screen or {}).get("result") or {}
                    retained = [
                        item
                        for item in list(screen_result.get("retained") or [])
                        if str(item.get("id")) in candidate_ids
                    ]

                    cur.execute(
                        """
                        SELECT job_type,status,count(*)::int
                        FROM research_jobs
                        WHERE run_id=%s
                        GROUP BY job_type,status
                        ORDER BY job_type,status
                        """,
                        (selected,),
                    )
                    job_counts = {}
                    for job_type, status, count in cur.fetchall():
                        job_counts.setdefault(job_type, {})[status] = int(count)

                    def completed(job_type: str) -> int:
                        return int(job_counts.get(job_type, {}).get("COMPLETED", 0))

                    cur.execute(
                        """
                        SELECT count(*)::int
                        FROM evidence_objects e
                        WHERE e.attributes->>'semantic_job_id' IN (
                          SELECT id
                          FROM research_jobs
                          WHERE run_id=%s
                            AND job_type='EXTRACT'
                        )
                        """,
                        (selected,),
                    )
                    evidence_count = int(cur.fetchone()[0])
                    cur.execute(
                        """
                        SELECT count(*)::int
                        FROM gap_candidates
                        WHERE attributes->>'research_run_id'=%s
                        """,
                        (selected,),
                    )
                    gap_count = int(cur.fetchone()[0])
                    cur.execute(
                        """
                        SELECT count(*)::int
                        FROM novelty_challenges nc
                        JOIN gap_candidates g ON g.id=nc.gap_candidate_id
                        WHERE g.attributes->>'research_run_id'=%s
                        """,
                        (selected,),
                    )
                    challenge_count = int(cur.fetchone()[0])
                    cur.execute(
                        """
                        SELECT status,count(*)::int
                        FROM human_review_requests
                        WHERE run_id=%s
                        GROUP BY status
                        ORDER BY status
                        """,
                        (selected,),
                    )
                    review_counts = dict(cur.fetchall())

                    pipeline = {
                        "discovered": len(candidates),
                        "retained": len(retained),
                        "acquired": completed("ACQUIRE"),
                        "extracted": completed("EXTRACT"),
                        "verified": completed("VERIFY"),
                        "checkpointed": completed("CHECKPOINT"),
                        "syntheses": completed("SYNTHESIZE"),
                        "evidence_objects": evidence_count,
                        "gaps": gap_count,
                        "novelty_search": job_counts.get("NOVELTY_SEARCH", {}),
                        "novelty_arbitrations": completed("NOVELTY_ARBITRATE"),
                        "novelty_challenges": challenge_count,
                        "human_reviews": review_counts,
                    }

                    cur.execute(
                        """
                        SELECT
                          count(*)::int,
                          COALESCE(sum(input_tokens),0)::bigint,
                          COALESCE(sum(cached_input_tokens),0)::bigint,
                          COALESCE(sum(output_tokens),0)::bigint,
                          COALESCE(sum(reasoning_output_tokens),0)::bigint,
                          COALESCE(sum(wall_time_ms),0)::bigint,
                          COALESCE(sum(web_search_count),0)::bigint
                        FROM research_usage_events
                        WHERE run_id=%s
                        """,
                        (selected,),
                    )
                    total = cur.fetchone()
                    usage_totals = {
                        "calls": int(total[0]),
                        "input_tokens": int(total[1]),
                        "cached_input_tokens": int(total[2]),
                        "output_tokens": int(total[3]),
                        "reasoning_output_tokens": int(total[4]),
                        "wall_time_ms": int(total[5]),
                        "web_search_count": int(total[6]),
                    }

                    cur.execute(
                        """
                        SELECT
                          COALESCE(logical_agent_id,worker_id,'unknown') AS agent,
                          count(*)::int AS calls,
                          COALESCE(sum(input_tokens),0)::bigint AS input_tokens,
                          COALESCE(sum(cached_input_tokens),0)::bigint AS cached_input_tokens,
                          COALESCE(sum(output_tokens),0)::bigint AS output_tokens,
                          COALESCE(sum(reasoning_output_tokens),0)::bigint AS reasoning_output_tokens,
                          COALESCE(sum(wall_time_ms),0)::bigint AS wall_time_ms,
                          (array_agg(model ORDER BY created_at DESC)
                            FILTER (WHERE model IS NOT NULL))[1] AS latest_model,
                          (array_agg(reasoning_effort ORDER BY created_at DESC)
                            FILTER (WHERE reasoning_effort IS NOT NULL))[1] AS latest_reasoning_effort,
                          (array_agg(provider ORDER BY created_at DESC)
                            FILTER (WHERE provider IS NOT NULL))[1] AS latest_provider
                        FROM research_usage_events
                        WHERE run_id=%s
                        GROUP BY COALESCE(logical_agent_id,worker_id,'unknown')
                        ORDER BY input_tokens DESC,agent
                        """,
                        (selected,),
                    )
                    usage_by_agent = _dict_rows(cur)

                    cur.execute(
                        """
                        SELECT
                          executor_tier,provider,model,reasoning_effort,
                          count(*)::int AS calls,
                          COALESCE(sum(input_tokens),0)::bigint AS input_tokens,
                          COALESCE(sum(cached_input_tokens),0)::bigint AS cached_input_tokens,
                          COALESCE(sum(output_tokens),0)::bigint AS output_tokens,
                          COALESCE(sum(reasoning_output_tokens),0)::bigint AS reasoning_output_tokens,
                          COALESCE(sum(wall_time_ms),0)::bigint AS wall_time_ms
                        FROM research_usage_events
                        WHERE run_id=%s
                        GROUP BY executor_tier,provider,model,reasoning_effort
                        ORDER BY input_tokens DESC,executor_tier,model
                        """,
                        (selected,),
                    )
                    usage_by_model = _dict_rows(cur)

                    cur.execute(
                        """
                        WITH latest AS (
                          SELECT DISTINCT ON (COALESCE(logical_agent_id,id))
                            id,COALESCE(logical_agent_id,id) AS logical_agent_id,
                            supported_tiers,state,last_heartbeat_at,created_at,
                            host_name,process_id,code_sha,backend,model,reasoning_effort
                          FROM research_workers
                          ORDER BY COALESCE(logical_agent_id,id),created_at DESC,id
                        )
                        SELECT
                          l.*,
                          j.id AS current_job_id,
                          j.job_type AS current_job_type,
                          a.started_at AS current_started_at
                        FROM latest l
                        LEFT JOIN research_jobs j
                          ON j.lease_owner=l.id
                         AND j.run_id=%s
                         AND j.status='RUNNING'
                        LEFT JOIN research_job_attempts a
                          ON a.job_id=j.id
                         AND a.attempt_no=j.attempt_count
                        ORDER BY l.logical_agent_id
                        """,
                        (selected,),
                    )
                    sessions = _dict_rows(cur)

                    agent_usage_map = {
                        row["agent"]: row for row in usage_by_agent
                    }
                    for row in sessions:
                        logical = row["logical_agent_id"]
                        usage = agent_usage_map.get(logical, {})
                        current_started = row.get("current_started_at")
                        agents.append(
                            {
                                "logical_agent_id": logical,
                                "worker_id": row["id"],
                                "supported_tiers": list(row["supported_tiers"]),
                                "state": row["state"],
                                "heartbeat_age_seconds": _seconds(now-row["last_heartbeat_at"]),
                                "session_age_seconds": _seconds(now-row["created_at"]),
                                "host_name": row.get("host_name"),
                                "process_id": row.get("process_id"),
                                "code_sha": row.get("code_sha"),
                                "backend": (
                                    row.get("backend")
                                    or usage.get("latest_provider")
                                ),
                                "model": (
                                    row.get("model")
                                    or usage.get("latest_model")
                                ),
                                "reasoning_effort": (
                                    row.get("reasoning_effort")
                                    or usage.get("latest_reasoning_effort")
                                ),
                                "current_job_id": row.get("current_job_id"),
                                "current_job_type": row.get("current_job_type"),
                                "current_job_elapsed_seconds": (
                                    _seconds(now-current_started)
                                    if current_started else None
                                ),
                                "calls": int(usage.get("calls") or 0),
                                "input_tokens": int(usage.get("input_tokens") or 0),
                                "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
                                "output_tokens": int(usage.get("output_tokens") or 0),
                                "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
                                "wall_time_ms": int(usage.get("wall_time_ms") or 0),
                            }
                        )

                    for hours in window_values:
                        cutoff = now - timedelta(hours=hours)
                        cur.execute(
                            """
                            SELECT j.job_type,count(*)::int
                            FROM research_job_attempts a
                            JOIN research_jobs j ON j.id=a.job_id
                            WHERE j.run_id=%s
                              AND a.status='COMPLETED'
                              AND a.finished_at >= %s
                            GROUP BY j.job_type
                            ORDER BY j.job_type
                            """,
                            (selected, cutoff),
                        )
                        completions = dict(cur.fetchall())
                        cur.execute(
                            """
                            SELECT
                              count(*)::int,
                              COALESCE(sum(input_tokens),0)::bigint,
                              COALESCE(sum(output_tokens),0)::bigint
                            FROM research_usage_events
                            WHERE run_id=%s AND created_at >= %s
                            """,
                            (selected, cutoff),
                        )
                        u = cur.fetchone()
                        cur.execute(
                            """
                            SELECT count(*)::int
                            FROM evidence_objects e
                            WHERE e.attributes->>'semantic_job_id' IN (
                              SELECT id
                              FROM research_jobs
                              WHERE run_id=%s
                                AND job_type='EXTRACT'
                            )
                              AND e.created_at >= %s
                            """,
                            (selected, cutoff),
                        )
                        ev = int(cur.fetchone()[0])
                        cur.execute(
                            """
                            SELECT count(*)::int
                            FROM gap_candidates
                            WHERE attributes->>'research_run_id'=%s
                              AND created_at >= %s
                            """,
                            (selected, cutoff),
                        )
                        gaps = int(cur.fetchone()[0])
                        key = f"{hours:g}h"
                        throughputs[key] = {
                            "window_hours": hours,
                            "completed_jobs": {k: int(v) for k, v in completions.items()},
                            "acquired_per_hour": int(completions.get("ACQUIRE", 0))/hours,
                            "extracted_per_hour": int(completions.get("EXTRACT", 0))/hours,
                            "verified_per_hour": int(completions.get("VERIFY", 0))/hours,
                            "evidence_per_hour": ev/hours,
                            "gaps_per_hour": gaps/hours,
                            "semantic_calls_per_hour": int(u[0])/hours,
                            "input_tokens_per_hour": int(u[1])/hours,
                            "output_tokens_per_hour": int(u[2])/hours,
                        }

        allowance: dict[str, Any] = {}
        for window_type in ("FIVE_HOUR", "WEEKLY"):
            rows = [
                row for row in allowance_rows
                if row["window_type"] == window_type
            ]
            allowance[window_type] = _forecast_observations(rows, now=now)

        verified = int(pipeline.get("verified") or 0)
        evidence_count = int(pipeline.get("evidence_objects") or 0)
        gap_count = int(pipeline.get("novelty_challenges") or 0)
        total_tokens = usage_totals["input_tokens"] + usage_totals["output_tokens"]
        efficiency = {
            "input_tokens_per_verified_source": (
                usage_totals["input_tokens"]/verified if verified else None
            ),
            "total_tokens_per_verified_source": (
                total_tokens/verified if verified else None
            ),
            "semantic_calls_per_verified_source": (
                usage_totals["calls"]/verified if verified else None
            ),
            "total_tokens_per_evidence_object": (
                total_tokens/evidence_count if evidence_count else None
            ),
            "total_tokens_per_challenged_gap": (
                total_tokens/gap_count if gap_count else None
            ),
            "wall_minutes_per_verified_source": (
                usage_totals["wall_time_ms"]/60000/verified if verified else None
            ),
        }

        input_total = max(1, usage_totals["input_tokens"])
        for row in usage_by_agent:
            row["input_token_share"] = row["input_tokens"]/input_total

        if run:
            run = {
                **run,
                "created_at": _iso(run["created_at"]),
                "updated_at": _iso(run["updated_at"]),
            }
        for row in backoffs:
            row["blocked_until"] = _iso(row["blocked_until"])

        systemd = _systemd_state()
        daemon_offline = (
            systemd.get("available")
            and systemd.get("ActiveState") != "active"
        )
        if daemon_offline:
            for agent in agents:
                agent["effective_state"] = "OFFLINE"
        else:
            for agent in agents:
                agent["effective_state"] = agent.get("state")

        alerts = []
        if daemon_offline:
            alerts.append(
                "daemon is inactive; persisted run is paused operationally"
            )
        if queue.get("by_status", {}).get("FAILED"):
            alerts.append(f"{queue['by_status']['FAILED']} failed job(s)")
        if queue.get("expired_leases"):
            alerts.append(f"{queue['expired_leases']} expired lease(s)")
        for row in backoffs:
            alerts.append(f"{row['provider']} backoff until {row['blocked_until']}")
        if queue.get("oldest_pending_review_age_seconds") is not None:
            alerts.append(
                "pending human review age "
                + _fmt_duration(queue["oldest_pending_review_age_seconds"])
            )

        return {
            "captured_at": now.isoformat(),
            "systemd": systemd,
            "run": run,
            "allowance": allowance,
            "agents": agents,
            "pipeline": pipeline,
            "throughput": throughputs,
            "queue": queue,
            "usage": {
                "totals": usage_totals,
                "by_agent": usage_by_agent,
                "by_model": usage_by_model,
            },
            "efficiency": efficiency,
            "provider_backoffs": backoffs,
            "alerts": alerts,
        }


def _allowance_line(label: str, data: dict[str, Any]) -> str:
    if data.get("percent_used") is None:
        return f"{label:<7} no observation"
    used = f"{data['percent_used']:.1f}% used"
    confidence = data.get("confidence") or "NO_DATA"
    burn = data.get("burn_percent_per_hour")
    eta = data.get("eta_hours")
    pieces = [f"{label:<7} {used}", confidence]
    if burn is not None:
        pieces.append(f"burn {burn:.2f}%/h")
    if eta is not None:
        pieces.append(f"ETA {_fmt_duration(eta*3600)}")
    if data.get("reset_at"):
        pieces.append(f"reset {data['reset_at']}")
    return " | ".join(pieces)


def render_hud(snapshot: dict[str, Any], *, color: bool = True) -> str:
    width = max(100, shutil.get_terminal_size((120, 40)).columns)
    hr = "─" * min(width, 140)
    run = snapshot.get("run") or {}
    systemd = snapshot.get("systemd") or {}

    daemon = (
        f"{systemd.get('ActiveState','?')}/{systemd.get('SubState','?')}"
        if systemd.get("available")
        else "db-only"
    )
    lines = [
        "RESEARCH ENGINE HUD",
        hr,
        (
            f"Daemon {daemon} | Run {run.get('id','-')} | "
            f"{run.get('status','-')} / {run.get('current_phase','-')} | "
            f"Gate {run.get('human_gate_state','-')}"
        ),
        f"Captured {snapshot.get('captured_at','-')}",
        "",
        "ALLOWANCE",
        _allowance_line("5-hour", snapshot["allowance"]["FIVE_HOUR"]),
        _allowance_line("weekly", snapshot["allowance"]["WEEKLY"]),
        "",
        "AGENTS",
        f"{'agent':24} {'state':8} {'model':16} {'job':18} {'elapsed':8} {'hb':8} {'calls':>5} {'input':>9} {'output':>9}",
    ]

    agents = snapshot.get("agents") or []
    if not agents:
        lines.append("(no worker sessions)")
    for agent in agents:
        job = agent.get("current_job_type") or "-"
        model = agent.get("model") or agent.get("backend") or "-"
        lines.append(
            f"{str(agent.get('logical_agent_id','-'))[:24]:24} "
            f"{str(agent.get('effective_state') or agent.get('state','-'))[:8]:8} "
            f"{str(model)[:16]:16} "
            f"{str(job)[:18]:18} "
            f"{_fmt_duration(agent.get('current_job_elapsed_seconds')):8} "
            f"{_fmt_duration(agent.get('heartbeat_age_seconds')):8} "
            f"{agent.get('calls',0):5d} "
            f"{_fmt_count(agent.get('input_tokens',0)):>9} "
            f"{_fmt_count(agent.get('output_tokens',0)):>9}"
        )

    p = snapshot.get("pipeline") or {}
    lines.extend(
        [
            "",
            "PIPELINE",
            (
                f"discover {p.get('discovered',0)} -> retain {p.get('retained',0)} -> "
                f"acquire {p.get('acquired',0)} -> extract {p.get('extracted',0)} -> "
                f"verify {p.get('verified',0)} | evidence {p.get('evidence_objects',0)}"
            ),
            (
                f"checkpoint {p.get('checkpointed',0)} | synth {p.get('syntheses',0)} | "
                f"gaps {p.get('gaps',0)} | novelty-search "
                f"{(p.get('novelty_search') or {}).get('COMPLETED',0)} complete / "
                f"{(p.get('novelty_search') or {}).get('QUEUED',0)} queued | "
                f"arbitrate {p.get('novelty_arbitrations',0)}"
            ),
            "",
            "THROUGHPUT",
            f"{'window':8} {'acq/h':>8} {'ext/h':>8} {'ver/h':>8} {'evid/h':>9} {'calls/h':>9} {'input/h':>10}",
        ]
    )
    for key, row in snapshot.get("throughput", {}).items():
        lines.append(
            f"{key:8} {row['acquired_per_hour']:8.2f} {row['extracted_per_hour']:8.2f} "
            f"{row['verified_per_hour']:8.2f} {row['evidence_per_hour']:9.2f} "
            f"{row['semantic_calls_per_hour']:9.2f} {_fmt_count(row['input_tokens_per_hour']):>10}"
        )

    q = snapshot.get("queue") or {}
    totals = snapshot.get("usage", {}).get("totals", {})
    eff = snapshot.get("efficiency") or {}
    lines.extend(
        [
            "",
            "USAGE / EFFICIENCY",
            (
                f"calls {totals.get('calls',0)} | input {_fmt_count(totals.get('input_tokens',0))} "
                f"(cached {_fmt_count(totals.get('cached_input_tokens',0))}) | "
                f"output {_fmt_count(totals.get('output_tokens',0))} | "
                f"reasoning {_fmt_count(totals.get('reasoning_output_tokens',0))}"
            ),
            (
                "per verified source: "
                f"input {_fmt_count(eff.get('input_tokens_per_verified_source'))} | "
                f"total {_fmt_count(eff.get('total_tokens_per_verified_source'))} | "
                f"calls {_fmt_count(eff.get('semantic_calls_per_verified_source'))}"
            ),
            "",
            "QUEUE / ALERTS",
            (
                f"status {q.get('by_status',{})} | oldest queued "
                f"{_fmt_duration(q.get('oldest_queue_age_seconds'))} | expired leases "
                f"{q.get('expired_leases',0)}"
            ),
        ]
    )
    alerts = snapshot.get("alerts") or []
    if alerts:
        lines.extend(f"! {alert}" for alert in alerts)
    else:
        lines.append("no active alerts")
    return "\n".join(lines)


def hud_loop(
    service: OpsService,
    *,
    run_id: str | None = None,
    refresh_seconds: float = 2.0,
    once: bool = False,
    color: bool = True,
) -> None:
    while True:
        snapshot = service.snapshot(run_id=run_id)
        if not once:
            print("\033[2J\033[H", end="")
        print(render_hud(snapshot, color=color), flush=True)
        if once:
            return
        time.sleep(max(0.25, refresh_seconds))


def service_from_env() -> OpsService:
    return OpsService(PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"]))
