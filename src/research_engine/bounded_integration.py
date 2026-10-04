from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from psycopg import sql

from .postgres import PostgresStore


SCIENTIFIC_TABLES = (
    "works",
    "work_versions",
    "external_identifiers",
    "source_artifacts",
    "discovery_runs",
    "acquisition_attempts",
    "extraction_runs",
    "source_locators",
    "evidence_objects",
    "evidence_relations",
    "verification_events",
    "knowledge_checkpoints",
    "synthesis_artifacts",
    "inferences",
    "gap_candidates",
    "novelty_challenges",
    "research_programs",
    "funding_opportunities",
    "solicitation_requirements",
    "proposal_compilations",
    "execution_bridge_plans",
    "research_runs",
    "research_jobs",
    "research_job_attempts",
    "research_usage_events",
    "research_allowance_observations",
    "research_calibration_events",
    "semantic_materializations",
    "human_review_requests",
    "embeddings",
)


class BoundedProductionIntegrationInspector:
    def __init__(self, store: PostgresStore):
        self.store = store

    def quiescence(self) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*)::int FROM research_runs WHERE status='RUNNING'"
                )
                active_runs = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT count(*)::int
                    FROM research_jobs
                    WHERE status IN ('QUEUED','RUNNING')
                      AND executor_tier <> 'HUMAN'
                    """
                )
                pending_nonhuman = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT count(*)::int
                    FROM human_review_requests
                    WHERE status='PENDING'
                    """
                )
                pending_human = int(cur.fetchone()[0])
        return {
            "active_running_runs": active_runs,
            "queued_or_running_nonhuman_jobs": pending_nonhuman,
            "pending_human_reviews": pending_human,
            "safe_for_idle_daemon_smoke": (
                active_runs == 0 and pending_nonhuman == 0
            ),
        }

    def scientific_fingerprint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                for table in SCIENTIFIC_TABLES:
                    cur.execute(
                        sql.SQL(
                            "SELECT row_to_json(t)::text FROM {} t "
                            "ORDER BY row_to_json(t)::text"
                        ).format(sql.Identifier(table))
                    )
                    rows = [str(row[0]) for row in cur.fetchall()]
                    payload[table] = {
                        "rows": len(rows),
                        "sha256": hashlib.sha256(
                            "\n".join(rows).encode("utf-8")
                        ).hexdigest(),
                    }
        return {
            "schema_version": 1,
            "tables": payload,
        }

    def workers_for_process(self, process_id: int) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,supported_tiers,state,logical_agent_id,process_id,
                           backend,model,reasoning_effort,last_heartbeat_at
                    FROM research_workers
                    WHERE process_id=%s
                    ORDER BY logical_agent_id,id
                    """,
                    (int(process_id),),
                )
                workers = [
                    {
                        "id": row[0],
                        "supported_tiers": list(row[1]),
                        "state": row[2],
                        "logical_agent_id": row[3],
                        "process_id": row[4],
                        "backend": row[5],
                        "model": row[6],
                        "reasoning_effort": row[7],
                        "last_heartbeat_at": row[8].isoformat(),
                    }
                    for row in cur.fetchall()
                ]
        expected = {
            "deterministic:0": ["DETERMINISTIC"],
            "semantic:cheap:0": ["CHEAP"],
            "semantic:standard:0": ["STANDARD"],
            "semantic:strong:0": ["STRONG"],
        }
        actual = {
            row["logical_agent_id"]: row["supported_tiers"]
            for row in workers
            if row["state"] == "ACTIVE"
        }
        return {
            "daemon_pid": int(process_id),
            "workers": workers,
            "expected_active_workers": expected,
            "actual_active_workers": actual,
            "worker_registration_pass": actual == expected,
            "quiescence": self.quiescence(),
        }

    def sidecar_telemetry(self, source_artifact_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status,compiler_version,formula_pages,
                           fast_empty_tables,accurate_pages,paddle_fallbacks,
                           wall_time_ms,routing_summary,
                           local_inference_summary,created_at
                    FROM document_sidecar_events
                    WHERE source_artifact_id=%s
                    ORDER BY id DESC
                    LIMIT 2
                    """,
                    (source_artifact_id,),
                )
                rows = [
                    {
                        "status": row[0],
                        "compiler_version": row[1],
                        "formula_pages": row[2],
                        "fast_empty_tables": row[3],
                        "accurate_pages": row[4],
                        "paddle_fallbacks": row[5],
                        "wall_time_ms": row[6],
                        "routing_summary": row[7],
                        "local_inference_summary": row[8],
                        "created_at": row[9].isoformat(),
                    }
                    for row in cur.fetchall()
                ]
        return {
            "artifact_id": source_artifact_id,
            "latest_events": rows,
            "telemetry_pass": [row["status"] for row in rows]
            == ["CACHE_HIT", "COMPLETED"],
        }


def validate_sidecar_smoke(smoke: dict[str, Any]) -> dict[str, Any]:
    first = dict(smoke.get("first") or {})
    second = dict(smoke.get("second") or {})
    return {
        "artifact_id": smoke.get("artifact_id"),
        "cache_miss_then_hit": bool(smoke.get("cache_miss_then_hit")),
        "first_status": first.get("status"),
        "second_status": second.get("status"),
        "authority": first.get("authority"),
        "semantic_baseline": first.get("semantic_baseline"),
        "routing_summary": first.get("routing_summary") or {},
        "broker_final_health": smoke.get("broker_final_health") or {},
        "pass": bool(smoke.get("cache_miss_then_hit"))
        and first.get("authority") == "DERIVED_SIDECAR_ONLY"
        and first.get("semantic_baseline") == "AUTHORITATIVE_PDF_HYDRATOR",
    }


def compare_fingerprints(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_tables = dict(before.get("tables") or {})
    after_tables = dict(after.get("tables") or {})
    changed = sorted(
        table
        for table in set(before_tables) | set(after_tables)
        if before_tables.get(table) != after_tables.get(table)
    )
    return {
        "unchanged": not changed,
        "changed_tables": changed,
        "tables_compared": len(set(before_tables) | set(after_tables)),
    }


def build_integration_report(
    *,
    research_engine_head: str,
    preflight: dict[str, Any],
    sidecar: dict[str, Any],
    workers: dict[str, Any],
    fingerprint_comparison: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    decision = (
        "PASS"
        if preflight.get("safe_for_idle_daemon_smoke")
        and sidecar.get("pass")
        and workers.get("worker_registration_pass")
        and workers.get("quiescence", {}).get("safe_for_idle_daemon_smoke")
        and fingerprint_comparison.get("unchanged")
        and telemetry.get("telemetry_pass")
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "decision": decision,
        "research_engine_head": research_engine_head,
        "preflight": preflight,
        "sidecar": sidecar,
        "daemon": {
            "daemon_pid": workers.get("daemon_pid"),
            "workers": workers.get("workers") or [],
            "worker_registration_pass": workers.get(
                "worker_registration_pass"
            ),
            "nonhuman_work_processed": False,
            "final_required_state": "inactive/dead PID 0",
        },
        "scientific_fingerprint": fingerprint_comparison,
        "sidecar_telemetry": telemetry,
        "semantic_calls_performed": 0,
        "scientific_state_materialized": False,
        "semantic_baseline": "AUTHORITATIVE_PDF_HYDRATOR",
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
