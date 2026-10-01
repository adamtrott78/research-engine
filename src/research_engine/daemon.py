from __future__ import annotations

import os
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .orchestration import (
    DEFAULT_FANOUT_LIMITS,
    ExecutorTier,
    OrchestratorStore,
    ResearchPlanner,
)
from .postgres import PostgresStore


class BackpressureError(RuntimeError):
    pass


class BudgetPolicyError(ValueError):
    pass


SEMANTIC_TIERS = {
    ExecutorTier.CHEAP,
    ExecutorTier.STANDARD,
    ExecutorTier.STRONG,
}


def validate_budget_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    policy = dict(policy or {})
    budgets = dict(policy.get("budgets") or {})
    for key in ("max_input_tokens", "max_output_tokens", "max_semantic_jobs"):
        if key in budgets:
            value = budgets[key]
            if not isinstance(value, int) or value < 0:
                raise BudgetPolicyError(f"{key} must be a non-negative integer")
    by_tier = dict(budgets.get("max_jobs_by_tier") or {})
    allowed = {str(tier) for tier in SEMANTIC_TIERS}
    unknown = sorted(set(by_tier) - allowed)
    if unknown:
        raise BudgetPolicyError(f"unknown semantic budget tiers: {unknown}")
    for tier, value in by_tier.items():
        if not isinstance(value, int) or value < 0:
            raise BudgetPolicyError(
                f"max_jobs_by_tier[{tier}] must be a non-negative integer"
            )
    budgets["max_jobs_by_tier"] = by_tier
    policy["budgets"] = budgets

    fanout = dict(policy.get("fanout") or {})
    unknown_fanout = sorted(set(fanout) - set(DEFAULT_FANOUT_LIMITS))
    if unknown_fanout:
        raise BudgetPolicyError(f"unknown fanout limits: {unknown_fanout}")
    normalized_fanout: dict[str, int] = {}
    for key, default in DEFAULT_FANOUT_LIMITS.items():
        value = fanout.get(key, default)
        if not isinstance(value, int) or value <= 0:
            raise BudgetPolicyError(f"{key} must be a positive integer")
        normalized_fanout[key] = value
    policy["fanout"] = normalized_fanout
    return policy


@dataclass(frozen=True)
class DaemonLimits:
    max_active_runs: int = 4
    max_pending_human: int = 8
    stale_worker_seconds: int = 900


@dataclass(frozen=True)
class DaemonTick:
    recovered_leases: int
    jobs_processed: int
    reconciled_runs: int
    stale_workers_marked: int
    budget_paused_runs: int


def worker_counts_from_env() -> dict[ExecutorTier, int]:
    return {
        ExecutorTier.DETERMINISTIC: int(
            os.getenv("RESEARCH_ENGINE_DETERMINISTIC_WORKERS", "2")
        ),
        ExecutorTier.CHEAP: int(os.getenv("RESEARCH_ENGINE_CHEAP_WORKERS", "4")),
        ExecutorTier.STANDARD: int(os.getenv("RESEARCH_ENGINE_STANDARD_WORKERS", "2")),
        ExecutorTier.STRONG: int(os.getenv("RESEARCH_ENGINE_STRONG_WORKERS", "1")),
    }


class ResearchDaemon:
    """24/7 supervisor over the durable queue; never owns scientific truth."""

    def __init__(
        self,
        *,
        queue: OrchestratorStore,
        planner: ResearchPlanner,
        workers,
        poll_seconds: float = 2.0,
        limits: DaemonLimits | None = None,
    ):
        self.queue = queue
        self.planner = planner
        self.workers = tuple(workers)
        self.poll_seconds = poll_seconds
        self.limits = limits or DaemonLimits()

    def _active_run_ids(self) -> list[str]:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM research_runs WHERE status='RUNNING' ORDER BY created_at,id"
                )
                return [row[0] for row in cur.fetchall()]

    def _pending_human_count(self) -> int:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM human_review_requests WHERE status='PENDING'"
                )
                return int(cur.fetchone()[0])

    def admission_status(self) -> dict[str, Any]:
        active = len(self._active_run_ids())
        pending = self._pending_human_count()
        reasons = []
        if active >= self.limits.max_active_runs:
            reasons.append("MAX_ACTIVE_RUNS")
        if pending >= self.limits.max_pending_human:
            reasons.append("MAX_PENDING_HUMAN")
        return {
            "allowed": not reasons,
            "active_runs": active,
            "max_active_runs": self.limits.max_active_runs,
            "pending_human": pending,
            "max_pending_human": self.limits.max_pending_human,
            "reasons": reasons,
        }

    def submit_run(
        self,
        *,
        topic: str,
        run_id: str | None = None,
        policy: dict[str, Any] | None = None,
        discovery_payload: dict[str, Any] | None = None,
    ) -> str:
        policy = validate_budget_policy(policy)
        admission = self.admission_status()
        if not admission["allowed"]:
            raise BackpressureError(
                "research daemon is applying admission backpressure: "
                + ",".join(admission["reasons"])
            )
        run_id = run_id or f"research-run:{uuid.uuid4()}"
        try:
            state = self.queue.run_state(run_id)
        except ValueError:
            self.queue.create_run(
                run_id=run_id,
                topic=topic,
                policy=policy,
                discovery_payload=discovery_payload,
            )
            return run_id
        if state["topic"] != topic:
            raise ValueError(f"research run {run_id} already exists for a different topic")
        return run_id

    def budget_status(self, run_id: str) -> dict[str, Any]:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status,policy FROM research_runs WHERE id=%s",
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"run not found: {run_id}")
                status, policy = row
                policy = validate_budget_policy(policy)
                cur.execute(
                    """
                    SELECT COALESCE(sum(input_tokens),0)::bigint,
                           COALESCE(sum(output_tokens),0)::bigint,
                           count(*)::int
                    FROM research_usage_events
                    WHERE run_id=%s
                    """,
                    (run_id,),
                )
                input_tokens, output_tokens, semantic_jobs = cur.fetchone()
                cur.execute(
                    """
                    SELECT executor_tier,count(*)::int
                    FROM research_usage_events
                    WHERE run_id=%s
                    GROUP BY executor_tier
                    ORDER BY executor_tier
                    """,
                    (run_id,),
                )
                by_tier = dict(cur.fetchall())
                cur.execute(
                    """
                    SELECT executor_tier,count(*)::int
                    FROM research_jobs
                    WHERE run_id=%s
                      AND status IN ('QUEUED','RUNNING')
                      AND executor_tier IN ('CHEAP','STANDARD','STRONG')
                    GROUP BY executor_tier
                    ORDER BY executor_tier
                    """,
                    (run_id,),
                )
                pending_by_tier = dict(cur.fetchall())

        budgets = policy["budgets"]
        saturated_reasons: list[str] = []
        reasons: list[str] = []
        pending_semantic_jobs = sum(pending_by_tier.values())
        max_input = budgets.get("max_input_tokens")
        max_output = budgets.get("max_output_tokens")
        max_jobs = budgets.get("max_semantic_jobs")
        if max_input is not None and input_tokens >= max_input:
            saturated_reasons.append("MAX_INPUT_TOKENS")
            if pending_semantic_jobs:
                reasons.append("MAX_INPUT_TOKENS")
        if max_output is not None and output_tokens >= max_output:
            saturated_reasons.append("MAX_OUTPUT_TOKENS")
            if pending_semantic_jobs:
                reasons.append("MAX_OUTPUT_TOKENS")
        if max_jobs is not None and semantic_jobs >= max_jobs:
            saturated_reasons.append("MAX_SEMANTIC_JOBS")
            if pending_semantic_jobs:
                reasons.append("MAX_SEMANTIC_JOBS")
        for tier, limit in budgets.get("max_jobs_by_tier", {}).items():
            if by_tier.get(tier, 0) >= limit:
                reason = f"MAX_{tier}_JOBS"
                saturated_reasons.append(reason)
                if pending_by_tier.get(tier, 0):
                    reasons.append(reason)

        return {
            "run_id": run_id,
            "status": status,
            "budgets": budgets,
            "usage": {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "semantic_jobs": int(semantic_jobs),
                "jobs_by_tier": by_tier,
                "pending_jobs_by_tier": pending_by_tier,
            },
            "exhausted": bool(reasons),
            "reasons": reasons,
            "saturated_reasons": saturated_reasons,
        }

    def _enforce_budgets(self) -> list[str]:
        paused: list[str] = []
        for run_id in self._active_run_ids():
            status = self.budget_status(run_id)
            if not status["exhausted"]:
                continue
            with self.queue.store.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE research_runs
                        SET status='PAUSED_BUDGET',
                            human_gate_state='BUDGET_REVIEW',
                            updated_at=now()
                        WHERE id=%s AND status='RUNNING'
                        """,
                        (run_id,),
                    )
                    if cur.rowcount:
                        paused.append(run_id)
                conn.commit()
        return paused

    def resume_budget(
        self,
        run_id: str,
        *,
        budgets: dict[str, Any],
    ) -> dict[str, Any]:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status,policy FROM research_runs WHERE id=%s FOR UPDATE",
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"run not found: {run_id}")
                status, policy = row
                if status != "PAUSED_BUDGET":
                    raise ValueError(
                        f"run {run_id} is not paused for budget review"
                    )
                merged = dict(policy or {})
                current = dict(merged.get("budgets") or {})
                current.update(dict(budgets or {}))
                merged["budgets"] = current
                merged = validate_budget_policy(merged)
                from psycopg.types.json import Jsonb
                cur.execute(
                    """
                    UPDATE research_runs
                    SET policy=%s,status='RUNNING',
                        human_gate_state='NOT_REQUIRED',updated_at=now()
                    WHERE id=%s
                    """,
                    (Jsonb(merged), run_id),
                )
            conn.commit()
        return self.budget_status(run_id)

    def _mark_stale_workers(self) -> int:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_workers w
                    SET state='STALE'
                    WHERE w.state='ACTIVE'
                      AND w.last_heartbeat_at < now() - make_interval(secs => %s)
                      AND NOT EXISTS (
                        SELECT 1 FROM research_jobs j
                        WHERE j.lease_owner=w.id
                          AND j.status='RUNNING'
                          AND j.lease_expires_at > now()
                      )
                    """,
                    (self.limits.stale_worker_seconds,),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def tick(self) -> DaemonTick:
        recovered = self.queue.recover_expired_leases()
        initial_runs = self._active_run_ids()
        for run_id in initial_runs:
            self.planner.reconcile(run_id)

        budget_paused = set(self._enforce_budgets())
        run_ids = self._active_run_ids()

        processed = 0
        if self.workers:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
                jobs = list(pool.map(lambda worker: worker.run_once(), self.workers))
            processed = sum(job is not None for job in jobs)

        current_runs = self._active_run_ids()
        for run_id in current_runs:
            self.planner.reconcile(run_id)
        budget_paused.update(self._enforce_budgets())

        stale = self._mark_stale_workers()
        return DaemonTick(
            recovered_leases=len(recovered),
            jobs_processed=processed,
            reconciled_runs=len(
                set(initial_runs) | set(run_ids) | set(current_runs)
            ),
            stale_workers_marked=stale,
            budget_paused_runs=len(budget_paused),
        )

    def run_forever(self, *, max_ticks: int | None = None) -> dict[str, Any]:
        ticks = 0
        totals = {
            "recovered_leases": 0,
            "jobs_processed": 0,
            "stale_workers_marked": 0,
            "budget_paused_runs": 0,
        }
        while True:
            tick = self.tick()
            ticks += 1
            totals["recovered_leases"] += tick.recovered_leases
            totals["jobs_processed"] += tick.jobs_processed
            totals["stale_workers_marked"] += tick.stale_workers_marked
            totals["budget_paused_runs"] += tick.budget_paused_runs

            if max_ticks is not None and ticks >= max_ticks:
                return {
                    "ticks": ticks,
                    **totals,
                    "health": self.health_snapshot(),
                }
            if self.poll_seconds > 0:
                time.sleep(self.poll_seconds)

    def health_snapshot(self) -> dict[str, Any]:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status,count(*)::int FROM research_runs GROUP BY status ORDER BY status"
                )
                run_counts = dict(cur.fetchall())
                cur.execute(
                    """
                    SELECT executor_tier,status,count(*)::int
                    FROM research_jobs
                    GROUP BY executor_tier,status
                    ORDER BY executor_tier,status
                    """
                )
                job_counts = [
                    {"executor_tier": row[0], "status": row[1], "count": row[2]}
                    for row in cur.fetchall()
                ]
                cur.execute(
                    """
                    SELECT id,supported_tiers,state,last_heartbeat_at
                    FROM research_workers
                    ORDER BY id
                    """
                )
                workers = [
                    {
                        "id": row[0],
                        "supported_tiers": list(row[1]),
                        "state": row[2],
                        "last_heartbeat_at": row[3].isoformat(),
                    }
                    for row in cur.fetchall()
                ]
                cur.execute(
                    "SELECT count(*)::int FROM human_review_requests WHERE status='PENDING'"
                )
                pending_human = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT id,topic,policy
                    FROM research_runs
                    WHERE status='PAUSED_BUDGET'
                    ORDER BY updated_at,id
                    """
                )
                budget_paused = [
                    {"id": row[0], "topic": row[1], "policy": row[2]}
                    for row in cur.fetchall()
                ]
                cur.execute(
                    """
                    SELECT count(*)::int
                    FROM research_jobs
                    WHERE status='RUNNING' AND lease_expires_at <= now()
                    """
                )
                expired_leases = cur.fetchone()[0]
                cur.execute(
                    """
                    SELECT provider,blocked_until,reason,detail
                    FROM research_provider_backoffs
                    WHERE blocked_until > now()
                    ORDER BY provider
                    """
                )
                provider_backoffs = [
                    {
                        "provider": row[0],
                        "blocked_until": row[1].isoformat(),
                        "reason": row[2],
                        "detail": row[3],
                    }
                    for row in cur.fetchall()
                ]
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "runs": run_counts,
            "jobs": job_counts,
            "workers": workers,
            "pending_human": pending_human,
            "budget_paused": budget_paused,
            "expired_leases": expired_leases,
            "provider_backoffs": provider_backoffs,
            "admission": self.admission_status(),
        }

    @staticmethod
    def _usage_from_result(result: Any) -> tuple[int, int]:
        if not isinstance(result, dict):
            return 0, 0
        execution = result.get("_execution")
        if not isinstance(execution, dict):
            return 0, 0
        usage = execution.get("usage")
        if not isinstance(usage, dict):
            return 0, 0
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    def overnight_report(self, *, hours: float = 12.0) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,topic,status,current_phase,knowledge_checkpoint_id,
                           human_gate_state,created_at,updated_at
                    FROM research_runs
                    WHERE created_at >= %s OR updated_at >= %s
                    ORDER BY updated_at,id
                    """,
                    (cutoff, cutoff),
                )
                runs = [
                    {
                        "id": row[0],
                        "topic": row[1],
                        "status": row[2],
                        "current_phase": row[3],
                        "knowledge_checkpoint_id": row[4],
                        "human_gate_state": row[5],
                        "created_at": row[6].isoformat(),
                        "updated_at": row[7].isoformat(),
                    }
                    for row in cur.fetchall()
                ]

                cur.execute(
                    """
                    SELECT job_type,executor_tier,status,result,last_error
                    FROM research_jobs
                    WHERE updated_at >= %s
                    ORDER BY updated_at,id
                    """,
                    (cutoff,),
                )
                jobs = cur.fetchall()
                cur.execute(
                    """
                    SELECT executor_tier,provider,model,input_tokens,output_tokens
                    FROM research_usage_events
                    WHERE created_at >= %s
                    ORDER BY created_at,id
                    """,
                    (cutoff,),
                )
                usage_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT id,run_id,status,payload,decision,created_at,decided_at
                    FROM human_review_requests
                    WHERE created_at >= %s OR status='PENDING'
                    ORDER BY created_at,id
                    """,
                    (cutoff,),
                )
                reviews = [
                    {
                        "id": row[0],
                        "run_id": row[1],
                        "status": row[2],
                        "payload": row[3],
                        "decision": row[4],
                        "created_at": row[5].isoformat(),
                        "decided_at": row[6].isoformat() if row[6] else None,
                    }
                    for row in cur.fetchall()
                ]

        by_type: dict[str, dict[str, int]] = {}
        failures = []
        for job_type, tier, status, result, last_error in jobs:
            key = f"{job_type}:{tier}"
            by_type.setdefault(key, {})
            by_type[key][status] = by_type[key].get(status, 0) + 1
            if status == "FAILED":
                failures.append(
                    {
                        "job_type": job_type,
                        "executor_tier": tier,
                        "error": last_error,
                    }
                )

        usage_by_tier: dict[str, dict[str, int]] = {}
        input_tokens = 0
        output_tokens = 0
        for tier, provider, model, inp, out in usage_rows:
            input_tokens += int(inp)
            output_tokens += int(out)
            key = f"{tier}:{provider}:{model or 'unknown'}"
            bucket = usage_by_tier.setdefault(
                key, {"jobs": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["jobs"] += 1
            bucket["input_tokens"] += int(inp)
            bucket["output_tokens"] += int(out)

        return {
            "window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs": runs,
            "job_summary": by_type,
            "model_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "by_tier_model": usage_by_tier,
            },
            "failures": failures,
            "human_reviews": reviews,
            "health": self.health_snapshot(),
        }


def daemon_from_env(
    *,
    poll_seconds: float = 2.0,
    lease_seconds: int = 300,
    limits: DaemonLimits | None = None,
    with_workers: bool = True,
) -> ResearchDaemon:
    from .worker_runtime import deterministic_worker_from_env, semantic_worker_from_env

    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    if limits is None:
        limits = DaemonLimits(
            max_active_runs=int(
                os.getenv("RESEARCH_ENGINE_MAX_ACTIVE_RUNS", "4")
            ),
            max_pending_human=int(
                os.getenv("RESEARCH_ENGINE_MAX_PENDING_HUMAN", "8")
            ),
            stale_worker_seconds=int(
                os.getenv("RESEARCH_ENGINE_STALE_WORKER_SECONDS", "900")
            ),
        )
    store = PostgresStore(dsn)
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)
    workers = []
    if with_workers:
        prefix = f"daemon:{socket.gethostname()}:{os.getpid()}"
        counts = worker_counts_from_env()
        for index in range(counts[ExecutorTier.DETERMINISTIC]):
            workers.append(
                deterministic_worker_from_env(
                    worker_id=f"{prefix}:deterministic:{index}",
                    lease_seconds=lease_seconds,
                )
            )
        for tier in (ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG):
            for index in range(counts[tier]):
                workers.append(
                    semantic_worker_from_env(
                        tiers=[tier],
                        worker_id=f"{prefix}:{str(tier).lower()}:{index}",
                        lease_seconds=lease_seconds,
                    )
                )

    return ResearchDaemon(
        queue=queue,
        planner=planner,
        workers=workers,
        poll_seconds=poll_seconds,
        limits=limits,
    )
