from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Protocol


class ExecutorTier(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    CHEAP = "CHEAP"
    STANDARD = "STANDARD"
    STRONG = "STRONG"
    HUMAN = "HUMAN"


class JobType(StrEnum):
    DISCOVERY = "DISCOVERY"
    SCREEN = "SCREEN"
    ACQUIRE = "ACQUIRE"
    EXTRACT = "EXTRACT"
    VERIFY = "VERIFY"
    CHECKPOINT = "CHECKPOINT"
    SYNTHESIZE = "SYNTHESIZE"
    GAP_GENERATE = "GAP_GENERATE"
    NOVELTY_SEARCH = "NOVELTY_SEARCH"
    NOVELTY_ARBITRATE = "NOVELTY_ARBITRATE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


JOB_TIERS: dict[JobType, ExecutorTier] = {
    JobType.DISCOVERY: ExecutorTier.CHEAP,
    JobType.SCREEN: ExecutorTier.CHEAP,
    JobType.ACQUIRE: ExecutorTier.DETERMINISTIC,
    JobType.EXTRACT: ExecutorTier.STANDARD,
    JobType.VERIFY: ExecutorTier.STANDARD,
    JobType.CHECKPOINT: ExecutorTier.DETERMINISTIC,
    JobType.SYNTHESIZE: ExecutorTier.STRONG,
    JobType.GAP_GENERATE: ExecutorTier.STRONG,
    JobType.NOVELTY_SEARCH: ExecutorTier.CHEAP,
    JobType.NOVELTY_ARBITRATE: ExecutorTier.STRONG,
    JobType.HUMAN_REVIEW: ExecutorTier.HUMAN,
}


DEFAULT_FANOUT_LIMITS: dict[str, int] = {
    "max_gap_candidates": 2,
    "max_novelty_queries_per_gap": 2,
}


def fanout_limits_from_policy(policy: dict[str, Any] | None) -> dict[str, int]:
    configured = dict((policy or {}).get("fanout") or {})
    return {
        key: int(configured.get(key, default))
        for key, default in DEFAULT_FANOUT_LIMITS.items()
    }


def bounded_gap_candidates(
    result: dict[str, Any] | None,
    policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    limits = fanout_limits_from_policy(policy)
    bounded: list[dict[str, Any]] = []
    for raw_gap in list((result or {}).get("gaps") or [])[
        : limits["max_gap_candidates"]
    ]:
        gap = dict(raw_gap)
        gap["search_queries"] = list(gap.get("search_queries") or [])[
            : limits["max_novelty_queries_per_gap"]
        ]
        bounded.append(gap)
    return bounded


def bounded_discovery_candidates(job: "ResearchJob") -> list[dict[str, Any]]:
    candidates = list((job.result or {}).get("candidates") or [])
    limit = job.payload.get("candidate_limit")
    if isinstance(limit, int) and limit > 0:
        candidates = candidates[:limit]
    return [dict(item) for item in candidates]


def canonical_screened_candidates(
    discovery_job: "ResearchJob",
    screen_job: "ResearchJob",
) -> list[dict[str, Any]]:
    discovered = bounded_discovery_candidates(discovery_job)
    retained_ids = {
        str(item.get("id"))
        for item in list((screen_job.result or {}).get("retained") or [])
        if item.get("id") is not None
    }
    return [
        item
        for item in discovered
        if str(item.get("id")) in retained_ids
    ]


def acquisition_candidate_supported(item: dict[str, Any]) -> bool:
    # Import lazily to keep the orchestration module independent at import time.
    from .deterministic_executor import arxiv_id_from_uri

    return arxiv_id_from_uri(str(item.get("source_uri") or "")) is not None


def canonical_acquisition_candidate(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    from .deterministic_executor import arxiv_id_from_uri

    arxiv_id = arxiv_id_from_uri(str(item.get("source_uri") or ""))
    if arxiv_id is None:
        return None

    base_id = re.sub(r"v\\d+$", "", arxiv_id)
    canonical = dict(item)
    canonical["discovery_id"] = str(item.get("id") or "")
    canonical["id"] = f"arxiv:{base_id}"
    return canonical


def canonical_acquisition_candidates(
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        normalized = canonical_acquisition_candidate(item)
        if normalized is None:
            continue
        work_id = str(normalized["id"])
        if work_id in seen:
            continue
        seen.add(work_id)
        canonical.append(normalized)
    return canonical


@dataclass(frozen=True)
class ResearchJob:
    id: str
    run_id: str
    job_type: JobType
    executor_tier: ExecutorTier
    status: str
    priority: int
    idempotency_key: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    depends_on: list[str]
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: Any | None


@dataclass(frozen=True)
class ExecutorProfile:
    tier: ExecutorTier
    provider: str
    model_alias: str | None = None
    reasoning_class: str | None = None
    max_parallelism: int | None = None


DEFAULT_EXECUTOR_PROFILES = {
    ExecutorTier.DETERMINISTIC: ExecutorProfile(
        ExecutorTier.DETERMINISTIC, "local-code", reasoning_class="none"
    ),
    ExecutorTier.CHEAP: ExecutorProfile(
        ExecutorTier.CHEAP, "model", model_alias="cheap", reasoning_class="low"
    ),
    ExecutorTier.STANDARD: ExecutorProfile(
        ExecutorTier.STANDARD, "model", model_alias="standard", reasoning_class="medium"
    ),
    ExecutorTier.STRONG: ExecutorProfile(
        ExecutorTier.STRONG, "model", model_alias="strong", reasoning_class="high"
    ),
    ExecutorTier.HUMAN: ExecutorProfile(
        ExecutorTier.HUMAN, "human", reasoning_class="human"
    ),
}


class JobExecutor(Protocol):
    def execute(self, job: ResearchJob) -> dict[str, Any]: ...


class JobExecutionError(RuntimeError):
    """Executor failure carrying retry semantics for the durable queue."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        detail: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail or {}


def _job_id(run_id: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{idempotency_key}".encode()).hexdigest()[:24]
    return f"job:{digest}"


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


class OrchestratorStore:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _row_to_job(row) -> ResearchJob:
        return ResearchJob(
            id=row[0],
            run_id=row[1],
            job_type=JobType(row[2]),
            executor_tier=ExecutorTier(row[3]),
            status=row[4],
            priority=row[5],
            idempotency_key=row[6],
            payload=row[7],
            result=row[8],
            depends_on=list(row[9]),
            attempt_count=row[10],
            max_attempts=row[11],
            lease_owner=row[12],
            lease_expires_at=row[13],
        )

    def create_run(
        self,
        *,
        run_id: str,
        topic: str,
        policy: dict[str, Any] | None = None,
        discovery_payload: dict[str, Any] | None = None,
    ) -> str:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_runs(id, topic, status, current_phase, policy)
                    VALUES (%s,%s,'RUNNING','DISCOVERY',%s)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (run_id, topic, Jsonb(policy or {})),
                )
            conn.commit()
        initial_discovery = {"topic": topic}
        initial_discovery.update(dict(discovery_payload or {}))
        self.enqueue(
            run_id=run_id,
            job_type=JobType.DISCOVERY,
            idempotency_key="stage:discovery",
            payload=initial_discovery,
            priority=100,
        )
        return run_id

    def register_worker(
        self,
        worker_id: str,
        supported_tiers: Iterable[ExecutorTier],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from psycopg.types.json import Jsonb

        tiers = [str(x) for x in supported_tiers]
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_workers(id, supported_tiers, metadata, state, last_heartbeat_at)
                    VALUES (%s,%s,%s,'ACTIVE',now())
                    ON CONFLICT(id) DO UPDATE SET
                      supported_tiers = EXCLUDED.supported_tiers,
                      metadata = EXCLUDED.metadata,
                      state = 'ACTIVE',
                      last_heartbeat_at = now()
                    """,
                    (worker_id, tiers, Jsonb(metadata or {})),
                )
            conn.commit()

    def worker_heartbeat(self, worker_id: str) -> None:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE research_workers SET last_heartbeat_at=now() WHERE id=%s",
                    (worker_id,),
                )
                if cur.rowcount != 1:
                    raise ValueError(f"worker not registered: {worker_id}")
            conn.commit()

    def verified_work_ids(self, work_ids: Iterable[str]) -> set[str]:
        selected = sorted({str(work_id) for work_id in work_ids if work_id})
        if not selected:
            return set()
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT work_id
                    FROM evidence_objects
                    WHERE work_id = ANY(%s)
                      AND verification_state='MACHINE_CHECKED'
                    """,
                    (selected,),
                )
                return {str(row[0]) for row in cur.fetchall()}

    def provider_backoff(self, provider: str) -> dict[str, Any] | None:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT provider, blocked_until, reason, detail
                    FROM research_provider_backoffs
                    WHERE provider=%s AND blocked_until > now()
                    """,
                    (provider,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "provider": row[0],
            "blocked_until": row[1],
            "reason": row[2],
            "detail": row[3],
        }

    def defer_for_provider_limit(
        self,
        job: ResearchJob,
        worker_id: str,
        error: dict[str, Any],
        *,
        provider: str,
        retry_after_seconds: int,
    ) -> None:
        from psycopg.types.json import Jsonb

        delay = max(60, int(retry_after_seconds))
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET status='QUEUED', last_error=%s, lease_owner=NULL,
                        lease_expires_at=NULL,
                        available_at=now() + make_interval(secs => %s),
                        updated_at=now()
                    WHERE id=%s AND status='RUNNING' AND lease_owner=%s
                    """,
                    (Jsonb(error), delay, job.id, worker_id),
                )
                if cur.rowcount != 1:
                    raise ValueError(
                        f"worker {worker_id} cannot defer job {job.id}"
                    )
                cur.execute(
                    """
                    UPDATE research_job_attempts
                    SET status='DEFERRED_PLAN_LIMIT', finished_at=now(), detail=%s
                    WHERE job_id=%s AND attempt_no=%s
                    """,
                    (Jsonb(error), job.id, job.attempt_count),
                )
                cur.execute(
                    """
                    INSERT INTO research_provider_backoffs(
                      provider, blocked_until, reason, detail, updated_at
                    )
                    VALUES (
                      %s, now() + make_interval(secs => %s),
                      'CODEX_PLAN_LIMIT', %s, now()
                    )
                    ON CONFLICT(provider) DO UPDATE SET
                      blocked_until=GREATEST(
                        research_provider_backoffs.blocked_until,
                        EXCLUDED.blocked_until
                      ),
                      reason=EXCLUDED.reason,
                      detail=EXCLUDED.detail,
                      updated_at=now()
                    """,
                    (provider, delay, Jsonb(error)),
                )
            conn.commit()

    def enqueue(
        self,
        *,
        run_id: str,
        job_type: JobType,
        idempotency_key: str,
        payload: dict[str, Any],
        priority: int = 0,
        depends_on: list[str] | None = None,
        max_attempts: int = 3,
        status: str = "QUEUED",
    ) -> str:
        from psycopg.types.json import Jsonb

        if job_type == JobType.HUMAN_REVIEW:
            status = "WAITING_HUMAN"
        jid = _job_id(run_id, idempotency_key)
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_jobs(
                      id, run_id, job_type, executor_tier, status, priority,
                      idempotency_key, payload, depends_on, max_attempts
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(run_id, idempotency_key) DO NOTHING
                    """,
                    (
                        jid,
                        run_id,
                        str(job_type),
                        str(JOB_TIERS[job_type]),
                        status,
                        priority,
                        idempotency_key,
                        Jsonb(payload),
                        depends_on or [],
                        max_attempts,
                    ),
                )
                cur.execute(
                    """
                    SELECT job_type, executor_tier, payload, depends_on, max_attempts
                    FROM research_jobs WHERE id=%s
                    """,
                    (jid,),
                )
                existing = cur.fetchone()
                expected = (
                    str(job_type),
                    str(JOB_TIERS[job_type]),
                    payload,
                    depends_on or [],
                    max_attempts,
                )
                if existing is None:
                    raise RuntimeError(f"failed to create job {jid}")
                if tuple(existing) != expected:
                    raise ValueError(
                        f"idempotency key {idempotency_key} already maps to different job content"
                    )
            conn.commit()
        return jid

    def lease_next(
        self,
        *,
        worker_id: str,
        supported_tiers: Iterable[ExecutorTier],
        lease_seconds: int = 120,
    ) -> ResearchJob | None:
        tiers = [str(x) for x in supported_tiers if x != ExecutorTier.HUMAN]
        if not tiers:
            return None
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                # Serialize only lease admission so concurrent workers observe
                # each other's RUNNING reservations before evaluating budgets.
                # The lock is transaction-scoped and released immediately after
                # the leased job/attempt record commits; model execution is not
                # serialized.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("research-engine:lease-budget",),
                )
                cur.execute(
                    """
                    WITH candidate AS (
                      SELECT j.id
                      FROM research_jobs j
                      JOIN research_runs r ON r.id=j.run_id
                      WHERE j.status='QUEUED'
                        AND r.status='RUNNING'
                        AND j.available_at <= now()
                        AND j.executor_tier = ANY(%s)
                        AND (
                          j.executor_tier='DETERMINISTIC'
                          OR (
                            (
                              NULLIF(
                                r.policy->'budgets'->>'max_input_tokens',
                                ''
                              ) IS NULL
                              OR COALESCE((
                                SELECT sum(u.input_tokens)
                                FROM research_usage_events u
                                WHERE u.run_id=r.id
                              ),0) < (
                                r.policy->'budgets'->>'max_input_tokens'
                              )::bigint
                            )
                            AND (
                              NULLIF(
                                r.policy->'budgets'->>'max_output_tokens',
                                ''
                              ) IS NULL
                              OR COALESCE((
                                SELECT sum(u.output_tokens)
                                FROM research_usage_events u
                                WHERE u.run_id=r.id
                              ),0) < (
                                r.policy->'budgets'->>'max_output_tokens'
                              )::bigint
                            )
                            AND (
                              NULLIF(
                                r.policy->'budgets'->>'max_semantic_jobs',
                                ''
                              ) IS NULL
                              OR (
                                COALESCE((
                                  SELECT count(*)
                                  FROM research_usage_events u
                                  WHERE u.run_id=r.id
                                ),0)
                                + COALESCE((
                                  SELECT count(*)
                                  FROM research_jobs running
                                  WHERE running.run_id=r.id
                                    AND running.status='RUNNING'
                                    AND running.executor_tier IN (
                                      'CHEAP','STANDARD','STRONG'
                                    )
                                ),0)
                              ) < (
                                r.policy->'budgets'->>'max_semantic_jobs'
                              )::bigint
                            )
                            AND (
                              (
                                r.policy->'budgets'->'max_jobs_by_tier'
                                  ->>j.executor_tier
                              ) IS NULL
                              OR (
                                COALESCE((
                                  SELECT count(*)
                                  FROM research_usage_events u
                                  WHERE u.run_id=r.id
                                    AND u.executor_tier=j.executor_tier
                                ),0)
                                + COALESCE((
                                  SELECT count(*)
                                  FROM research_jobs running
                                  WHERE running.run_id=r.id
                                    AND running.status='RUNNING'
                                    AND running.executor_tier=j.executor_tier
                                ),0)
                              ) < (
                                r.policy->'budgets'->'max_jobs_by_tier'
                                  ->>j.executor_tier
                              )::bigint
                            )
                          )
                        )
                        AND NOT EXISTS (
                          SELECT 1
                          FROM unnest(j.depends_on) dep(job_id)
                          LEFT JOIN research_jobs d ON d.id=dep.job_id
                          WHERE d.id IS NULL OR d.status <> 'COMPLETED'
                        )
                      ORDER BY j.priority DESC, j.created_at, j.id
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                    )
                    UPDATE research_jobs j
                    SET status='RUNNING',
                        lease_owner=%s,
                        lease_expires_at=now() + make_interval(secs => %s),
                        attempt_count=j.attempt_count+1,
                        updated_at=now()
                    FROM candidate c
                    WHERE j.id=c.id
                    RETURNING j.id,j.run_id,j.job_type,j.executor_tier,j.status,
                              j.priority,j.idempotency_key,j.payload,j.result,
                              j.depends_on,j.attempt_count,j.max_attempts,
                              j.lease_owner,j.lease_expires_at
                    """,
                    (tiers, worker_id, lease_seconds),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                job = self._row_to_job(row)
                cur.execute(
                    """
                    INSERT INTO research_job_attempts(
                      job_id, attempt_no, worker_id, executor_tier, status
                    )
                    VALUES (%s,%s,%s,%s,'RUNNING')
                    """,
                    (job.id, job.attempt_count, worker_id, str(job.executor_tier)),
                )
                cur.execute(
                    "UPDATE research_workers SET last_heartbeat_at=now() WHERE id=%s",
                    (worker_id,),
                )
            conn.commit()
        return job

    def heartbeat_lease(self, job_id: str, worker_id: str, lease_seconds: int = 120) -> None:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET lease_expires_at=now() + make_interval(secs => %s), updated_at=now()
                    WHERE id=%s AND status='RUNNING' AND lease_owner=%s
                    """,
                    (lease_seconds, job_id, worker_id),
                )
                if cur.rowcount != 1:
                    raise ValueError(f"worker {worker_id} does not hold running job {job_id}")
                cur.execute(
                    "UPDATE research_workers SET last_heartbeat_at=now() WHERE id=%s",
                    (worker_id,),
                )
            conn.commit()

    def complete(self, job: ResearchJob, worker_id: str, result: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET status='COMPLETED', result=%s, lease_owner=NULL,
                        lease_expires_at=NULL, updated_at=now()
                    WHERE id=%s AND status='RUNNING' AND lease_owner=%s
                    """,
                    (Jsonb(result), job.id, worker_id),
                )
                if cur.rowcount != 1:
                    raise ValueError(f"worker {worker_id} cannot complete job {job.id}")
                cur.execute(
                    """
                    UPDATE research_job_attempts
                    SET status='COMPLETED', finished_at=now(), detail=%s
                    WHERE job_id=%s AND attempt_no=%s
                    """,
                    (Jsonb({"result_recorded": True}), job.id, job.attempt_count),
                )

                execution = result.get("_execution") if isinstance(result, dict) else None
                usage = execution.get("usage") if isinstance(execution, dict) else None
                if isinstance(execution, dict) and isinstance(usage, dict):
                    cur.execute(
                        """
                        INSERT INTO research_usage_events(
                          run_id,job_id,executor_tier,provider,model,
                          input_tokens,output_tokens
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(job_id) DO NOTHING
                        """,
                        (
                            job.run_id,
                            job.id,
                            str(job.executor_tier),
                            str(execution.get("provider") or "unknown"),
                            execution.get("model"),
                            int(usage.get("input_tokens") or 0),
                            int(usage.get("output_tokens") or 0),
                        ),
                    )
            conn.commit()

    def fail(
        self,
        job: ResearchJob,
        worker_id: str,
        error: dict[str, Any],
        *,
        retryable: bool = True,
        retry_delay_seconds: int = 0,
    ) -> str:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT count(*)::int
                    FROM research_job_attempts
                    WHERE job_id=%s
                      AND status IN ('FAILED','RETRY','EXPIRED')
                    """,
                    (job.id,),
                )
                prior_failures = int(cur.fetchone()[0])
                terminal = (not retryable) or (
                    prior_failures + 1 >= job.max_attempts
                )
                new_status = "FAILED" if terminal else "QUEUED"
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET status=%s, last_error=%s, lease_owner=NULL, lease_expires_at=NULL,
                        available_at=CASE WHEN %s='QUEUED'
                          THEN now() + make_interval(secs => %s) ELSE available_at END,
                        updated_at=now()
                    WHERE id=%s AND status='RUNNING' AND lease_owner=%s
                    """,
                    (
                        new_status,
                        Jsonb(error),
                        new_status,
                        retry_delay_seconds,
                        job.id,
                        worker_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise ValueError(f"worker {worker_id} cannot fail job {job.id}")
                cur.execute(
                    """
                    UPDATE research_job_attempts
                    SET status=%s, finished_at=now(), detail=%s
                    WHERE job_id=%s AND attempt_no=%s
                    """,
                    ("FAILED" if terminal else "RETRY", Jsonb(error), job.id, job.attempt_count),
                )
                if terminal:
                    cur.execute(
                        """
                        UPDATE research_runs
                        SET status='FAILED', current_phase=%s, updated_at=now()
                        WHERE id=%s AND status NOT IN ('COMPLETED','CANCELLED')
                        """,
                        (str(job.job_type), job.run_id),
                    )
            conn.commit()
        return new_status

    def recover_expired_leases(self) -> list[str]:
        recovered: list[str] = []
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, run_id, job_type, executor_tier, status, priority,
                           idempotency_key, payload, result, depends_on, attempt_count,
                           max_attempts, lease_owner, lease_expires_at
                    FROM research_jobs
                    WHERE status='RUNNING' AND lease_expires_at <= now()
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED
                    """
                )
                jobs = [self._row_to_job(row) for row in cur.fetchall()]
                for job in jobs:
                    cur.execute(
                        """
                        SELECT count(*)::int
                        FROM research_job_attempts
                        WHERE job_id=%s
                          AND status IN ('FAILED','RETRY','EXPIRED')
                        """,
                        (job.id,),
                    )
                    prior_failures = int(cur.fetchone()[0])
                    terminal = prior_failures + 1 >= job.max_attempts
                    status = "FAILED" if terminal else "QUEUED"
                    cur.execute(
                        """
                        UPDATE research_jobs
                        SET status=%s, lease_owner=NULL, lease_expires_at=NULL,
                            last_error=jsonb_build_object('type','LEASE_EXPIRED'),
                            updated_at=now()
                        WHERE id=%s
                        """,
                        (status, job.id),
                    )
                    cur.execute(
                        """
                        UPDATE research_job_attempts
                        SET status=%s, finished_at=now(),
                            detail=jsonb_build_object('type','LEASE_EXPIRED')
                        WHERE job_id=%s AND attempt_no=%s AND status='RUNNING'
                        """,
                        ("FAILED" if terminal else "EXPIRED", job.id, job.attempt_count),
                    )
                    if terminal:
                        cur.execute(
                            "UPDATE research_runs SET status='FAILED', updated_at=now() WHERE id=%s",
                            (job.run_id,),
                        )
                    recovered.append(job.id)
            conn.commit()
        return recovered

    def annotate_cancelled(
        self,
        job_id: str,
        detail: dict[str, Any],
    ) -> None:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET status='CANCELLED', last_error=%s, updated_at=now()
                    WHERE id=%s AND status IN ('QUEUED','CANCELLED')
                    """,
                    (Jsonb(detail), job_id),
                )
                if cur.rowcount != 1:
                    raise ValueError(
                        f"job {job_id} cannot be cancelled after starting"
                    )
            conn.commit()

    def cancel_queued_except(
        self,
        *,
        run_id: str,
        job_type: JobType,
        keep_idempotency_keys: Iterable[str],
    ) -> int:
        keep = list(keep_idempotency_keys)
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                if keep:
                    cur.execute(
                        """
                        UPDATE research_jobs
                        SET status='CANCELLED', updated_at=now()
                        WHERE run_id=%s AND job_type=%s AND status='QUEUED'
                          AND NOT (idempotency_key = ANY(%s))
                        """,
                        (run_id, str(job_type), keep),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE research_jobs
                        SET status='CANCELLED', updated_at=now()
                        WHERE run_id=%s AND job_type=%s AND status='QUEUED'
                        """,
                        (run_id, str(job_type)),
                    )
                count = cur.rowcount
            conn.commit()
        return count

    def jobs(self, run_id: str, job_type: JobType | None = None) -> list[ResearchJob]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                if job_type is None:
                    cur.execute(
                        """
                        SELECT id,run_id,job_type,executor_tier,status,priority,
                               idempotency_key,payload,result,depends_on,attempt_count,
                               max_attempts,lease_owner,lease_expires_at
                        FROM research_jobs WHERE run_id=%s ORDER BY created_at,id
                        """,
                        (run_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id,run_id,job_type,executor_tier,status,priority,
                               idempotency_key,payload,result,depends_on,attempt_count,
                               max_attempts,lease_owner,lease_expires_at
                        FROM research_jobs WHERE run_id=%s AND job_type=%s
                        ORDER BY created_at,id
                        """,
                        (run_id, str(job_type)),
                    )
                return [self._row_to_job(row) for row in cur.fetchall()]

    def run_state(self, run_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,topic,status,current_phase,policy,knowledge_checkpoint_id,
                           human_gate_state,created_at,updated_at
                    FROM research_runs WHERE id=%s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"run not found: {run_id}")
                cols = [d.name for d in cur.description]
                return dict(zip(cols, row, strict=True))

    def set_phase(
        self,
        run_id: str,
        phase: JobType,
        *,
        knowledge_checkpoint_id: str | None = None,
    ) -> None:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE research_runs
                    SET current_phase=%s,
                        knowledge_checkpoint_id=COALESCE(%s, knowledge_checkpoint_id),
                        updated_at=now()
                    WHERE id=%s AND status='RUNNING'
                    """,
                    (str(phase), knowledge_checkpoint_id, run_id),
                )
            conn.commit()

    def ensure_human_review(self, run_id: str, payload: dict[str, Any]) -> str:
        from psycopg.types.json import Jsonb

        job_id = self.enqueue(
            run_id=run_id,
            job_type=JobType.HUMAN_REVIEW,
            idempotency_key="stage:human-review",
            payload=payload,
            priority=1000,
            status="WAITING_HUMAN",
        )
        request_id = f"human-review:{run_id}"
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO human_review_requests(id,run_id,job_id,status,payload)
                    VALUES (%s,%s,%s,'PENDING',%s)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (request_id, run_id, job_id, Jsonb(payload)),
                )
                cur.execute(
                    """
                    UPDATE research_runs
                    SET status='WAITING_HUMAN', current_phase='HUMAN_REVIEW',
                        human_gate_state='PENDING', updated_at=now()
                    WHERE id=%s AND status='RUNNING'
                    """,
                    (run_id,),
                )
            conn.commit()
        return request_id

    def resolve_human_review(
        self,
        request_id: str,
        *,
        approved: bool,
        decision: dict[str, Any],
    ) -> None:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, job_id, status
                    FROM human_review_requests WHERE id=%s FOR UPDATE
                    """,
                    (request_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"human review not found: {request_id}")
                run_id, job_id, status = row
                if status != "PENDING":
                    return
                final = "APPROVED" if approved else "REJECTED"
                cur.execute(
                    """
                    UPDATE human_review_requests
                    SET status=%s, decision=%s, decided_at=now()
                    WHERE id=%s
                    """,
                    (final, Jsonb(decision), request_id),
                )
                cur.execute(
                    """
                    UPDATE research_jobs
                    SET status='COMPLETED', result=%s, updated_at=now()
                    WHERE id=%s AND status='WAITING_HUMAN'
                    """,
                    (Jsonb({"approved": approved, "decision": decision}), job_id),
                )
                cur.execute(
                    """
                    UPDATE research_runs
                    SET status='COMPLETED', human_gate_state=%s, updated_at=now()
                    WHERE id=%s
                    """,
                    (final, run_id),
                )
            conn.commit()


class ResearchPlanner:
    """Idempotent reconciler for the v1 autonomous research cycle."""

    def __init__(self, queue: OrchestratorStore):
        self.queue = queue

    @staticmethod
    def _completed(jobs: list[ResearchJob], kind: JobType) -> list[ResearchJob]:
        return [j for j in jobs if j.job_type == kind and j.status == "COMPLETED"]

    @staticmethod
    def _all_completed(jobs: list[ResearchJob], kind: JobType) -> bool:
        stage = [j for j in jobs if j.job_type == kind]
        return bool(stage) and all(j.status == "COMPLETED" for j in stage)

    def reconcile(self, run_id: str) -> None:
        state = self.queue.run_state(run_id)
        if state["status"] != "RUNNING":
            return
        policy = state.get("policy") or {}
        jobs = self.queue.jobs(run_id)

        discovery = self._completed(jobs, JobType.DISCOVERY)
        if discovery:
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.SCREEN,
                idempotency_key="stage:screen",
                payload={
                    "candidates": bounded_discovery_candidates(discovery[0])
                },
                depends_on=[discovery[0].id],
                priority=90,
            )

        jobs = self.queue.jobs(run_id)
        screen = self._completed(jobs, JobType.SCREEN)
        if screen and discovery:
            retained = canonical_screened_candidates(
                discovery[0],
                screen[0],
            )
            existing_acquisition_keys = {
                job.idempotency_key
                for job in jobs
                if job.job_type == JobType.ACQUIRE
            }

            # Preserve unsupported retained candidates as explicit cancelled
            # acquisition jobs. Raw discovery/screen results remain available
            # for future adapters.
            for item in retained:
                if acquisition_candidate_supported(item):
                    continue
                digest = hashlib.sha256(
                    json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:16]
                idem = f"acquire-unsupported:{digest}"
                if idem in existing_acquisition_keys:
                    continue
                job_id = self.queue.enqueue(
                    run_id=run_id,
                    job_type=JobType.ACQUIRE,
                    idempotency_key=idem,
                    payload={"item": item},
                    depends_on=[screen[0].id],
                    priority=80,
                    status="CANCELLED",
                )
                self.queue.annotate_cancelled(
                    job_id,
                    {
                        "type": "SKIPPED_UNSUPPORTED_SOURCE_PROVIDER",
                        "source_uri": item.get("source_uri"),
                    },
                )
                existing_acquisition_keys.add(idem)

            canonical_items = canonical_acquisition_candidates(retained)
            verified_existing = self.queue.verified_work_ids(
                item["id"] for item in canonical_items
            )
            for item in canonical_items:
                key = str(item["id"])
                idem = f"acquire:{key}"
                if idem in existing_acquisition_keys:
                    continue
                already_verified = key in verified_existing
                job_id = self.queue.enqueue(
                    run_id=run_id,
                    job_type=JobType.ACQUIRE,
                    idempotency_key=idem,
                    payload={"item": item},
                    depends_on=[screen[0].id],
                    priority=80,
                    status="CANCELLED" if already_verified else "QUEUED",
                )
                if already_verified:
                    self.queue.annotate_cancelled(
                        job_id,
                        {
                            "type": "SKIPPED_ALREADY_VERIFIED_WORK",
                            "work_id": key,
                            "source_uri": item.get("source_uri"),
                        },
                    )

        jobs = self.queue.jobs(run_id)
        for job in self._completed(jobs, JobType.ACQUIRE):
            key = str(job.payload["item"]["id"])
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.EXTRACT,
                idempotency_key=f"extract:{key}",
                payload={"item": job.payload["item"], "acquisition": job.result},
                depends_on=[job.id],
                priority=70,
            )

        jobs = self.queue.jobs(run_id)
        for job in self._completed(jobs, JobType.EXTRACT):
            key = str(job.payload["item"]["id"])
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.VERIFY,
                idempotency_key=f"verify:{key}",
                payload={"item": job.payload["item"], "extraction": job.result},
                depends_on=[job.id],
                priority=60,
            )

        jobs = self.queue.jobs(run_id)
        screen_done = self._completed(jobs, JobType.SCREEN)
        retained_items = (
            canonical_screened_candidates(discovery[0], screen_done[0])
            if discovery and screen_done
            else []
        )
        acquisition_jobs = [
            j for j in jobs
            if j.job_type == JobType.ACQUIRE
            and j.status != "CANCELLED"
        ]
        retained_count = len(acquisition_jobs)
        verifies = [j for j in jobs if j.job_type == JobType.VERIFY]

        if screen_done and not retained_items:
            self.queue.ensure_human_review(
                run_id,
                {
                    "topic": state["topic"],
                    "reason": "NO_RELEVANT_CANDIDATES",
                    "bounded_discovery_count": len(
                        bounded_discovery_candidates(discovery[0])
                    ),
                },
            )
            return

        if screen_done and retained_items and retained_count == 0:
            cancelled = [
                j for j in jobs
                if j.job_type == JobType.ACQUIRE
                and j.status == "CANCELLED"
            ]
            self.queue.ensure_human_review(
                run_id,
                {
                    "topic": state["topic"],
                    "reason": "NO_NEW_ACQUIRABLE_SOURCES",
                    "screened_candidate_ids": [
                        str(item.get("id")) for item in retained_items
                    ],
                    "cancelled_acquisition_count": len(cancelled),
                },
            )
            return

        if retained_count == len(verifies) and retained_count > 0 and all(
            j.status == "COMPLETED" for j in verifies
        ):
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.CHECKPOINT,
                idempotency_key="stage:checkpoint",
                payload={"verified_job_ids": [j.id for j in verifies]},
                depends_on=[j.id for j in verifies],
                priority=50,
            )

        jobs = self.queue.jobs(run_id)
        checkpoints = self._completed(jobs, JobType.CHECKPOINT)
        if checkpoints:
            checkpoint_id = checkpoints[0].result["knowledge_checkpoint_id"]
            self.queue.set_phase(
                run_id,
                JobType.SYNTHESIZE,
                knowledge_checkpoint_id=checkpoint_id,
            )
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.SYNTHESIZE,
                idempotency_key="stage:synthesize",
                payload={"knowledge_checkpoint_id": checkpoint_id},
                depends_on=[checkpoints[0].id],
                priority=40,
            )

        jobs = self.queue.jobs(run_id)
        synth = self._completed(jobs, JobType.SYNTHESIZE)
        if synth:
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.GAP_GENERATE,
                idempotency_key="stage:gap-generate",
                payload={"synthesis": synth[0].result},
                depends_on=[synth[0].id],
                priority=35,
            )

        jobs = self.queue.jobs(run_id)
        gap_jobs = self._completed(jobs, JobType.GAP_GENERATE)
        if gap_jobs:
            gaps = bounded_gap_candidates(gap_jobs[0].result, policy)
            existing_by_key = {job.idempotency_key: job for job in jobs}
            allowed_search_keys: set[str] = set()
            for gap in gaps:
                gap_id = str(gap["id"])
                for index, query in enumerate(gap.get("search_queries", [])):
                    key = f"novelty-search:{gap_id}:{index}"
                    allowed_search_keys.add(key)
                    if key not in existing_by_key:
                        self.queue.enqueue(
                            run_id=run_id,
                            job_type=JobType.NOVELTY_SEARCH,
                            idempotency_key=key,
                            payload={"gap": gap, "query": query},
                            depends_on=[gap_jobs[0].id],
                            priority=30,
                        )
            self.queue.cancel_queued_except(
                run_id=run_id,
                job_type=JobType.NOVELTY_SEARCH,
                keep_idempotency_keys=allowed_search_keys,
            )

        jobs = self.queue.jobs(run_id)
        gap_jobs = self._completed(jobs, JobType.GAP_GENERATE)
        if gap_jobs:
            gaps = bounded_gap_candidates(gap_jobs[0].result, policy)
            allowed_arbitration_keys: set[str] = set()
            for gap in gaps:
                gap_id = str(gap["id"])
                search_keys = {
                    f"novelty-search:{gap_id}:{index}"
                    for index, _ in enumerate(gap.get("search_queries", []))
                }
                searches = [
                    j
                    for j in jobs
                    if j.job_type == JobType.NOVELTY_SEARCH
                    and j.idempotency_key in search_keys
                ]
                if search_keys and len(searches) == len(search_keys) and all(
                    j.status == "COMPLETED" for j in searches
                ):
                    arbitration_key = f"novelty-arbitrate:{gap_id}"
                    allowed_arbitration_keys.add(arbitration_key)
                    if arbitration_key not in {job.idempotency_key for job in jobs}:
                        self.queue.enqueue(
                            run_id=run_id,
                            job_type=JobType.NOVELTY_ARBITRATE,
                            idempotency_key=arbitration_key,
                            payload={
                                "gap": gap,
                                "search_results": [
                                    {"query": j.payload["query"], "result": j.result}
                                    for j in searches
                                ],
                            },
                            depends_on=[j.id for j in searches],
                            priority=20,
                        )
            self.queue.cancel_queued_except(
                run_id=run_id,
                job_type=JobType.NOVELTY_ARBITRATE,
                keep_idempotency_keys=allowed_arbitration_keys,
            )

        jobs = self.queue.jobs(run_id)
        gap_jobs = self._completed(jobs, JobType.GAP_GENERATE)
        if gap_jobs:
            gaps = bounded_gap_candidates(gap_jobs[0].result, policy)
            arbitration_keys = {
                f"novelty-arbitrate:{gap['id']}"
                for gap in gaps
            }
            arbitrations = [
                j
                for j in self._completed(jobs, JobType.NOVELTY_ARBITRATE)
                if j.idempotency_key in arbitration_keys
            ]
            if gaps and len(arbitrations) == len(gaps):
                self.queue.ensure_human_review(
                    run_id,
                    {
                        "topic": state["topic"],
                        "knowledge_checkpoint_id": self.queue.run_state(run_id)[
                            "knowledge_checkpoint_id"
                        ],
                        "gap_decisions": [
                            {"gap_id": j.payload["gap"]["id"], "decision": j.result}
                            for j in arbitrations
                        ],
                    },
                )


class AgentWorker:
    def __init__(
        self,
        *,
        worker_id: str,
        queue: OrchestratorStore,
        planner: ResearchPlanner,
        supported_tiers: Iterable[ExecutorTier],
        executors: dict[ExecutorTier, JobExecutor],
        lease_seconds: int = 120,
        backoff_provider: str | None = None,
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.planner = planner
        self.supported_tiers = tuple(supported_tiers)
        self.executors = executors
        self.lease_seconds = lease_seconds
        self.backoff_provider = backoff_provider
        queue.register_worker(worker_id, self.supported_tiers)

    def run_once(self) -> ResearchJob | None:
        if (
            self.backoff_provider
            and self.queue.provider_backoff(self.backoff_provider) is not None
        ):
            self.queue.worker_heartbeat(self.worker_id)
            return None
        job = self.queue.lease_next(
            worker_id=self.worker_id,
            supported_tiers=self.supported_tiers,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return None
        executor = self.executors.get(job.executor_tier)
        if executor is None:
            self.queue.fail(
                job,
                self.worker_id,
                {"type": "NO_EXECUTOR", "tier": str(job.executor_tier)},
                retryable=False,
            )
            return job
        try:
            result = executor.execute(job)
        except JobExecutionError as exc:
            error = {
                "type": type(exc).__name__,
                "code": exc.code,
                "message": str(exc),
                "detail": exc.detail,
            }
            if exc.code == "CODEX_PLAN_LIMIT" and self.backoff_provider:
                self.queue.defer_for_provider_limit(
                    job,
                    self.worker_id,
                    error,
                    provider=self.backoff_provider,
                    retry_after_seconds=int(
                        exc.detail.get("retry_after_seconds") or 3600
                    ),
                )
            else:
                self.queue.fail(
                    job,
                    self.worker_id,
                    error,
                    retryable=exc.retryable,
                )
        except Exception as exc:
            self.queue.fail(
                job,
                self.worker_id,
                {"type": type(exc).__name__, "message": str(exc)},
                retryable=True,
            )
        else:
            self.queue.complete(job, self.worker_id, result)
            self.planner.reconcile(job.run_id)
        return job

    def run_until_idle(self, max_jobs: int = 100) -> int:
        count = 0
        for _ in range(max_jobs):
            job = self.run_once()
            if job is None:
                break
            count += 1
        return count
