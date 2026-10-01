from __future__ import annotations

import hashlib
import json
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
        self.enqueue(
            run_id=run_id,
            job_type=JobType.DISCOVERY,
            idempotency_key="stage:discovery",
            payload=discovery_payload or {"topic": topic},
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
                cur.execute(
                    """
                    WITH candidate AS (
                      SELECT j.id
                      FROM research_jobs j
                      WHERE j.status='QUEUED'
                        AND j.available_at <= now()
                        AND j.executor_tier = ANY(%s)
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

        terminal = (not retryable) or job.attempt_count >= job.max_attempts
        new_status = "FAILED" if terminal else "QUEUED"
        with self.store.connect() as conn:
            with conn.cursor() as cur:
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
                    terminal = job.attempt_count >= job.max_attempts
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
        jobs = self.queue.jobs(run_id)

        discovery = self._completed(jobs, JobType.DISCOVERY)
        if discovery:
            self.queue.enqueue(
                run_id=run_id,
                job_type=JobType.SCREEN,
                idempotency_key="stage:screen",
                payload={"candidates": discovery[0].result.get("candidates", [])},
                depends_on=[discovery[0].id],
                priority=90,
            )

        jobs = self.queue.jobs(run_id)
        screen = self._completed(jobs, JobType.SCREEN)
        if screen:
            retained = screen[0].result.get("retained", [])
            for item in retained:
                key = str(item["id"])
                self.queue.enqueue(
                    run_id=run_id,
                    job_type=JobType.ACQUIRE,
                    idempotency_key=f"acquire:{key}",
                    payload={"item": item},
                    depends_on=[screen[0].id],
                    priority=80,
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
        retained_count = len(screen_done[0].result.get("retained", [])) if screen_done else 0
        verifies = [j for j in jobs if j.job_type == JobType.VERIFY]
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
            for gap in gap_jobs[0].result.get("gaps", []):
                gap_id = str(gap["id"])
                for index, query in enumerate(gap.get("search_queries", [])):
                    self.queue.enqueue(
                        run_id=run_id,
                        job_type=JobType.NOVELTY_SEARCH,
                        idempotency_key=f"novelty-search:{gap_id}:{index}",
                        payload={"gap": gap, "query": query},
                        depends_on=[gap_jobs[0].id],
                        priority=30,
                    )

        jobs = self.queue.jobs(run_id)
        gap_jobs = self._completed(jobs, JobType.GAP_GENERATE)
        if gap_jobs:
            for gap in gap_jobs[0].result.get("gaps", []):
                gap_id = str(gap["id"])
                searches = [
                    j for j in jobs
                    if j.job_type == JobType.NOVELTY_SEARCH
                    and str(j.payload["gap"]["id"]) == gap_id
                ]
                if searches and all(j.status == "COMPLETED" for j in searches):
                    self.queue.enqueue(
                        run_id=run_id,
                        job_type=JobType.NOVELTY_ARBITRATE,
                        idempotency_key=f"novelty-arbitrate:{gap_id}",
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

        jobs = self.queue.jobs(run_id)
        gap_jobs = self._completed(jobs, JobType.GAP_GENERATE)
        if gap_jobs:
            gaps = gap_jobs[0].result.get("gaps", [])
            arbitrations = self._completed(jobs, JobType.NOVELTY_ARBITRATE)
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
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.planner = planner
        self.supported_tiers = tuple(supported_tiers)
        self.executors = executors
        self.lease_seconds = lease_seconds
        queue.register_worker(worker_id, self.supported_tiers)

    def run_once(self) -> ResearchJob | None:
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
            self.queue.fail(
                job,
                self.worker_id,
                {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "message": str(exc),
                    "detail": exc.detail,
                },
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
