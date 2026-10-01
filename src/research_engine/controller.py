from __future__ import annotations

import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from .orchestration import (
    AgentWorker,
    ExecutorTier,
    OrchestratorStore,
    ResearchPlanner,
)


TERMINAL_RUN_STATUSES = {
    "WAITING_HUMAN",
    "PAUSED_BUDGET",
    "FAILED",
    "COMPLETED",
    "CANCELLED",
}


@dataclass(frozen=True)
class ControllerResult:
    run_id: str
    status: str
    cycles: int
    jobs_processed: int
    recovered_leases: int
    state: dict[str, Any]
    human_review: dict[str, Any] | None


class AutonomousResearchController:
    """Drive the durable mixed-worker research cycle to a terminal/human gate."""

    def __init__(
        self,
        *,
        queue: OrchestratorStore,
        planner: ResearchPlanner,
        workers: Iterable[AgentWorker],
        poll_seconds: float = 1.0,
    ):
        self.queue = queue
        self.planner = planner
        self.workers = tuple(workers)
        self.poll_seconds = poll_seconds

    def _ensure_run(
        self,
        *,
        run_id: str,
        topic: str,
        policy: dict[str, Any] | None,
        discovery_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            state = self.queue.run_state(run_id)
        except ValueError:
            self.queue.create_run(
                run_id=run_id,
                topic=topic,
                policy=policy,
                discovery_payload=discovery_payload,
            )
            return self.queue.run_state(run_id)

        if state["topic"] != topic:
            raise ValueError(
                f"research run {run_id} already exists for a different topic"
            )
        return state

    def _human_review(self, run_id: str) -> dict[str, Any] | None:
        with self.queue.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,status,payload,decision,created_at,decided_at
                    FROM human_review_requests
                    WHERE run_id=%s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d.name for d in cur.description]
                return dict(zip(cols, row, strict=True))

    def run(
        self,
        *,
        topic: str,
        run_id: str | None = None,
        policy: dict[str, Any] | None = None,
        discovery_payload: dict[str, Any] | None = None,
        max_cycles: int | None = None,
    ) -> ControllerResult:
        run_id = run_id or f"research-run:{uuid.uuid4()}"
        state = self._ensure_run(
            run_id=run_id,
            topic=topic,
            policy=policy,
            discovery_payload=discovery_payload,
        )
        cycles = 0
        processed = 0
        recovered_total = 0

        if state["status"] in TERMINAL_RUN_STATUSES:
            return ControllerResult(
                run_id=run_id,
                status=state["status"],
                cycles=0,
                jobs_processed=0,
                recovered_leases=0,
                state=state,
                human_review=self._human_review(run_id),
            )

        while True:
            cycles += 1
            recovered = self.queue.recover_expired_leases()
            recovered_total += len(recovered)
            self.planner.reconcile(run_id)

            progressed = bool(recovered)
            for worker in self.workers:
                job = worker.run_once()
                if job is not None:
                    processed += 1
                    progressed = True

            self.planner.reconcile(run_id)
            state = self.queue.run_state(run_id)
            if state["status"] in TERMINAL_RUN_STATUSES:
                return ControllerResult(
                    run_id=run_id,
                    status=state["status"],
                    cycles=cycles,
                    jobs_processed=processed,
                    recovered_leases=recovered_total,
                    state=state,
                    human_review=self._human_review(run_id),
                )

            if max_cycles is not None and cycles >= max_cycles:
                raise TimeoutError(
                    f"research run {run_id} did not reach a terminal state "
                    f"within {max_cycles} controller cycles"
                )
            if not progressed and self.poll_seconds > 0:
                time.sleep(self.poll_seconds)


def controller_from_env(
    *,
    worker_id_prefix: str | None = None,
    lease_seconds: int = 300,
    poll_seconds: float = 1.0,
    semantic_client: Any | None = None,
) -> AutonomousResearchController:
    from .deterministic_executor import DeterministicResearchExecutor
    from .openai_executor import OpenAIResponsesExecutor
    from .postgres import PostgresStore
    from .semantic_pipeline import (
        AuthoritativeSemanticExecutor,
        ResearchContextHydrator,
        SemanticResultMaterializer,
    )

    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    artifact_root = os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
    store = PostgresStore(dsn)
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)
    prefix = worker_id_prefix or f"controller:{socket.gethostname()}:{os.getpid()}"

    deterministic = DeterministicResearchExecutor(
        store=store,
        artifact_root=artifact_root,
    )
    hydrator = ResearchContextHydrator(store=store, artifact_root=artifact_root)
    materializer = SemanticResultMaterializer(store=store)

    workers: list[AgentWorker] = [
        AgentWorker(
            worker_id=f"{prefix}:deterministic",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.DETERMINISTIC],
            executors={ExecutorTier.DETERMINISTIC: deterministic},
            lease_seconds=lease_seconds,
        )
    ]
    for tier in (ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG):
        semantic = AuthoritativeSemanticExecutor(
            delegate=OpenAIResponsesExecutor(tier=tier, client=semantic_client),
            hydrator=hydrator,
            materializer=materializer,
        )
        workers.append(
            AgentWorker(
                worker_id=f"{prefix}:{str(tier).lower()}",
                queue=queue,
                planner=planner,
                supported_tiers=[tier],
                executors={tier: semantic},
                lease_seconds=lease_seconds,
            )
        )

    return AutonomousResearchController(
        queue=queue,
        planner=planner,
        workers=workers,
        poll_seconds=poll_seconds,
    )
