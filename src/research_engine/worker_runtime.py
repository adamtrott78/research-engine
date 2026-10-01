from __future__ import annotations

import os
import socket
import time
from typing import Iterable

from .openai_executor import OpenAIResponsesExecutor
from .orchestration import (
    AgentWorker,
    ExecutorTier,
    OrchestratorStore,
    ResearchPlanner,
)
from .postgres import PostgresStore


def semantic_worker_from_env(
    *,
    tiers: Iterable[ExecutorTier] = (
        ExecutorTier.CHEAP,
        ExecutorTier.STANDARD,
        ExecutorTier.STRONG,
    ),
    worker_id: str | None = None,
    lease_seconds: int = 300,
) -> AgentWorker:
    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    store = PostgresStore(dsn)
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)
    selected = tuple(tiers)
    executors = {
        tier: OpenAIResponsesExecutor(tier=tier)
        for tier in selected
        if tier in {ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG}
    }
    return AgentWorker(
        worker_id=worker_id or f"semantic:{socket.gethostname()}:{os.getpid()}",
        queue=queue,
        planner=planner,
        supported_tiers=selected,
        executors=executors,
        lease_seconds=lease_seconds,
    )


def run_semantic_worker_loop(
    *,
    tiers: Iterable[ExecutorTier],
    worker_id: str | None = None,
    poll_seconds: float = 2.0,
    lease_seconds: int = 300,
    max_idle_polls: int | None = None,
) -> int:
    worker = semantic_worker_from_env(
        tiers=tiers,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    idle = 0
    completed = 0
    while True:
        job = worker.run_once()
        if job is None:
            idle += 1
            if max_idle_polls is not None and idle >= max_idle_polls:
                return completed
            time.sleep(poll_seconds)
            continue
        idle = 0
        completed += 1
