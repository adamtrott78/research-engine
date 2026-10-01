from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .orchestration import (
    AgentWorker,
    ExecutorTier,
    OrchestratorStore,
    ResearchPlanner,
)
from .postgres import PostgresStore


def _code_sha() -> str | None:
    explicit = os.getenv("RESEARCH_ENGINE_CODE_SHA")
    if explicit:
        return explicit.strip() or None
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for cwd in candidates:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return None


def _semantic_metadata(
    selected: tuple[ExecutorTier, ...],
    *,
    logical_agent_id: str,
    backend: str,
) -> dict[str, object]:
    model = None
    reasoning_effort = None
    if backend == "codex" and len(selected) == 1:
        from .codex_executor import profiles_from_env

        profile = profiles_from_env()[selected[0]]
        model = profile.model
        reasoning_effort = profile.reasoning_effort
    return {
        "logical_agent_id": logical_agent_id,
        "host_name": socket.gethostname(),
        "process_id": os.getpid(),
        "code_sha": _code_sha(),
        "backend": backend,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }


def _semantic_delegate(tier: ExecutorTier):
    backend = os.getenv("RESEARCH_ENGINE_SEMANTIC_BACKEND", "codex").strip().lower()
    if backend == "codex":
        from .codex_executor import CodexExecExecutor

        return CodexExecExecutor(tier=tier)
    if backend == "openai_api":
        from .openai_executor import OpenAIResponsesExecutor

        return OpenAIResponsesExecutor(tier=tier)
    raise ValueError(f"unsupported RESEARCH_ENGINE_SEMANTIC_BACKEND: {backend}")


def semantic_worker_from_env(
    *,
    tiers: Iterable[ExecutorTier] = (
        ExecutorTier.CHEAP,
        ExecutorTier.STANDARD,
        ExecutorTier.STRONG,
    ),
    worker_id: str | None = None,
    logical_agent_id: str | None = None,
    lease_seconds: int = 300,
) -> AgentWorker:
    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    artifact_root = os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
    store = PostgresStore(dsn)
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)
    selected = tuple(tiers)

    from .semantic_pipeline import (
        AuthoritativeSemanticExecutor,
        ResearchContextHydrator,
        SemanticResultMaterializer,
    )

    hydrator = ResearchContextHydrator(
        store=store,
        artifact_root=artifact_root,
        verify_page_radius=int(
            os.getenv("RESEARCH_ENGINE_VERIFY_PAGE_RADIUS", "0")
        ),
    )
    materializer = SemanticResultMaterializer(store=store)
    executors = {
        tier: AuthoritativeSemanticExecutor(
            delegate=_semantic_delegate(tier),
            hydrator=hydrator,
            materializer=materializer,
        )
        for tier in selected
        if tier in {ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG}
    }
    backend = os.getenv("RESEARCH_ENGINE_SEMANTIC_BACKEND", "codex").strip().lower()
    if logical_agent_id is None:
        tier_name = "+".join(str(tier).lower() for tier in selected) or "none"
        logical_agent_id = f"semantic:{tier_name}:standalone"
    return AgentWorker(
        worker_id=worker_id or f"semantic:{socket.gethostname()}:{os.getpid()}",
        queue=queue,
        planner=planner,
        supported_tiers=selected,
        executors=executors,
        lease_seconds=lease_seconds,
        backoff_provider=(
            "codex-chatgpt-plan" if backend == "codex" else None
        ),
        worker_metadata=_semantic_metadata(
            selected,
            logical_agent_id=logical_agent_id,
            backend=backend,
        ),
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


def deterministic_worker_from_env(
    *,
    worker_id: str | None = None,
    logical_agent_id: str | None = None,
    lease_seconds: int = 300,
) -> AgentWorker:
    from .deterministic_executor import DeterministicResearchExecutor

    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    artifact_root = os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
    store = PostgresStore(dsn)
    queue = OrchestratorStore(store)
    planner = ResearchPlanner(queue)
    executor = DeterministicResearchExecutor(
        store=store,
        artifact_root=artifact_root,
    )
    logical_agent_id = logical_agent_id or "deterministic:standalone"
    return AgentWorker(
        worker_id=worker_id or f"deterministic:{socket.gethostname()}:{os.getpid()}",
        queue=queue,
        planner=planner,
        supported_tiers=[ExecutorTier.DETERMINISTIC],
        executors={ExecutorTier.DETERMINISTIC: executor},
        lease_seconds=lease_seconds,
        worker_metadata={
            "logical_agent_id": logical_agent_id,
            "host_name": socket.gethostname(),
            "process_id": os.getpid(),
            "code_sha": _code_sha(),
            "backend": "deterministic",
            "model": None,
            "reasoning_effort": None,
        },
    )


def run_deterministic_worker_loop(
    *,
    worker_id: str | None = None,
    poll_seconds: float = 2.0,
    lease_seconds: int = 300,
    max_idle_polls: int | None = None,
) -> int:
    worker = deterministic_worker_from_env(
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
