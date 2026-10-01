from pathlib import Path
from types import SimpleNamespace

import os
import pytest

from research_engine.artifact_store import ContentAddressedArtifactStore


class ScriptedSemanticExecutor:
    def execute(self, job):
        jt = str(job.job_type)
        if jt == "DISCOVERY":
            return {
                "candidates": [
                    {
                        "id": "work:dynamic-rf",
                        "title": "Dynamic RF",
                        "source_uri": "https://arxiv.org/abs/2605.15593v1",
                        "rationale": "counterevidence",
                    },
                    {
                        "id": "work:ood-rf",
                        "title": "OOD RF",
                        "source_uri": "https://arxiv.org/abs/2606.12718v1",
                        "rationale": "counterevidence",
                    },
                ]
            }
        if jt == "SCREEN":
            return {"retained": job.payload["candidates"]}
        if jt == "EXTRACT":
            return {"evidence": [{"id": f"e:{job.payload['item']['id']}"}]}
        if jt == "VERIFY":
            return {
                "verified": True,
                "item_id": job.payload["item"]["id"],
                "accepted_evidence_ids": [f"e:{job.payload['item']['id']}"],
                "issues": [],
            }
        if jt == "SYNTHESIZE":
            return {"themes": ["open-set RF"], "statements": []}
        if jt == "GAP_GENERATE":
            return {
                "gaps": [
                    {
                        "id": "gap:one",
                        "statement": "candidate",
                        "gap_type": "TEST",
                        "rationale": "test",
                        "search_queries": ["q1", "q2"],
                    }
                ]
            }
        if jt == "NOVELTY_SEARCH":
            return {"query": job.payload["query"], "counterevidence": [], "notes": ""}
        if jt == "NOVELTY_ARBITRATE":
            return {
                "status": "NEEDS_HUMAN_REVIEW",
                "reasoning": "requires human judgment",
                "surviving_statement": "candidate",
                "search_count": len(job.payload["search_results"]),
            }
        raise AssertionError(jt)


class FakeArxivProvider:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store

    def acquire_pdf(self, arxiv_id):
        from research_engine.arxiv import ArxivAcquisition

        body = b"%PDF-" + arxiv_id.encode()
        receipt = self.artifact_store.put_bytes(body)
        return ArxivAcquisition(
            arxiv_id=arxiv_id,
            versioned_pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            resolved_url=f"https://arxiv.org/pdf/{arxiv_id}",
            media_type="application/pdf",
            receipt=receipt,
        )


@pytest.mark.integration
def test_mixed_production_deterministic_workers_reach_human_gate(tmp_path):
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.deterministic_executor import DeterministicResearchExecutor
    from research_engine.orchestration import (
        AgentWorker,
        ExecutorTier,
        OrchestratorStore,
        ResearchPlanner,
    )
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE human_review_requests, research_job_attempts, research_jobs,
                         research_workers, research_runs, novelty_challenges, gap_candidates,
                         knowledge_checkpoints, verification_events, evidence_relations,
                         evidence_objects, source_locators, extraction_runs,
                         acquisition_attempts, discovery_runs, external_identifiers,
                         source_artifacts, work_versions, works CASCADE
                """
            )
        conn.commit()

    queue.create_run(
        run_id="research-run:mixed-v1",
        topic="open-set RF fingerprinting",
    )
    planner = ResearchPlanner(queue)

    artifact_store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    deterministic = DeterministicResearchExecutor(
        store=store,
        artifact_root=tmp_path / "artifacts",
        arxiv_provider=FakeArxivProvider(artifact_store),
    )
    semantic = ScriptedSemanticExecutor()

    workers = [
        AgentWorker(
            worker_id="mixed:cheap",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.CHEAP],
            executors={ExecutorTier.CHEAP: semantic},
        ),
        AgentWorker(
            worker_id="mixed:standard",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STANDARD],
            executors={ExecutorTier.STANDARD: semantic},
        ),
        AgentWorker(
            worker_id="mixed:strong",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STRONG],
            executors={ExecutorTier.STRONG: semantic},
        ),
        AgentWorker(
            worker_id="mixed:deterministic",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.DETERMINISTIC],
            executors={ExecutorTier.DETERMINISTIC: deterministic},
        ),
    ]

    for _ in range(50):
        for worker in workers:
            worker.run_until_idle()
        planner.reconcile("research-run:mixed-v1")
        if queue.run_state("research-run:mixed-v1")["status"] == "WAITING_HUMAN":
            break

    state = queue.run_state("research-run:mixed-v1")
    assert state["status"] == "WAITING_HUMAN"
    assert state["knowledge_checkpoint_id"].startswith("knowledge-checkpoint:sha256:")

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM works")
            assert cur.fetchone()[0] == 2
            cur.execute("SELECT count(*) FROM source_artifacts")
            assert cur.fetchone()[0] == 2
            cur.execute(
                "SELECT record_counts FROM knowledge_checkpoints WHERE id=%s",
                (state["knowledge_checkpoint_id"],),
            )
            counts = cur.fetchone()[0]
            assert counts["works"] == 2
            assert counts["source_artifacts"] == 2
            cur.execute(
                "SELECT count(*) FROM human_review_requests WHERE status='PENDING'"
            )
            assert cur.fetchone()[0] == 1
