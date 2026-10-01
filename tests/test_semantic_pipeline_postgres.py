import os
from pathlib import Path

import pytest

from research_engine.artifact_store import ContentAddressedArtifactStore


class FakeArxivProvider:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store

    def acquire_pdf(self, arxiv_id):
        from research_engine.arxiv import ArxivAcquisition

        receipt = self.artifact_store.put_bytes(
            b"%PDF-fake-autonomous-source-" + arxiv_id.encode()
        )
        return ArxivAcquisition(
            arxiv_id=arxiv_id,
            versioned_pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            resolved_url=f"https://arxiv.org/pdf/{arxiv_id}",
            media_type="application/pdf",
            receipt=receipt,
        )


class ExactTextStub:
    def __call__(self, path):
        assert Path(path).read_bytes().startswith(b"%PDF-")
        return [
            {
                "page_number": 1,
                "text": (
                    "Exact source evidence: the evaluated method detects unseen "
                    "transmitters under dynamic RF conditions."
                ),
            }
        ]


class ContextAwareSemantic:
    def execute(self, job):
        jt = str(job.job_type)
        if jt == "DISCOVERY":
            return {
                "candidates": [
                    {
                        "id": "work:agentic-real",
                        "title": "Agentic RF Test Paper",
                        "source_uri": "https://arxiv.org/abs/2605.15593v1",
                        "rationale": "relevant RF open-set evidence",
                    }
                ],
                "_execution": {"provider": "scripted", "model": "test-cheap"},
            }
        if jt == "SCREEN":
            return {
                "retained": job.payload["candidates"],
                "_execution": {"provider": "scripted", "model": "test-cheap"},
            }
        if jt == "EXTRACT":
            ctx = job.payload["_research_context"]
            assert ctx["pages"][0]["page_number"] == 1
            assert "Exact source evidence" in ctx["pages"][0]["text"]
            return {
                "evidence": [
                    {
                        "id": "model:evidence:1",
                        "type": "RESULT",
                        "normalized_form": "The method detects unseen transmitters under dynamic RF conditions.",
                        "raw_source_form": "the evaluated method detects unseen transmitters under dynamic RF conditions",
                        "page_number": 1,
                        "confidence": 0.95,
                    }
                ],
                "_execution": {"provider": "scripted", "model": "test-standard"},
            }
        if jt == "VERIFY":
            ctx = job.payload["_research_context"]
            evidence = ctx["authoritative_evidence"]
            assert len(evidence) == 1
            assert evidence[0]["source_sha256"]
            return {
                "verified": True,
                "item_id": job.payload["item"]["id"],
                "accepted_evidence_ids": [evidence[0]["id"]],
                "issues": [],
                "_execution": {"provider": "scripted", "model": "test-standard"},
            }
        if jt == "SYNTHESIZE":
            checkpoint = job.payload["_research_context"]["knowledge_checkpoint"]
            checked = [
                e for e in checkpoint["evidence"]
                if e["verification_state"] == "MACHINE_CHECKED"
            ]
            assert len(checked) == 1
            return {
                "themes": ["dynamic open-set RF"],
                "statements": [
                    {
                        "text": "Dynamic open-set RF evidence exists.",
                        "evidence_ids": [checked[0]["id"]],
                        "qualifiers": ["single-paper acceptance fixture"],
                    }
                ],
                "_execution": {"provider": "scripted", "model": "test-strong"},
            }
        if jt == "GAP_GENERATE":
            checkpoint = job.payload["_research_context"]["knowledge_checkpoint"]
            evidence_id = checkpoint["evidence"][0]["id"]
            return {
                "gaps": [
                    {
                        "id": "model:gap:1",
                        "statement": "A narrower RF robustness question remains.",
                        "gap_type": "ROBUSTNESS",
                        "rationale": "Current exact-source evidence is limited.",
                        "motivating_evidence_ids": [evidence_id],
                        "search_queries": ["RF open set robustness counterevidence"],
                    }
                ],
                "_execution": {"provider": "scripted", "model": "test-strong"},
            }
        if jt == "NOVELTY_SEARCH":
            return {
                "query": job.payload["query"],
                "counterevidence": [],
                "notes": "No counterevidence in scripted acceptance fixture.",
                "_execution": {"provider": "scripted", "model": "test-cheap"},
            }
        if jt == "NOVELTY_ARBITRATE":
            assert job.payload["_research_context"]["gap_candidate"]["id"] == job.payload["gap"]["id"]
            return {
                "status": "NEEDS_HUMAN_REVIEW",
                "reasoning": "The automated search is insufficient to establish novelty.",
                "surviving_statement": job.payload["gap"]["statement"],
                "search_count": len(job.payload["search_results"]),
                "_execution": {"provider": "scripted", "model": "test-strong"},
            }
        raise AssertionError(jt)


class ExplodingDelegate:
    def execute(self, job):
        raise AssertionError("cached materialization must prevent a second model call")


@pytest.mark.integration
def test_authoritative_semantic_cycle_materializes_scientific_state_and_replays(tmp_path):
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from research_engine.deterministic_executor import DeterministicResearchExecutor
    from research_engine.orchestration import (
        AgentWorker,
        ExecutorTier,
        JobType,
        OrchestratorStore,
        ResearchPlanner,
    )
    from research_engine.postgres import PostgresStore
    from research_engine.semantic_pipeline import (
        AuthoritativeSemanticExecutor,
        ResearchContextHydrator,
        SemanticResultMaterializer,
    )

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    queue = OrchestratorStore(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE human_review_requests, semantic_materializations,
                         research_job_attempts, research_jobs, research_workers,
                         research_runs, novelty_challenges, gap_candidates,
                         synthesis_artifacts, knowledge_checkpoints,
                         verification_events, evidence_relations, evidence_objects,
                         source_locators, extraction_runs, acquisition_attempts,
                         discovery_runs, external_identifiers, source_artifacts,
                         work_versions, works CASCADE
                """
            )
        conn.commit()

    run_id = "research-run:authoritative-agent-v1"
    queue.create_run(run_id=run_id, topic="dynamic open-set RF fingerprinting")
    planner = ResearchPlanner(queue)

    artifact_root = tmp_path / "artifacts"
    artifact_store = ContentAddressedArtifactStore(artifact_root)
    deterministic = DeterministicResearchExecutor(
        store=store,
        artifact_root=artifact_root,
        arxiv_provider=FakeArxivProvider(artifact_store),
    )
    hydrator = ResearchContextHydrator(
        store=store,
        artifact_root=artifact_root,
        text_extractor=ExactTextStub(),
    )
    materializer = SemanticResultMaterializer(store=store)
    semantic = ContextAwareSemantic()

    workers = [
        AgentWorker(
            worker_id="auth:cheap",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.CHEAP],
            executors={
                ExecutorTier.CHEAP: AuthoritativeSemanticExecutor(
                    delegate=semantic, hydrator=hydrator, materializer=materializer
                )
            },
        ),
        AgentWorker(
            worker_id="auth:standard",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STANDARD],
            executors={
                ExecutorTier.STANDARD: AuthoritativeSemanticExecutor(
                    delegate=semantic, hydrator=hydrator, materializer=materializer
                )
            },
        ),
        AgentWorker(
            worker_id="auth:strong",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.STRONG],
            executors={
                ExecutorTier.STRONG: AuthoritativeSemanticExecutor(
                    delegate=semantic, hydrator=hydrator, materializer=materializer
                )
            },
        ),
        AgentWorker(
            worker_id="auth:deterministic",
            queue=queue,
            planner=planner,
            supported_tiers=[ExecutorTier.DETERMINISTIC],
            executors={ExecutorTier.DETERMINISTIC: deterministic},
        ),
    ]

    for _ in range(50):
        for worker in workers:
            worker.run_until_idle()
        planner.reconcile(run_id)
        if queue.run_state(run_id)["status"] == "WAITING_HUMAN":
            break

    state = queue.run_state(run_id)
    assert state["status"] == "WAITING_HUMAN"
    assert state["knowledge_checkpoint_id"].startswith("knowledge-checkpoint:sha256:")

    with store.connect() as conn:
        with conn.cursor() as cur:
            checks = {
                "evidence": "SELECT count(*) FROM evidence_objects",
                "extractions": "SELECT count(*) FROM extraction_runs",
                "verification_events": "SELECT count(*) FROM verification_events",
                "syntheses": "SELECT count(*) FROM synthesis_artifacts",
                "gaps": "SELECT count(*) FROM gap_candidates",
                "challenges": "SELECT count(*) FROM novelty_challenges",
                "semantic_materializations": "SELECT count(*) FROM semantic_materializations WHERE status='COMPLETED'",
            }
            counts = {}
            for key, sql in checks.items():
                cur.execute(sql)
                counts[key] = cur.fetchone()[0]

            cur.execute(
                "SELECT verification_state FROM evidence_objects"
            )
            verification_state = cur.fetchone()[0]
            cur.execute(
                "SELECT novelty_status,human_review_state FROM gap_candidates"
            )
            gap_state = cur.fetchone()
            cur.execute(
                "SELECT result,knowledge_checkpoint_id FROM novelty_challenges"
            )
            challenge = cur.fetchone()

    assert counts["evidence"] == 1
    assert counts["extractions"] == 1
    assert counts["verification_events"] == 1
    assert counts["syntheses"] == 1
    assert counts["gaps"] == 1
    assert counts["challenges"] == 1
    assert counts["semantic_materializations"] >= 8
    assert verification_state == "MACHINE_CHECKED"
    assert gap_state == ("NEEDS_HUMAN_REVIEW", "REQUIRED")
    assert challenge[0] == "NEEDS_HUMAN_REVIEW"
    assert challenge[1] == state["knowledge_checkpoint_id"]

    extract_job = queue.jobs(run_id, JobType.EXTRACT)[0]
    replay = AuthoritativeSemanticExecutor(
        delegate=ExplodingDelegate(),
        hydrator=hydrator,
        materializer=materializer,
    ).execute(extract_job)
    assert replay["_materialization"]["evidence_ids"]
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM evidence_objects")
            assert cur.fetchone()[0] == 1
