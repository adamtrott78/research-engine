import json
import os
from pathlib import Path

import pytest


BASE = Path(__file__).parent / "data" / "rf_osr_five_work.json"
DELTA = Path(__file__).parent / "data" / "rf_osr_counterevidence_delta.json"
PILOT = Path(__file__).parent / "data" / "novelty_pilot.json"
RECHALLENGE = Path(__file__).parent / "data" / "novelty_rechallenge_corpus.json"


@pytest.mark.integration
def test_counterevidence_promotion_changes_checkpoint_and_preserves_gap_outcomes():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.corpus_loader import PostgresCorpusLoader
    from research_engine.novelty import NoveltyChallengeService
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    loader = PostgresCorpusLoader(store)
    novelty = NoveltyChallengeService(store)

    base = json.loads(BASE.read_text())
    delta = json.loads(DELTA.read_text())
    pilot = json.loads(PILOT.read_text())
    rechallenge = json.loads(RECHALLENGE.read_text())

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE novelty_challenges, gap_candidates, knowledge_checkpoints,
                         verification_events, evidence_relations, evidence_objects,
                         source_locators, extraction_runs, source_artifacts,
                         work_versions, works
                CASCADE
                """
            )
        conn.commit()

    loader.load(base)
    cp5 = store.create_knowledge_checkpoint()

    for candidate in pilot["candidates"]:
        novelty.put_gap_candidate(
            candidate_id=candidate["id"],
            statement=candidate["statement"],
            gap_type=candidate["gap_type"],
            knowledge_checkpoint_id=cp5.id,
            motivating_evidence_ids=candidate["motivating_evidence_ids"],
            rationale=candidate["rationale"],
        )

    for challenge in pilot["challenges"]:
        novelty.record_challenge(
            challenge_id=challenge["id"],
            gap_candidate_id=challenge["gap_candidate_id"],
            knowledge_checkpoint_id=cp5.id,
            proposed_result=challenge["proposed_result"],
            search_provenance=challenge["search_provenance"],
            counterevidence=challenge["counterevidence"],
            reasoning=challenge["reasoning"],
        )

    loader.load(delta)
    loader.load(delta)
    cp7 = store.create_knowledge_checkpoint()

    assert cp7.id != cp5.id
    assert cp7.record_counts["works"] == 7
    assert cp7.record_counts["evidence_objects"] == 29
    assert cp7.record_counts["evidence_relations"] == 16

    outcomes = {}
    for challenge in rechallenge["challenges"]:
        outcome = novelty.record_challenge(
            challenge_id=challenge["id"],
            gap_candidate_id=challenge["gap_candidate_id"],
            knowledge_checkpoint_id=cp7.id,
            proposed_result=challenge["proposed_result"],
            search_provenance=challenge["search_provenance"],
            counterevidence=challenge["counterevidence"],
            reasoning=challenge["reasoning"],
        )
        outcomes[outcome.gap_candidate_id] = outcome.result

    assert outcomes == {
        "gap:rf-osr:dynamic-environment-generalization:v1": "REJECTED_PRIOR_WORK_FOUND",
        "gap:rf-osr:efficient-blind-generative-outliers:v1": "NARROWED",
    }

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_level, count(*)::int
                FROM evidence_objects
                GROUP BY source_level
                ORDER BY source_level
                """
            )
            counts = dict(cur.fetchall())

            cur.execute(
                """
                SELECT id, novelty_status
                FROM gap_candidates
                ORDER BY id
                """
            )
            gaps = dict(cur.fetchall())

            cur.execute(
                """
                SELECT gap_candidate_id, count(*)::int
                FROM novelty_challenges
                GROUP BY gap_candidate_id
                ORDER BY gap_candidate_id
                """
            )
            history_counts = dict(cur.fetchall())

    assert counts["FULLTEXT_STRUCTURED"] == 8
    assert counts["FULLTEXT_PDF_LAYOUT"] == 18
    assert counts["OCR_DERIVED"] == 3
    assert gaps == outcomes
    assert history_counts == {
        "gap:rf-osr:dynamic-environment-generalization:v1": 2,
        "gap:rf-osr:efficient-blind-generative-outliers:v1": 2,
    }
    assert "SUPPORTED_AS_OPEN" not in gaps.values()
