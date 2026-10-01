import json
import os
from pathlib import Path

import pytest


CORPUS = Path(__file__).parent / "data" / "rf_osr_five_work.json"
NOVELTY = Path(__file__).parent / "data" / "novelty_pilot.json"


@pytest.mark.integration
def test_novelty_challenges_are_checkpointed_adversarial_and_idempotent():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.corpus_loader import ImmutableRecordConflict, PostgresCorpusLoader
    from research_engine.novelty import NoveltyChallengeService
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    loader = PostgresCorpusLoader(store)
    novelty = NoveltyChallengeService(store)
    corpus = json.loads(CORPUS.read_text())
    pilot = json.loads(NOVELTY.read_text())

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

    loader.load(corpus)
    checkpoint = store.create_knowledge_checkpoint()

    for candidate in pilot["candidates"]:
        novelty.put_gap_candidate(
            candidate_id=candidate["id"],
            statement=candidate["statement"],
            gap_type=candidate["gap_type"],
            knowledge_checkpoint_id=checkpoint.id,
            motivating_evidence_ids=candidate["motivating_evidence_ids"],
            rationale=candidate["rationale"],
        )
        # Exact replay is idempotent.
        novelty.put_gap_candidate(
            candidate_id=candidate["id"],
            statement=candidate["statement"],
            gap_type=candidate["gap_type"],
            knowledge_checkpoint_id=checkpoint.id,
            motivating_evidence_ids=candidate["motivating_evidence_ids"],
            rationale=candidate["rationale"],
        )

    # Scientific identity under the same candidate id cannot silently change.
    changed = dict(pilot["candidates"][0])
    changed["statement"] += " silently changed"
    with pytest.raises(ImmutableRecordConflict):
        novelty.put_gap_candidate(
            candidate_id=changed["id"],
            statement=changed["statement"],
            gap_type=changed["gap_type"],
            knowledge_checkpoint_id=checkpoint.id,
            motivating_evidence_ids=changed["motivating_evidence_ids"],
            rationale=changed["rationale"],
        )

    outcomes = {}
    for challenge in pilot["challenges"]:
        outcome = novelty.record_challenge(
            challenge_id=challenge["id"],
            gap_candidate_id=challenge["gap_candidate_id"],
            knowledge_checkpoint_id=checkpoint.id,
            proposed_result=challenge["proposed_result"],
            search_provenance=challenge["search_provenance"],
            counterevidence=challenge["counterevidence"],
            reasoning=challenge["reasoning"],
        )
        outcomes[outcome.gap_candidate_id] = outcome.result
        # Exact challenge replay is idempotent.
        replay = novelty.record_challenge(
            challenge_id=challenge["id"],
            gap_candidate_id=challenge["gap_candidate_id"],
            knowledge_checkpoint_id=checkpoint.id,
            proposed_result=challenge["proposed_result"],
            search_provenance=challenge["search_provenance"],
            counterevidence=challenge["counterevidence"],
            reasoning=challenge["reasoning"],
        )
        assert replay == outcome

    assert outcomes == {
        "gap:rf-osr:dynamic-environment-generalization:v1": "REJECTED_PRIOR_WORK_FOUND",
        "gap:rf-osr:efficient-blind-generative-outliers:v1": "NARROWED",
    }

    empty_search = novelty.record_challenge(
        challenge_id="challenge:efficient-blind-generative:empty-search-regression",
        gap_candidate_id="gap:rf-osr:efficient-blind-generative-outliers:v1",
        knowledge_checkpoint_id=checkpoint.id,
        proposed_result="NO_COUNTEREVIDENCE_FOUND",
        search_provenance={
            "provider": "test",
            "queries": ["extremely narrow synonym search"],
            "provider_run_ids": ["test-run-empty"],
        },
        counterevidence=[],
        reasoning="An empty automated search is not proof of novelty.",
    )
    assert empty_search.result == "NEEDS_HUMAN_REVIEW"

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, novelty_status, human_review_state, knowledge_checkpoint_id
                FROM gap_candidates ORDER BY id
                """
            )
            gaps = cur.fetchall()
            cur.execute(
                """
                SELECT id, result, knowledge_checkpoint_id
                FROM novelty_challenges ORDER BY id
                """
            )
            challenges = cur.fetchall()

    assert len(gaps) == 2
    assert len(challenges) == 3
    assert all(row[3] == checkpoint.id for row in gaps)
    assert all(row[2] == checkpoint.id for row in challenges)
    assert any(row[1] == "REJECTED_PRIOR_WORK_FOUND" for row in gaps)
    assert any(row[1] == "NEEDS_HUMAN_REVIEW" for row in gaps)
    assert not any(row[1] == "SUPPORTED_AS_OPEN" for row in gaps)
