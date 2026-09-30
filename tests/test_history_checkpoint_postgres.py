import copy
import json
import os
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "data" / "rf_osr_five_work.json"
WORK_IDS = [
    "work:arxiv:2108.13099",
    "work:scheirer:2013:osr",
    "work:liu:2020:energy-ood",
    "work:baye:2024:varmax",
    "work:broggi:2025:varmax",
]


@pytest.mark.integration
def test_immutable_evidence_supersession_history_and_knowledge_checkpoints():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.corpus_loader import ImmutableRecordConflict, PostgresCorpusLoader
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    loader = PostgresCorpusLoader(store)
    data = json.loads(FIXTURE.read_text())

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE knowledge_checkpoints, verification_events,
                         evidence_relations, evidence_objects, source_locators,
                         extraction_runs, source_artifacts, work_versions, works
                CASCADE
                """
            )
        conn.commit()

    loader.load(data)
    cp1 = store.create_knowledge_checkpoint(WORK_IDS)
    cp2 = store.create_knowledge_checkpoint(list(reversed(WORK_IDS)))
    assert cp1.id == cp2.id
    assert cp1.fingerprint == cp2.fingerprint
    assert cp1.record_counts["works"] == 5
    assert cp1.record_counts["evidence_objects"] == 21
    assert cp1.record_counts["evidence_relations"] == 11

    changed = copy.deepcopy(data)
    changed["evidence_objects"][0]["normalized_form"] += " silently rewritten"
    with pytest.raises(ImmutableRecordConflict, match="different authoritative content"):
        loader.load(changed)

    changed_relation = copy.deepcopy(data)
    changed_relation["evidence_relations"][0]["rationale"] = "silently rewritten relation"
    with pytest.raises(ImmutableRecordConflict, match="relation id"):
        loader.load(changed_relation)

    superseding = copy.deepcopy(data)
    old = next(x for x in superseding["evidence_objects"] if x["id"] == "ev:2108:limitation:ellipsoid")
    new = copy.deepcopy(old)
    new["id"] = "ev:2108:limitation:ellipsoid:v2"
    new["supersedes_id"] = old["id"]
    new["normalized_form"] = (
        "The ellipsoidal blind-generation method requires empirical tuning of delta; "
        "this superseding record preserves an explicit revised normalization."
    )
    superseding["evidence_objects"].append(new)
    loader.load(superseding)

    store.record_verification_event(
        event_id="verify:supersede:ev-2108-ellipsoid:v1",
        record_type="evidence_object",
        record_id=old["id"],
        new_state="SUPERSEDED",
        verifier_type="RULE",
        verifier_identity="history-regression-test",
        rationale="superseded by ev:2108:limitation:ellipsoid:v2",
        evidence=[new["id"]],
    )
    # Replay must be idempotent and must not append another event.
    store.record_verification_event(
        event_id="verify:supersede:ev-2108-ellipsoid:v1",
        record_type="evidence_object",
        record_id=old["id"],
        new_state="SUPERSEDED",
        verifier_type="RULE",
        verifier_identity="history-regression-test",
        rationale="superseded by ev:2108:limitation:ellipsoid:v2",
        evidence=[new["id"]],
    )

    cp3 = store.create_knowledge_checkpoint(WORK_IDS)
    assert cp3.id != cp1.id
    assert cp3.record_counts["evidence_objects"] == 22

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, verification_state, supersedes_id
                FROM evidence_objects
                WHERE id IN (%s, %s)
                ORDER BY id
                """,
                (old["id"], new["id"]),
            )
            rows = cur.fetchall()
            cur.execute("SELECT count(*) FROM verification_events WHERE id = %s", ("verify:supersede:ev-2108-ellipsoid:v1",))
            event_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM knowledge_checkpoints")
            checkpoint_count = cur.fetchone()[0]

    assert rows == [
        ("ev:2108:limitation:ellipsoid", "SUPERSEDED", None),
        ("ev:2108:limitation:ellipsoid:v2", "MACHINE_CHECKED", "ev:2108:limitation:ellipsoid"),
    ]
    assert event_count == 1
    assert checkpoint_count == 2
