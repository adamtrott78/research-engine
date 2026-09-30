import json
import os
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "data" / "rf_osr_five_work.json"


@pytest.mark.integration
def test_five_work_corpus_load_is_idempotent_and_queryable():
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    pytest.importorskip("psycopg")

    from research_engine.corpus_loader import PostgresCorpusLoader
    from research_engine.postgres import PostgresStore

    store = PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1] / "sql" / "schema.sql")
    data = json.loads(FIXTURE.read_text())
    loader = PostgresCorpusLoader(store)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE evidence_relations, evidence_objects, source_locators,
                         extraction_runs, source_artifacts, work_versions, works
                CASCADE
                """
            )
        conn.commit()

    loader.load(data)
    loader.load(data)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM works")
            assert cur.fetchone()[0] == 5
            cur.execute("SELECT count(*) FROM evidence_objects")
            assert cur.fetchone()[0] == 21
            cur.execute("SELECT count(*) FROM evidence_relations")
            assert cur.fetchone()[0] == 11

    assert store.source_level_counts() == {
        "FULLTEXT_PDF_LAYOUT": 18,
        "OCR_DERIVED": 3,
    }

    support = store.claim_support("ev:2108:claim:conclusion")
    assert [row["result_id"] for row in support] == [
        "ev:2108:result:ellipsoid",
        "ev:2108:result:optimization",
    ]
    assert {row["source_sha256"] for row in support} == {
        "7fcfc9929fd80a94095657ebf644be3301000d0a4342b74de1adc2af529ff7ab"
    }
    assert {row["page_number"] for row in support} == {5, 6}

    qualifiers = store.qualifiers_for_method("ev:2108:method:ellipsoid")
    assert [row["qualifier_id"] for row in qualifiers] == [
        "ev:2108:limitation:ellipsoid"
    ]
    assert qualifiers[0]["source_sha256"] == (
        "7fcfc9929fd80a94095657ebf644be3301000d0a4342b74de1adc2af529ff7ab"
    )

    path = store.provenance_path(
        "ev:2108:result:ellipsoid",
        "ev:2108:claim:conclusion",
    )
    assert path
    assert path[0]["relation_type"] == "SUPPORTS"

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relation_type, verification_state
                FROM evidence_relations
                WHERE id IN ('rel:baye-liu', 'rel:broggi-liu', 'rel:broggi-baye')
                ORDER BY id
                """
            )
            rows = cur.fetchall()
    assert len(rows) == 3
    assert {row[0] for row in rows} == {
        "USES",
        "COMPARES_AGAINST",
        "SAME_METHOD_LINEAGE",
    }
