from pathlib import Path


def test_postgres_schema_preserves_authority_projection_boundaries():
    sql=(Path(__file__).parents[1]/'sql'/'schema.sql').read_text()
    assert 'CREATE EXTENSION IF NOT EXISTS vector' in sql
    for table in ['works','work_versions','source_artifacts','source_locators','evidence_objects','evidence_relations','verification_events','knowledge_checkpoints','gap_candidates','novelty_challenges','embeddings']:
        assert f'CREATE TABLE IF NOT EXISTS {table}' in sql
    assert 'VECTOR NOT NULL' in sql
    assert 'HNSW' in sql and 'Intentionally no HNSW/IVFFlat index' in sql
    assert "relation_not_self" in sql

    assert "supersedes_id TEXT REFERENCES evidence_relations(id)" in sql
    assert "CREATE TABLE IF NOT EXISTS knowledge_checkpoints" in sql

    assert "knowledge_checkpoint_id TEXT NOT NULL REFERENCES knowledge_checkpoints(id)" in sql
    assert "human_review_state TEXT NOT NULL DEFAULT 'NOT_REVIEWED'" in sql
    assert "human_approval JSONB" in sql
