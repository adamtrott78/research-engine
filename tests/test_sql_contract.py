from pathlib import Path


def test_postgres_schema_preserves_authority_projection_boundaries():
    sql=(Path(__file__).parents[1]/'sql'/'schema.sql').read_text()
    assert 'CREATE EXTENSION IF NOT EXISTS vector' in sql
    for table in ['works','work_versions','source_artifacts','source_locators','evidence_objects','evidence_relations','verification_events','embeddings']:
        assert f'CREATE TABLE IF NOT EXISTS {table}' in sql
    assert 'VECTOR NOT NULL' in sql
    assert 'HNSW' in sql and 'Intentionally no HNSW/IVFFlat index' in sql
    assert "relation_not_self" in sql
