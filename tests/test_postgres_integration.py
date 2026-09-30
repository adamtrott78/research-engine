import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_postgres_schema_and_support_query():
    dsn=os.getenv('RESEARCH_ENGINE_DATABASE_URL')
    if not dsn:
        pytest.skip('RESEARCH_ENGINE_DATABASE_URL not set')
    pytest.importorskip('psycopg')
    from research_engine.postgres import PostgresStore

    store=PostgresStore(dsn)
    store.initialize(Path(__file__).parents[1]/'sql'/'schema.sql')
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE evidence_relations, evidence_objects, source_locators, extraction_runs, source_artifacts, work_versions, works CASCADE")
            cur.execute("INSERT INTO works(id,title) VALUES ('w','paper')")
            cur.execute("INSERT INTO work_versions(id,work_id) VALUES ('wv','w')")
            cur.execute("INSERT INTO source_artifacts(id,work_version_id,sha256,byte_size,media_type,format,access_class) VALUES ('a','wv',%s,10,'application/pdf','pdf','OPEN_ACCESS')", ('a'*64,))
            cur.execute("INSERT INTO source_locators(id,source_artifact_id,source_level,locator_quality,page_number,pdf_bbox) VALUES ('l','a','FULLTEXT_PDF_LAYOUT','EXACT',1,ARRAY[1.0,2.0,3.0,4.0])")
            cur.execute("INSERT INTO extraction_runs(id,work_version_id,source_artifact_id,source_level,extractor_identity) VALUES ('x','wv','a','FULLTEXT_PDF_LAYOUT','test')")
            cur.execute("INSERT INTO evidence_objects(id,evidence_type,work_id,work_version_id,source_artifact_id,source_locator_id,source_level,normalized_form,extraction_run_id,verification_state) VALUES ('result','RESULT','w','wv','a','l','FULLTEXT_PDF_LAYOUT','result text','x','MACHINE_CHECKED')")
            cur.execute("INSERT INTO evidence_objects(id,evidence_type,work_id,work_version_id,source_artifact_id,source_locator_id,source_level,normalized_form,extraction_run_id,verification_state) VALUES ('claim','CLAIM','w','wv','a','l','FULLTEXT_PDF_LAYOUT','claim text','x','MACHINE_CHECKED')")
            cur.execute("INSERT INTO evidence_relations(id,source_id,target_id,relation_type,verification_state,rationale) VALUES ('r','result','claim','SUPPORTS','MACHINE_CHECKED','test support')")
        conn.commit()
    rows=store.claim_support('claim')
    assert rows[0]['result_id']=='result'
    assert rows[0]['source_sha256']=='a'*64
    assert rows[0]['page_number']==1
