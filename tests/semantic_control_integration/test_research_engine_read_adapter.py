import json
import os
from pathlib import Path
import time
import warnings

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_integration"
RUNTIME_COMMIT="b09df52942fc0b7b49ee7f82a6f8242df0924fba"
BASELINE_CONTRACT_BLOB="1c97ab527db5d375ff7bb068719330a4acc0458c"

def load(name):
    return json.loads((BASE/name).read_text())

def setup_store():
    dsn=os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    from research_engine.corpus_loader import PostgresCorpusLoader
    from research_engine.postgres import PostgresStore
    store=PostgresStore(dsn)
    store.initialize(ROOT/"sql"/"schema.sql")
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                TRUNCATE knowledge_checkpoints, verification_events,
                         evidence_relations, evidence_objects, source_locators,
                         extraction_runs, acquisition_attempts, discovery_runs,
                         external_identifiers, source_artifacts, work_versions, works
                CASCADE
            """)
        conn.commit()
    fixture=json.loads((ROOT/"tests"/"data"/"rf_osr_five_work.json").read_text())
    PostgresCorpusLoader(store).load(fixture)
    return store

def make_adapter(store):
    from prototypes.semantic_control_integration.research_engine_adapter import ReadOnlyResearchEngineAdapter
    return ReadOnlyResearchEngineAdapter(
        store=store,
        baseline_contract=load("relational-baseline-contract.v1.json"),
        baseline_result=load("relational-baseline-result.v1.json"),
        runtime_repository_commit=RUNTIME_COMMIT,
        baseline_contract_blob_sha=BASELINE_CONTRACT_BLOB,
    )

@pytest.mark.integration
def test_current_store_adapter_matches_frozen_relational_answers_exactly():
    manifest=make_adapter(setup_store()).build_manifest()
    baseline=load("relational-baseline-result.v1.json")
    expected={item["query_id"]:item["canonical_result_sha256"] for item in baseline["query_results"]}
    actual={item["query_id"]:item["relational_result_sha256"] for item in manifest["query_records"]}
    reconstructed={item["query_id"]:item["semantic_reconstruction_sha256"] for item in manifest["query_records"]}
    assert actual==expected
    assert reconstructed==expected
    assert len(manifest["observations"])==6

@pytest.mark.integration
def test_adapter_manifest_is_deterministic_across_repeated_reads():
    adapter=make_adapter(setup_store())
    first=adapter.build_manifest()
    second=adapter.build_manifest()
    assert first==second
    assert len(first["semantic_manifest_sha256"])==64

@pytest.mark.integration
def test_adapter_preserves_exact_source_and_query_provenance():
    manifest=make_adapter(setup_store()).build_manifest()
    for observation in manifest["observations"]:
        assert observation["layer"]=="EPISTEMIC"
        assert observation["verification"]["state"]=="AUTHORITATIVE_RECORD"
        by_kind={item["source_kind"]:item for item in observation["provenance"]}
        assert by_kind["RUNTIME_REPOSITORY_CHECKPOINT"]["commit_sha"]==RUNTIME_COMMIT
        assert by_kind["STORE_INTERFACE"]["git_blob_sha"]=="dda83be0e41773d7ff45e20fa85a04e76f681008"
        assert by_kind["QUERY_TEMPLATE"]["git_blob_sha"]=="4daf119db5058766dc01a7c649c9fa41c8748cd7"
        assert by_kind["FIXTURE"]["git_blob_sha"]=="742feb261a72a760c7cb7a472beaca0cb283e4f8"
        assert by_kind["BASELINE_QUERY_CONTRACT"]["git_blob_sha"]==BASELINE_CONTRACT_BLOB
        assert by_kind["QUERY_TEMPLATE"]["query_id"]==observation["attributes"]["query_id"]

@pytest.mark.integration
def test_measure_pure_adapter_projection_overhead():
    store=setup_store()
    contract=load("relational-baseline-contract.v1.json")
    baseline=load("relational-baseline-result.v1.json")
    from prototypes.semantic_control_integration.research_engine_adapter import canonical_sha256,project_query_result,reconstruct_relational_answer
    raw_by_query={}
    for query in contract["queries"]:
        raw_by_query[query["query_id"]]=getattr(store,query["store_method"])(**query["arguments"])
    expected_hashes={item["query_id"]:item["canonical_result_sha256"] for item in baseline["query_results"]}
    results=[]
    for query in contract["queries"]:
        raw=raw_by_query[query["query_id"]]
        assert canonical_sha256(raw)==expected_hashes[query["query_id"]]
        def transform():
            observations=project_query_result(
                query=query,
                result=raw,
                runtime_repository_commit=RUNTIME_COMMIT,
                baseline_contract=contract,
                baseline_contract_blob_sha=BASELINE_CONTRACT_BLOB,
            )
            reconstructed=reconstruct_relational_answer(query["query_id"],observations)
            assert canonical_sha256(reconstructed)==expected_hashes[query["query_id"]]
        for _ in range(10):
            transform()
        samples=[]
        for _ in range(500):
            started=time.perf_counter_ns()
            transform()
            samples.append(time.perf_counter_ns()-started)
        ordered=sorted(samples)
        results.append({
            "query_id":query["query_id"],
            "warmup_repetitions":10,
            "measured_repetitions":500,
            "latency_ns":{
                "min":min(samples),
                "mean":round(sum(samples)/len(samples)),
                "p50":ordered[249],
                "p95":ordered[474],
                "max":max(samples),
            },
        })
    report={
        "schema_version":1,
        "measurement_id":"semantic-control-re-adapter-overhead-v1",
        "runtime_repository_commit":RUNTIME_COMMIT,
        "timing_scope":"pure relational-result to semantic-observation projection plus exact reconstruction validation; excludes database read",
        "query_results":results,
        "performance_promotion_claim":False,
        "latency_threshold_applied":False,
        "production_database_access":False,
        "production_mutation":False,
    }
    warnings.warn("SEMANTIC_CONTROL_RE_ADAPTER_OVERHEAD_V1="+json.dumps(report,sort_keys=True,separators=(",",":")),UserWarning,stacklevel=1)

def test_adapter_source_contains_no_mutating_store_or_external_runtime_paths():
    lowered=(ROOT/"prototypes"/"semantic_control_integration"/"research_engine_adapter.py").read_text().lower()
    for token in [
        ".connect(",
        ".initialize(",
        ".put_work(",
        ".record_verification_event(",
        ".create_knowledge_checkpoint(",
        "insert into",
        "update ",
        "delete from",
        "truncate ",
        "import requests",
        "from requests",
        "import openai",
        "from openai",
        "import subprocess",
        "from subprocess",
        "lambda",
        "gpu",
    ]:
        assert token not in lowered
