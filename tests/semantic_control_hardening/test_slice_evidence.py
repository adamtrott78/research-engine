import json
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def data():
    return json.loads((BASE/"ontology-evidence-cases.v1.json").read_text())

def make_observation(observation_id,subject_type,predicate,object_type=None):
    obj={"kind":"LITERAL","value":observation_id}
    if object_type:
        obj={"kind":"ENTITY","entity_type":object_type,"entity_id":f"{object_type}:{observation_id}"}
    return {
        "schema_version":1,
        "observation_id":observation_id,
        "layer":"EPISTEMIC",
        "subject":{"entity_type":subject_type,"entity_id":f"{subject_type}:{observation_id}"},
        "predicate":predicate,
        "object":obj,
        "assertion_state":"ACTIVE",
        "provenance":[{"source_id":f"source:{observation_id}"}],
        "verification":{"state":"MACHINE_CHECKED","basis":["synthetic hardening slice"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":{},
    }

def pool_for(case):
    observations=[]
    i=0
    for entity_type in case["required_entity_types"]:
        predicate=case["required_predicates"][i % len(case["required_predicates"])]
        observations.append(make_observation(f"required:{i}",entity_type,predicate))
        i+=1
    for predicate in case["required_predicates"]:
        observations.append(make_observation(f"predicate:{predicate}","Auxiliary",predicate))
    for j in range(30):
        observations.append(make_observation(f"distractor:{j}","Irrelevant","notRelevant"))
    for forbidden in case.get("forbidden_entity_types",[]):
        observations.append(make_observation(f"forbidden:{forbidden}",forbidden,case["required_predicates"][0]))
    return observations

def test_all_frozen_ontology_slices_are_bounded_deterministic_and_relevant():
    from prototypes.semantic_control_hardening.slice_evidence import select_bounded_ontology_slice
    for case in data()["ontology_slice_cases"]:
        pool=pool_for(case)
        first=select_bounded_ontology_slice(observations=pool,case=case,semantic_manifest_id="manifest:test")
        second=select_bounded_ontology_slice(observations=list(reversed(pool)),case=case,semantic_manifest_id="manifest:test")
        assert first==second
        assert len(first["observations"])<=case["max_observations"]
        assert set(case["required_entity_types"]).issubset(first["included_entity_types"])
        assert set(case["required_predicates"]).issubset(first["included_predicates"])
        assert first["authority_widening_allowed"] is False
        assert all(x not in first["included_entity_types"] for x in case.get("forbidden_entity_types",[]))
        assert len(first["source_schema_refs"])==len(first["observations"])

def test_frozen_e1_e6_generate_request_only_expected_reason_types():
    from prototypes.semantic_control_hardening.slice_evidence import derive_frozen_evidence_case
    cases=data()["evidence_requirement_cases"]
    for case in cases[:6]:
        requirement=derive_frozen_evidence_case(case)
        assert requirement is not None
        assert requirement["reason_types"]==case["expected_reason_types"]
        assert requirement["execution_effect"]=="REQUEST_ONLY"
        assert requirement["status"]=="OPEN"
        assert requirement["provenance"][0]["source_kind"]=="SYNTHETIC_HARDENING_FIXTURE"

def test_model_confidence_alone_generates_no_requirement():
    from prototypes.semantic_control_hardening.slice_evidence import derive_frozen_evidence_case
    case=next(x for x in data()["evidence_requirement_cases"] if x["case_id"]=="E7_MODEL_CONFIDENCE_ONLY")
    assert derive_frozen_evidence_case(case) is None

def test_combined_durable_reasons_are_deterministic_and_deduplicated():
    from prototypes.semantic_control_hardening.slice_evidence import derive_evidence_requirement
    trigger={
        "contradictions":["a","b"],
        "missing_facets":["evaluation"],
        "source_level":"ABSTRACT",
        "required_source_level":"FULLTEXT",
        "verification_state":"UNVERIFIED",
        "valid_to":"2026-01-01T00:00:00Z",
        "as_of":"2026-10-23T00:00:00Z",
        "benchmark_sufficient":False,
        "model_confidence":0.01,
    }
    first=derive_evidence_requirement(target_id="target",trigger=trigger,provenance=["source"])
    second=derive_evidence_requirement(target_id="target",trigger=deepcopy(trigger),provenance=["source"])
    assert first==second
    assert first["reason_types"]==[
        "BENCHMARK_CALIBRATED_INSUFFICIENCY",
        "CONTRADICTION",
        "MISSING_FACET",
        "SOURCE_LEVEL_INSUFFICIENT",
        "STALE",
        "VERIFICATION_INSUFFICIENT",
    ]
    assert first["execution_effect"]=="REQUEST_ONLY"

def test_slice_evidence_module_has_no_execution_or_external_runtime_paths():
    text=(ROOT/"prototypes"/"semantic_control_hardening"/"slice_evidence.py").read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "executor","lambda","gpu","acquisition","materialization",
        "insert into","update ","delete from","truncate ",
    ]:
        assert token not in text
