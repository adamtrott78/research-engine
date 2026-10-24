import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def load(name):
    return json.loads((BASE/name).read_text())

def test_hardening_contract_preserves_authority_and_extract_isolation():
    contract=load("hardening-contract.v1.json")
    assert contract["status"]=="FROZEN_BEFORE_HARDENING_IMPLEMENTATION"
    assert "No hardening result authorizes production mutation or EXTRACT changes." in contract["invariants"]
    assert contract["final_verdict_values"]==["HARDENING_PASS","REVISE","REJECT"]

def test_historical_replay_cases_pin_old_decision_despite_divergent_current_state():
    cases=load("historical-replay-cases.v1.json")["replay_cases"]
    expected={c["case_id"]:c["replay_expected"] for c in cases}
    assert expected=={
        "R1_GPU_SMALL_HISTORICAL_ALLOW":"ALLOW",
        "R2_GPU_HEAVY_HISTORICAL_DENY":"DENY",
        "R3_MISSING_HISTORICAL_AUTHORITY_ABSTAINS":"ABSTAIN",
    }
    assert cases[0]["historical"]["policy"]["decision"]!=cases[0]["divergent_current_state"]["decision"]
    assert cases[1]["historical"]["policy"]["decision"]!=cases[1]["divergent_current_state"]["decision"]

def test_temporal_and_version_skew_expected_answers_are_fail_closed():
    data=load("temporal-version-skew-adversaries.v1.json")
    assert data["temporal_cases"][1]["global_truth_resolution"]=="NONE"
    assert data["temporal_cases"][2]["expected_semantics"]=="UNKNOWN"
    authority_sensitive=[
        c for c in data["version_skew_cases"]
        if "expected_decision" in c
    ]
    assert all(c["expected_decision"] in {"DENY","ABSTAIN"} for c in authority_sensitive)
    assert not any(c["expected_decision"]=="ALLOW" for c in authority_sensitive)

def test_slice_cases_are_bounded_and_cannot_widen_authority():
    data=load("ontology-evidence-cases.v1.json")
    assert all(c["max_observations"]<=16 for c in data["ontology_slice_cases"])
    assert all(c["authority_widening_allowed"] is False for c in data["ontology_slice_cases"])

def test_evidence_requirements_are_request_only_and_confidence_alone_does_nothing():
    cases=load("ontology-evidence-cases.v1.json")["evidence_requirement_cases"]
    active=[c for c in cases if "expected_execution_effect" in c]
    assert all(c["expected_execution_effect"]=="REQUEST_ONLY" for c in active)
    confidence=next(c for c in cases if c["case_id"]=="E7_MODEL_CONFIDENCE_ONLY")
    assert confidence["expected_requirements"]==[]

def test_scale_contract_is_deterministic_empirical_only():
    contract=load("scale-contract.v1.json")
    assert contract["generator"]["seed"]==20261023
    assert contract["generator"]["sizes"]==[1000,10000,100000]
    assert contract["performance_policy"]["pass_fail_latency_threshold"] is None
    assert contract["performance_policy"]["performance_promotion_claim_allowed"] is False

def test_extract_socket_is_opaque_and_defines_no_payload_schema():
    contract=load("opaque-extract-socket.v1.json")
    assert contract["state"]=="OPAQUE_UNBOUND"
    forbidden=set(contract["forbidden_descriptor_fields"])
    for descriptor in contract["synthetic_test_descriptors"]:
        assert descriptor["payload_schema_opaque"] is True
        assert descriptor["adapter_target_contract"]=="SemanticObservationV1"
        assert forbidden.isdisjoint(descriptor.keys())
    unaccepted=next(x for x in contract["synthetic_test_descriptors"] if x["descriptor_id"]=="synthetic:unaccepted")
    accepted=next(x for x in contract["synthetic_test_descriptors"] if x["descriptor_id"]=="synthetic:accepted")
    assert unaccepted["expected_registration"]=="REJECT_BINDING"
    assert accepted["expected_registration"]=="REGISTER_DESCRIPTOR_ONLY"
