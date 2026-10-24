import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def historical_case():
    return json.loads((BASE/"historical-replay-cases.v1.json").read_text())["replay_cases"][0]

def make_candidate_and_evaluation():
    from prototypes.semantic_control.decision import make_candidate_action,evaluate_candidate_action
    case=historical_case()
    historical=case["historical"]
    policy=historical["policy"]
    candidate=make_candidate_action(
        candidate_action_id=historical["candidate"]["candidate_action_id"],
        action_type=historical["candidate"]["action_type"],
        target={"workload_class":historical["candidate"]["workload_class"]},
        rationale_observation_ids=["historical-rationale:R1"],
        preconditions=[{"precondition_id":"p:scope","description":"Frozen historical precondition."}],
        required_capabilities=["gpu-small"],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["historical pinned policy context"],
        provenance=["historical-replay-fixture"],
    )
    authority={
        "schema_version":1,
        "observation_id":"historical-policy:R1",
        "layer":"NORMATIVE",
        "subject":{"entity_type":"WorkloadClass","entity_id":"gpu-small"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":policy["decision"]},
        "assertion_state":"ACTIVE",
        "provenance":[{"source_id":"wc-workload-policy"}],
        "verification":{"state":"AUTHORITATIVE_RECORD","basis":["frozen historical audit fixture"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":{"policy_id":policy["policy_id"],"policy_epoch":policy["policy_epoch"]},
    }
    evaluation=evaluate_candidate_action(
        candidate=candidate,
        observations=[authority],
        semantic_manifest_id=historical["semantic_manifest_id"],
        knowledge_checkpoint_ids=historical["knowledge_checkpoint_ids"],
        precondition_results=[{"precondition_id":"p:scope","result":"PASS","evidence_ids":["evidence:scope-pass"]}],
        proposed_executor="historical-replay-executor",
    )
    return candidate,evaluation

def test_audit_chain_is_deterministic_and_binds_exact_decision_inputs():
    from prototypes.semantic_control_hardening.audit_chain import build_decision_audit_chain
    candidate,evaluation=make_candidate_and_evaluation()
    first=build_decision_audit_chain(candidate=candidate,evaluation=evaluation)
    second=build_decision_audit_chain(candidate=candidate,evaluation=evaluation)
    assert first==second
    assert len(first["audit_chain_sha256"])==64
    assert first["decision"]=="ALLOW"
    assert first["authority_source_ids"]==["wc-workload-policy"]
    assert first["precondition_evidence_ids"]==["evidence:scope-pass"]
    assert first["receipt_binding"]=={
        "status":"ABSENT","receipt_id":None,"receipt_sha256":None,"authority_effect":"NONE"
    }
    assert first["original_execution_receipt_id"] is None

def test_valid_later_receipt_is_auditable_but_never_authority():
    from prototypes.semantic_control_hardening.audit_chain import build_decision_audit_chain
    candidate,evaluation=make_candidate_and_evaluation()
    receipt={
        "receipt_id":"receipt:1",
        "evaluation_id":evaluation["evaluation_id"],
        "candidate_action_id":candidate["candidate_action_id"],
        "outcome":"SUCCESS",
    }
    chain=build_decision_audit_chain(candidate=candidate,evaluation=evaluation,receipt=receipt)
    assert chain["receipt_binding"]["status"]=="VALID_BINDING"
    assert chain["receipt_binding"]["receipt_id"]=="receipt:1"
    assert chain["receipt_binding"]["authority_effect"]=="NONE"
    assert chain["decision"]=="ALLOW"

def test_mismatched_receipt_is_labeled_invalid_without_rewriting_decision():
    from prototypes.semantic_control_hardening.audit_chain import build_decision_audit_chain
    candidate,evaluation=make_candidate_and_evaluation()
    receipt={
        "receipt_id":"receipt:bad",
        "evaluation_id":"eval:different",
        "candidate_action_id":candidate["candidate_action_id"],
        "outcome":"SUCCESS",
    }
    chain=build_decision_audit_chain(candidate=candidate,evaluation=evaluation,receipt=receipt)
    assert chain["receipt_binding"]["status"]=="INVALID_BINDING"
    assert chain["receipt_binding"]["authority_effect"]=="NONE"
    assert chain["decision"]=="ALLOW"

def test_audit_chain_verification_detects_tampering():
    from prototypes.semantic_control_hardening.audit_chain import build_decision_audit_chain,verify_decision_audit_chain
    candidate,evaluation=make_candidate_and_evaluation()
    chain=build_decision_audit_chain(candidate=candidate,evaluation=evaluation)
    changed=dict(chain)
    changed["decision"]="DENY"
    result=verify_decision_audit_chain(chain=changed,candidate=candidate,evaluation=evaluation)
    assert result["valid"] is False
    assert result["receipt_authority_effect"]=="NONE"

def test_audit_chain_matches_historical_replay_decision():
    from prototypes.semantic_control_hardening.audit_chain import build_decision_audit_chain
    from prototypes.semantic_control_hardening.replay_temporal import replay_historical_decision
    candidate,evaluation=make_candidate_and_evaluation()
    replay=replay_historical_decision(historical_case())
    chain=build_decision_audit_chain(candidate=candidate,evaluation=evaluation)
    assert replay["decision"]==chain["decision"]=="ALLOW"
    assert replay["knowledge_checkpoint_ids"]==chain["knowledge_checkpoint_ids"]

def test_audit_chain_source_has_no_execution_or_external_runtime_paths():
    text=(ROOT/"prototypes"/"semantic_control_hardening"/"audit_chain.py").read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "insert into","update ","delete from","truncate ",
        "execute(","send(","dispatch(",
    ]:
        assert token not in text
