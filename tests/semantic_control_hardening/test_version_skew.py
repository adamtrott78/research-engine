import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def cases():
    return json.loads((BASE/"temporal-version-skew-adversaries.v1.json").read_text())["version_skew_cases"]

def test_all_frozen_v1_v8_adversaries_match_expected_fail_closed_answers():
    from prototypes.semantic_control_hardening.version_skew import evaluate_frozen_adversary
    results={case["case_id"]:evaluate_frozen_adversary(case) for case in cases()}

    expected_decisions={
        "V1_STALE_ALLOW_CURRENT_DENY":("DENY","STALE_AUTHORITY_CONTEXT"),
        "V2_POLICY_EPOCH_MISMATCH_WITHOUT_CURRENT_RECORD":("ABSTAIN","REQUIRED_AUTHORITY_UNAVAILABLE"),
        "V3_TASK_CHECKPOINT_DRIFT":("ABSTAIN","TASK_STATE_STALE"),
        "V5_SOURCE_BLOB_DRIFT":("ABSTAIN","SOURCE_IDENTITY_DRIFT"),
        "V6_REPOSITORY_DISAGREEMENT":("ABSTAIN","CROSS_SOURCE_DISAGREEMENT"),
        "V7_FAKE_EPISTEMIC_ALLOW":("ABSTAIN","NON_AUTHORITATIVE_PERMISSION_CLAIM"),
    }
    for case_id,(decision,reason) in expected_decisions.items():
        assert results[case_id]["decision"]==decision
        assert results[case_id]["reason_code"]==reason

    assert results["V4_RECEIPT_BINDING_MISMATCH"]["receipt_status"]=="INVALID_BINDING"
    assert results["V4_RECEIPT_BINDING_MISMATCH"]["authority_effect"]=="NONE"
    assert results["V8_CORRUPT_PROJECTION_DUPLICATE_ID"]["projection_status"]=="FAIL_CLOSED"
    assert results["V8_CORRUPT_PROJECTION_DUPLICATE_ID"]["authority_effect"]=="NONE"

def test_no_frozen_authority_sensitive_adversary_can_allow():
    from prototypes.semantic_control_hardening.version_skew import evaluate_frozen_adversary
    for case in cases():
        result=evaluate_frozen_adversary(case)
        if "decision" in result:
            assert result["decision"] in {"DENY","ABSTAIN"}

def test_current_deny_overrides_stale_projected_allow():
    from prototypes.semantic_control_hardening.version_skew import evaluate_policy_context
    result=evaluate_policy_context(
        projected_policy={"epoch":2,"decision":"ALLOW"},
        required_policy={"epoch":3,"decision":"DENY"},
    )
    assert result=={"decision":"DENY","reason_code":"STALE_AUTHORITY_CONTEXT"}

def test_missing_required_epoch_never_inherits_old_allow():
    from prototypes.semantic_control_hardening.version_skew import evaluate_policy_context
    result=evaluate_policy_context(
        projected_policy={"epoch":2,"decision":"ALLOW"},
        required_policy_epoch=3,
    )
    assert result=={"decision":"ABSTAIN","reason_code":"REQUIRED_AUTHORITY_UNAVAILABLE"}

def test_receipt_cannot_create_authority_even_when_binding_is_valid():
    from prototypes.semantic_control_hardening.version_skew import validate_receipt_binding
    result=validate_receipt_binding(
        expected_evaluation_id="eval:1",
        expected_candidate_action_id="action:1",
        receipt={"evaluation_id":"eval:1","candidate_action_id":"action:1"},
    )
    assert result["receipt_status"]=="VALID_BINDING"
    assert result["authority_effect"]=="NONE"

def test_duplicate_projection_ids_fail_closed():
    from prototypes.semantic_control_hardening.version_skew import validate_projection_records
    result=validate_projection_records([
        {"observation_id":"same","value":1},
        {"observation_id":"same","value":2},
    ])
    assert result=={
        "projection_status":"FAIL_CLOSED",
        "reason_code":"DUPLICATE_OBSERVATION_ID",
        "authority_effect":"NONE",
    }

def test_version_skew_module_has_no_external_runtime_or_mutation_paths():
    text=(ROOT/"prototypes"/"semantic_control_hardening"/"version_skew.py").read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "insert into","update ","delete from","truncate ",
        "heartbeat","github",
    ]:
        assert token not in text
