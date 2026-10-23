import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_integration"

def test_persisted_pre_extract_integration_result_is_pass_but_not_final_promotion():
    result=json.loads((BASE/"pre-extract-integration-result.v1.json").read_text())
    assert result["status"]=="PRE_EXTRACT_INTEGRATION_PASS"
    assert result["extract_binding"]=="BLOCKED_EVIDENCE"
    assert result["final_production_verdict_issued"] is False
    assert result["production_promotion_authorized"] is False
    assert result["authorization_escape_count"]==0
    assert all(result["gates"].values())
    assert result["authority_results"]["gpu_small_decision"]=="ALLOW"
    assert result["authority_results"]["gpu_heavy_decision"]=="DENY"
    assert result["authority_results"]["fake_epistemic_decision"]=="ABSTAIN"
    assert result["authority_results"]["missing_authority_decision"]=="ABSTAIN"
    assert max(x["projection_over_relational_percent"] for x in result["latency_comparison"]) < 1.0
    assert result["performance_promotion_claim"] is False
    assert result["latency_threshold_applied"] is False

def test_persisted_pre_extract_ci_evidence_is_exact():
    evidence=json.loads((BASE/"pre-extract-integration-evidence.v1.json").read_text())
    assert evidence["pr_number"]==203
    assert evidence["pr_head_sha"]=="dd5717ebc7d2963b6c262f2ac3578b06826dbae8"
    assert evidence["ci_run_id"]==40039432119
    assert evidence["ci_job_id"]==120706748226
    assert evidence["status"]=="PRE_EXTRACT_INTEGRATION_PASS"
    assert evidence["extract_binding"]=="BLOCKED_EVIDENCE"
    assert evidence["production_promotion_authorized"] is False
