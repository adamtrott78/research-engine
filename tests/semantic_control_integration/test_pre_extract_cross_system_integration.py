import json
import os
from pathlib import Path
import warnings

import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.integration
def test_pre_extract_cross_system_integration_and_emit_result():
    if not os.getenv("RESEARCH_ENGINE_DATABASE_URL"):
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")
    from prototypes.semantic_control_integration.pre_extract_benchmark import run_pre_extract_integration
    result=run_pre_extract_integration(ROOT)
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
    assert result["performance_promotion_claim"] is False
    assert result["latency_threshold_applied"] is False
    warnings.warn(
        "SEMANTIC_CONTROL_PRE_EXTRACT_INTEGRATION_V1="+json.dumps(result,sort_keys=True,separators=(",",":")),
        UserWarning,
        stacklevel=1,
    )

def test_current_extract_gate_is_positive_but_not_accepted_output():
    gate=json.loads((ROOT/"benchmarks"/"semantic_control_integration"/"extract-dependency-gate-current.v1.json").read_text())
    assert gate["scientific_result"]["alignment_augment_decision"]=="ALIGNMENT_AUGMENT_FEASIBLE"
    assert gate["scientific_result"]["semantic_generalization_design_authorized"] is True
    assert gate["scientific_result"]["semantic_generalization_execution_authorized"] is False
    assert gate["scientific_result"]["production_promotion_authorized"] is False
    assert gate["accepted_output_checkpoint_present"] is False
    assert gate["gate_status"]=="BLOCKED_EVIDENCE"
