import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def test_persisted_scale_result_is_correct_and_non_authoritative():
    result=json.loads((BASE/"scale-result.v1.json").read_text())
    assert result["all_correct"] is True
    assert [item["size"] for item in result["results"]]==[1000,10000,100000]
    assert all(item["authorization_escape_count"]==0 for item in result["results"])
    assert all(item["deterministic_rebuild"] for item in result["results"])
    assert all(item["slice_bounded"] for item in result["results"])
    assert all(item["provenance_exact_generated_sources"] for item in result["results"])
    assert result["pass_fail_latency_threshold"] is None
    assert result["performance_promotion_claim"] is False
    assert result["production_mutation"] is False
    assert result["scientific_evidence"] is False

def test_persisted_scale_evidence_binds_exact_ci_run():
    evidence=json.loads((BASE/"scale-evidence.v1.json").read_text())
    assert evidence["pr_number"]==212
    assert evidence["pr_head_sha"]=="6f6ca884275e436a13bff655334fd7892dfad06b"
    assert evidence["ci_run_id"]==40047074263
    assert evidence["ci_job_id"]==120731149942
    assert evidence["all_correct"] is True
    assert evidence["production_mutation"] is False
    assert evidence["scientific_evidence"] is False
