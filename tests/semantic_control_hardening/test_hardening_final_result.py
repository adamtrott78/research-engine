import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"


def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def test_persisted_integrated_hardening_result_integrity_and_verdict():
    result=json.loads((BASE/"integrated-hardening-result.v1.json").read_text())
    stored=result.pop("result_sha256")
    actual=hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    assert actual==stored=="3b14229dca3b2c06915a97bd820a21536ecfdfa55d944458e709e6b1460b98f6"
    assert result["verdict"]=="HARDENING_PASS"
    assert all(result["gates"].values())
    assert result["authorization_escape_count"]==0
    assert result["audit_valid"] is True
    assert result["audit_tamper_detected"] is True
    assert result["hardening_authorizes_production"] is False
    assert result["hardening_binds_extract"] is False
    assert result["production_mutation"] is False
    assert result["extract_mutation"] is False


def test_persisted_integrated_hardening_evidence_binds_exact_corrected_ci():
    evidence=json.loads((BASE/"integrated-hardening-evidence.v1.json").read_text())
    assert evidence["pr_number"]==214
    assert evidence["corrected_pr_head_sha"]=="edc2e5f9c80de956f48919e4b3b3ec2a469d98b6"
    assert evidence["ci_run_id"]==40048589513
    assert evidence["ci_job_id"]==120735879415
    assert evidence["initial_collection_failure"]["semantic_failure"] is False
    assert evidence["verdict"]=="HARDENING_PASS"
    assert evidence["authorization_escape_count"]==0
    assert evidence["hardening_authorizes_production"] is False
    assert evidence["hardening_binds_extract"] is False
    assert evidence["production_mutation"] is False
    assert evidence["extract_mutation"] is False
