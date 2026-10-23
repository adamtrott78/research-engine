import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control_integration"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_persisted_relational_baseline_result_integrity_and_safety():
    result = json.loads((BASE / "relational-baseline-result.v1.json").read_text())
    stored = result.pop("baseline_result_sha256")
    actual = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    assert actual == stored == "09a2b695580a2262629bca33dc568fe450edea726d92d70c1b9f24d5c0d5fcca"
    assert result["all_answers_correct"] is True
    assert result["execution_environment"]["production_database_access"] is False
    assert result["performance_promotion_claim"] is False
    assert result["latency_threshold_applied"] is False
    assert result["production_mutation"] is False
    assert len(result["query_results"]) == 4
    assert all(item["answer_correct"] for item in result["query_results"])


def test_persisted_baseline_evidence_binds_exact_ci_run():
    evidence = json.loads((BASE / "relational-baseline-evidence.v1.json").read_text())
    assert evidence["pr_number"] == 195
    assert evidence["pr_head_sha"] == "460c76075a51ea0c98f406a6d455e853beb8bed9"
    assert evidence["ci_run_id"] == 40036307847
    assert evidence["ci_job_id"] == 120696530150
    assert evidence["result_sha256"] == "09a2b695580a2262629bca33dc568fe450edea726d92d70c1b9f24d5c0d5fcca"
    assert evidence["production_database_access"] is False
    assert evidence["production_mutation"] is False
