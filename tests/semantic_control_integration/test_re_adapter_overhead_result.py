import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_integration"

def test_persisted_re_adapter_overhead_evidence():
    result=json.loads((BASE/"research-engine-adapter-overhead-result.v1.json").read_text())
    assert result["ci_run_id"]==40037743704
    assert result["ci_job_id"]==120701255180
    assert result["adapter_merge_commit"]=="e57a2f18b52ba175e4e205b47336b5031bad65e1"
    assert result["performance_promotion_claim"] is False
    assert result["latency_threshold_applied"] is False
    assert result["production_database_access"] is False
    assert result["production_mutation"] is False
    assert [x["query_id"] for x in result["query_results"]]==[
        "Q1_CLAIM_SUPPORT","Q2_METHOD_QUALIFIERS","Q3_SOURCE_LEVEL_COUNTS","Q4_PROVENANCE_PATH"
    ]
    assert all(x["measured_repetitions"]==500 for x in result["query_results"])
