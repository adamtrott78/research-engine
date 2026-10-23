from __future__ import annotations

import pathlib


ROOT=pathlib.Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/".github"/"workflows"/"lambda-ci-local-extract-oracle-query.yml"


def test_oracle_query_workflow_is_exact_sha_single_gpu_and_bounded() -> None:
    text=WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 45" in text
    assert "/lambda-ci-local-extract-oracle-query " in text
    assert "Verify exact PR head" in text
    assert "Prove single-GPU public-payload execution boundary" in text
    assert "P5_MODEL_ID" in text
    assert "P5_MODEL_PATH" in text
    assert "run_local_extract_oracle_query.py" in text
    assert "ORACLE_QUERY_DIAGNOSTIC_RECEIPT=PASS" in text


def test_oracle_query_workflow_preserves_scientific_boundaries() -> None:
    text=WORKFLOW.read_text()
    assert "scientific_state_materialized" in text
    assert "production_routing_changed" in text
    assert "qwen_verify_changed" in text
    assert "wave7_started" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
