from __future__ import annotations

import pathlib


ROOT=pathlib.Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/".github"/"workflows"/"lambda-ci-local-extract-wave7a2.yml"


def test_wave7a2_workflow_is_exact_sha_and_bounded() -> None:
    text=WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 45" in text
    assert "/lambda-ci-local-extract-wave7a2 " in text
    assert "Verify exact PR head" in text
    assert "Prove exact Qwen runtime and two-GPU public binding" in text
    assert "model_calls" in text
    assert "<= 27" in text
    assert "WAVE7A2_GENERATION_RECEIPT=PASS" in text
    assert "WAVE7A2_EVAL_RECEIPT=PASS" in text


def test_wave7a2_workflow_preserves_scientific_boundaries() -> None:
    text=WORKFLOW.read_text()
    assert "scientific_state_materialized" in text
    assert "production_routing_changed" in text
    assert "qwen_verify_changed" in text
    assert "wave7b_started" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text


def test_wave7a2_workflow_references_existing_v2_contract() -> None:
    text=WORKFLOW.read_text()
    relative="benchmarks/document_intelligence/local-extract-wave7a2-evidence-conditioned.v2.json"
    assert relative in text
    assert (ROOT/relative).is_file()
