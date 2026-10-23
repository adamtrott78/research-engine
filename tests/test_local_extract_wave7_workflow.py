from __future__ import annotations

import pathlib


ROOT=pathlib.Path(__file__).resolve().parents[1]
WORKFLOW=ROOT/".github"/"workflows"/"lambda-ci-local-extract-wave7-intents.yml"


def test_wave7_workflow_uses_exact_qwen_binding_and_one_bounded_run() -> None:
    text=WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 45" in text
    assert "/lambda-ci-local-extract-wave7-intents " in text
    assert "Qwen/Qwen3.5-4B" in text
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in text
    assert "Generate bounded evidence intents with trusted Qwen" in text
    assert "Evaluate BM25 and BGE retrieval under fused top-24 budget" in text


def test_wave7_workflow_keeps_generation_and_retrieval_processes_separate() -> None:
    text=WORKFLOW.read_text()
    generate=text.index("run_local_extract_wave7_generate.py")
    evaluate=text.index("run_local_extract_wave7_evaluate.py")
    assert generate < evaluate
    assert "unset HF_HUB_OFFLINE" in text
    assert "unset TRANSFORMERS_OFFLINE" in text


def test_wave7_workflow_preserves_scientific_boundaries() -> None:
    text=WORKFLOW.read_text()
    assert "scientific_state_materialized" in text
    assert "production_routing_changed" in text
    assert "qwen_verify_changed" in text
    assert "wave7b_started" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
