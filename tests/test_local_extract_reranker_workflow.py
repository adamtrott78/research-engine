from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "lambda-ci-local-extract-reranker-bakeoff.yml"
)


def test_reranker_workflow_is_single_gpu_exact_sha_and_owner_gated() -> None:
    text = WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 45" in text
    assert "github.actor == 'adamtrott78'" in text
    assert "github.event.comment.author_association == 'OWNER'" in text
    assert "/lambda-ci-local-extract-reranker-bakeoff " in text
    assert "Verify exact PR head" in text
    assert "Checkout exact candidate SHA" in text
    assert "P5_MODEL_ID" in text
    assert "P5_MODEL_PATH" in text
    assert "GPU-a333e68b-d1dc-6d25-1678-97b527bf55ea" in text


def test_reranker_workflow_uses_exact_ephemeral_dependencies() -> None:
    text = WORKFLOW.read_text()
    assert "pypdf-6.21.1-py3-none-any.whl" in text
    assert (
        "553e2d42f5bb3bb2a785a7d3de8ef90bb7df74b5413eff5a7b4388007ea9254b"
        in text
    )
    assert "flagembedding-1.4.2-py3-none-any.whl" not in text.lower()
    assert "pip install" not in text
    assert "torch.__version__ == \"2.13.0+cu129\"" in text
    assert 'PYTHONPATH="$pydeps"' in text


def test_reranker_workflow_runs_only_the_bounded_diagnostic() -> None:
    text = WORKFLOW.read_text()
    assert "run_local_extract_reranker_bakeoff.py" in text
    assert "local-extract-proposition-reranker-bakeoff.v1.json" in text
    assert "local-extract-teacher-atom-support.v1.json" in text
    assert "PROPOSITION_RERANKER_BAKEOFF_RECEIPT=PASS" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
    assert "scientific_state_materialized" in text
    assert "production_routing_changed" in text
    assert "qwen_verify_changed" in text
