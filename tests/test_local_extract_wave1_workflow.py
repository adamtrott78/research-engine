from pathlib import Path


WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "lambda-ci-local-extract-wave1.yml"
)


def test_wave1_workflow_is_narrow_and_non_materializing():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "/lambda-ci-local-extract-wave1 " in text
    assert "workload=model-eval-local-extract-wave1" in text
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "P5_MODEL_ID" in text
    assert "P5_EXPANDED_RUNTIME_TREE_SHA256" in text
    assert "GPU-a333e68b-d1dc-6d25-1678-97b527bf55ea" in text
    assert "GPU-cfed3394-a935-63aa-9325-c4cae7b568b6" in text
    assert "run_local_extract_wave1.py" in text
    assert "local-extract-wave1-corpus.v1.json" in text
    assert "LOCAL_EXTRACT_WAVE1_RECEIPT=PASS" in text
    assert "RESEARCH_ENGINE_DATABASE_URL" not in text
    assert "SemanticResultMaterializer" not in text
    assert "docker.sock" not in text


def test_wave1_workflow_pins_ephemeral_pypdf_by_sha256():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pypdf-6.21.1-py3-none-any.whl" in text
    assert (
        "553e2d42f5bb3bb2a785a7d3de8ef90bb7df74b5413eff5a7b4388007ea9254b"
        in text
    )
    assert "-m zipfile -e" in text
