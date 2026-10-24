from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/lambda-ci-local-extract-amr-graph.yml")
DESIGN = Path(
    "benchmarks/document_intelligence/"
    "local-extract-amr-graph-certificate-design.v1.json"
)


def test_amr_workflow_is_exact_sha_gpu_gated() -> None:
    text = WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 90" in text
    assert "/lambda-ci-local-extract-amr-graph " in text
    assert "model-eval-local-extract-amr-graph-v1" in text
    assert 'actual_sha="$(python3 -c' in text
    assert '[[ "$actual_sha" == "$expected_sha" ]]' in text


def test_amr_workflow_pins_frozen_external_identities() -> None:
    text = WORKFLOW.read_text()
    for identity in (
        "38b29706ea982da5a5b081a04f00c93cc6cc1e9c",
        "eee40d776dbcaa028a9a82d325565b48937b9cfb",
        "0e040f26632ee16751240dbff28fa8d62c101ebb",
        "1bcd49cc8535056aa1c941763580fc5a572f1204",
        "22fa33834dccc11df99c4fc5fcc96c67f806dfdb",
        "5633e5e6f8a4f4398ce64abab71c473a",
        "1506463657",
        "71465f36c83cfef589facadcb014e70c0bad1b3b",
        "eb0b82a6f53b0a7d2a6288f0dff4ed2e72abf4ad",
        "802923ae6c7376a9930e93ce52cd6b9faf43c411",
        "7e816eb86104fa6c07d8eff9f9c6abc84bae2898",
    ):
        assert identity in text


def test_amr_workflow_never_fetches_extract_teacher_inputs() -> None:
    text = WORKFLOW.read_text().lower()
    assert "teacher-atom" not in text
    assert "wave1-corpus" not in text
    assert "codex" not in text
    assert "wave7b" not in text
    assert "run_local_extract_amr_graph_validation.py" in text


def test_frozen_design_remains_present() -> None:
    assert DESIGN.is_file()
    text = DESIGN.read_text()
    assert "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1" in text
    assert "AMR_GRAPH_CERTIFICATE_VALIDATED" in text
    assert "AMR_GRAPH_CERTIFICATE_PARTIAL" in text
    assert "AMR_GRAPH_CERTIFICATE_REJECTED" in text
