from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "local-extract-obligation-validity-microbenchmark.yml"
)


def test_workflow_is_exact_sha_inert_gpu_trigger() -> None:
    text = WORKFLOW.read_text()
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 45" in text
    assert "/local-extract-obligation-validity-microbenchmark " in text
    assert "Verify exact PR head" in text
    assert "Checkout exact candidate SHA" in text
    assert "expected_sha" in text
    assert "github.actor == 'adamtrott78'" in text
    assert "author_association == 'OWNER'" in text


def test_workflow_pins_reviewed_design_and_source_identities() -> None:
    text = WORKFLOW.read_text()
    assert "4f4238030b7b667e080c2f662646e5856deacb4f" in text
    assert "c472329d560df3d7dccab2bf49bf94d6805faef2" in text
    assert "12580076512" in text
    assert "696fa26d942767844ddb24119ae98a503febac02da4e8f3dcbf2eeb021304b89" in text
    assert "e1a2f61f16cae4e957b10a9c24b475d5cb481afa9381084de65a1cd9c6568d23" in text
    assert "1097bdfabf6cd1b1e26ff9df58b6bb810e4bf74fbd3890ad20e86316d2ee0c91" in text


def test_workflow_pins_existing_qwen_runtime_without_download() -> None:
    text = WORKFLOW.read_text()
    assert "Qwen/Qwen3.5-4B" in text
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in text
    assert "6e28e2560c07e7dabab92c55dbfb6941d5a7a75c8068bdc7a98115d8dc62072c" in text
    assert "f67ff9a26690c55930232c915e59e826e3cca2dd89012de9e35a852032b38f74" in text
    assert 'HF_HUB_OFFLINE: "1"' in text
    assert 'TRANSFORMERS_OFFLINE: "1"' in text


def test_workflow_references_existing_implementation_inputs() -> None:
    text = WORKFLOW.read_text()
    paths = [
        "benchmarks/document_intelligence/run_local_extract_obligation_validity_microbenchmark.py",
        "benchmarks/document_intelligence/local-extract-wave1-corpus.v1.json",
        "benchmarks/document_intelligence/local-extract-obligation-validity-microbenchmark.v1.json",
    ]
    for relative in paths:
        assert relative in text
        assert (ROOT / relative).is_file()


def test_workflow_enforces_one_shot_report_boundaries() -> None:
    text = WORKFLOW.read_text()
    assert 'report["runtime_metrics"]["model_calls"] == 14' in text
    assert 'all(report["experiment_integrity"].values())' in text
    assert '"BOTH_VALIDITY_MECHANISMS_CONFIRMED"' in text
    assert '"SLOT_GROUNDING_ONLY_CONFIRMED"' in text
    assert '"HOLISTIC_VERIFIER_ONLY_CONFIRMED"' in text
    assert '"VALIDITY_PREMISE_REJECTED"' in text
    assert 'report["acceptance_candidate_frozen"] is False' in text
    assert 'report["sealed_acceptance_benchmark_accessed"] is False' in text
    assert "OBLIGATION_VALIDITY_PRECISION_MICROBENCHMARK_RECEIPT=PASS" in text
