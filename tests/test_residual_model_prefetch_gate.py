import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "benchmarks" / "document_intelligence"


def test_source_audit_pass_receipt_opens_model_prefetch_only():
    receipt=json.loads((BENCH/"residual-source-audit-pass.v1.json").read_text(encoding="utf-8"))
    assert receipt["decision"] == "PASS"
    assert receipt["model_prefetch_ready"] is True
    assert receipt["candidate_model_downloads"] == 0
    assert receipt["candidate_model_inference_calls"] == 0
    assert receipt["scientific_state_materialized"] is False
    assert all(row["accepted"] for row in receipt["cases"].values())


def test_model_prefetch_contract_is_exactly_bounded():
    c=json.loads((BENCH/"residual-model-prefetch-contract.v1.json").read_text(encoding="utf-8"))
    assert c["source_gate_receipt"].endswith("residual-source-audit-pass.v1.json")
    assert c["required_state"]["minimum_free_disk_gib_before_download"] == 24
    assert c["required_state"]["candidate_model_inference_calls"] == 0
    assert len(c["model_artifacts"]) == 4
    assert {x["family"] for x in c["model_artifacts"]} == {
        "paddleocr-pp-ocrv6-medium","paddleocr-vl-1.6","qwen3.5-4b"
    }
    pinned={x["id"]:x["requested_revision"] for x in c["model_artifacts"] if x["revision_mode"]=="PINNED"}
    assert pinned["paddleocr-vl-1.6"] == "c5630abae1d940eafe0697512a0325494b02ab42"
    assert pinned["qwen3.5-4b"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert c["candidate_runtime_package_prefetch"][0]["wheel_sha256"] == "c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338"


def test_model_prefetch_helper_contains_no_inference_path():
    text=(BENCH/"prefetch_residual_models.py").read_text(encoding="utf-8")
    assert "snapshot_download" in text
    assert "local_files_only=True" in text
    for forbidden in [
        "from_pretrained(",
        ".generate(",
        "PaddleOCR(",
        "PaddleOCRVL(",
        "vllm",
        "sglang",
        "transformers",
    ]:
        assert forbidden not in text


def test_model_prefetch_runner_is_fail_closed_and_non_inference():
    text=(ROOT/"deploy"/"run-residual-model-prefetch-gate.sh").read_text(encoding="utf-8")
    assert "workspace-control-tmp" in text
    assert "require_stopped" in text
    assert "huggingface-hub==2.1.1" in text
    assert "paddleocr==3.7.0" in text
    assert "--no-deps --only-binary=:all:" in text
    assert "CANDIDATE_MODEL_INFERENCE_CALLS=0" in text
    assert "SCIENTIFIC_STATE_MATERIALIZED=false" in text
    for forbidden in [
        "systemctl --user start",
        "pip install paddlepaddle",
        "pip install vllm",
        "pip install sglang",
        "pip install transformers",
        "research-engine-daemon.service start",
    ]:
        assert forbidden not in text
