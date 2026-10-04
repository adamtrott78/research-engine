import json
from pathlib import Path

ROOT=Path(__file__).parents[1]
BENCH=ROOT/"benchmarks"/"document_intelligence"


def test_ppocrv6_contract_pins_prefetched_models_and_targets():
    c=json.loads((BENCH/"ppocrv6-residual-benchmark-contract.v1.json").read_text(encoding="utf-8"))
    assert c["prefetch_receipt"].endswith("residual-model-prefetch-pass.v1.json")
    assert c["models"]["detection"]["revision"] == "8e0f56fb2ef86b461d99cfc7ac5c137738985f61"
    assert c["models"]["recognition"]["revision"] == "e5a92bcbc5cc1b494628e458d267778f0704fd7c"
    assert c["models"]["detection"]["inference_pdiparams_sha256"] == "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960"
    assert c["models"]["recognition"]["inference_pdiparams_sha256"] == "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319"
    assert [x["page_index_zero_based"] for x in c["authoritative_source"]["targets"]] == [3,16]
    assert c["inference_plan"]["candidate_inference_calls"] == 3
    assert c["quality_boundary"]["automatic_promotion"] is False
    assert c["quality_boundary"]["human_review_required"] is True


def test_ppocrv6_helper_is_single_family_and_resource_bounded():
    text=(BENCH/"run_ppocrv6_residual_benchmark.py").read_text(encoding="utf-8")
    assert "PaddleOCR(" in text
    assert 'engine="paddle_static"' in text
    assert 'use_doc_orientation_classify=False' in text
    assert 'use_doc_unwarping=False' in text
    assert 'use_textline_orientation=False' in text
    assert "GpuProcessMonitor" in text
    assert '"candidate_inference_calls": 0' in text
    for forbidden in ["PaddleOCRVL", "Qwen", "openai", "codex"]:
        assert forbidden not in text


def test_ppocrv6_runner_keeps_daemon_stopped_and_uses_only_two_pages():
    text=(ROOT/"deploy"/"run-ppocrv6-residual-benchmark.sh").read_text(encoding="utf-8")
    assert "workspace-control-tmp" in text
    assert "require_stopped" in text
    assert "paddlepaddle-gpu==3.2.0" in text
    assert "paddleocr-3.7.0-" in text
    assert 'targets=[("scan_textless",3),("low_confidence_text",16)]' in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "MEASURED_PENDING_HUMAN_QUALITY" in text
    for forbidden in [
        "systemctl --user start",
        "PaddleOCR-VL",
        "Qwen3.5",
        "OPENAI_API_KEY",
        "codex",
    ]:
        assert forbidden not in text
