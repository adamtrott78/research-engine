import json
from pathlib import Path

ROOT=Path(__file__).parents[1]
BENCH=ROOT/"benchmarks"/"document_intelligence"


def test_ppocr_review_is_fail_closed_not_promotion():
    r=json.loads((BENCH/"ppocrv6-human-quality-review.v1.json").read_text(encoding="utf-8"))
    assert r["overall_decision"]=="PP_OCRV6_RESOURCE_PASS_QUALITY_NOT_PROMOTED"
    assert r["human_review"]["scan_textless"]["benchmark_target_validity"]=="INVALID_FOR_OCR_RECOVERY_QUALITY"
    assert r["human_review"]["low_confidence_text"]["quality_decision"]=="FAIL_PROMOTION_THRESHOLD"
    assert r["production_change"] is False


def test_vlm_contract_is_exactly_targeted():
    c=json.loads((BENCH/"paddleocr-vl-residual-benchmark-contract.v1.json").read_text(encoding="utf-8"))
    assert c["model"]["revision"]=="c5630abae1d940eafe0697512a0325494b02ab42"
    assert c["model"]["model_safetensors_sha256"]=="85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
    assert c["runtime"]["engine"]=="transformers"
    assert c["runtime"]["torchvision_version"]=="0.23.0"
    assert c["runtime"]["trust_remote_code"] is True
    assert c["runtime"]["trust_remote_code_scope"]=="PINNED_LOCAL_PADDLEOCR_VL_1_6_MODEL_ONLY"
    assert c["runtime"]["use_layout_detection"] is False
    assert c["runtime"]["use_doc_orientation_classify"] is False
    assert c["runtime"]["use_doc_unwarping"] is False
    assert c["inference_plan"]["candidate_inference_calls"]==3
    assert [x["page_index_zero_based"] for x in c["authoritative_source"]["targets"]]==[7,3]
    assert c["quality_boundary"]["automatic_promotion"] is False
    assert c["quality_boundary"]["contemporaneous_codex_control_required_before_promotion"] is True


def test_vlm_helper_only_uses_paddleocr_vl_family():
    text=(BENCH/"run_paddleocr_vl_residual_benchmark.py").read_text(encoding="utf-8")
    assert "PaddleOCRVL(" in text
    assert 'engine="transformers"' in text
    assert 'engine_config={"trust_remote_code": True}' in text
    assert "use_layout_detection=False" in text
    assert "use_doc_orientation_classify=False" in text
    assert "use_doc_unwarping=False" in text
    assert "GpuProcessMonitor" in text
    for forbidden in ["Qwen", "openai", "codex", "PP-DocLayoutV3"]:
        assert forbidden not in text


def test_vlm_runner_is_offline_and_keeps_daemon_stopped():
    text=(ROOT/"deploy"/"run-paddleocr-vl-residual-benchmark.sh").read_text(encoding="utf-8")
    assert "workspace-control-tmp" in text
    assert "require_stopped" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "torch==2.8.0" in text
    assert "torchvision==0.23.0" in text
    assert "transformers==4.55.0" in text
    assert "STALE_VLM_RUNTIME_SENTINEL=1" in text
    assert "figure6_cifar_training_chart" in text
    assert "figure3_network_architecture" in text
    for forbidden in [
        "systemctl --user start",
        "PP-DocLayoutV3",
        "Qwen3.5",
        "OPENAI_API_KEY",
        "codex",
    ]:
        assert forbidden not in text
