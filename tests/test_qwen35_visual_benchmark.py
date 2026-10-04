import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BENCH = ROOT / "benchmarks" / "document_intelligence"


def test_paddle_review_opens_qwen_fallback():
    review = json.loads(
        (BENCH / "paddleocr-vl-human-quality-review.v1.json").read_text()
    )
    assert review["overall_decision"] == "PERCEPTION_USEFUL_SEMANTIC_NOT_PROMOTED"
    assert (
        review["next_gate"]
        == "QWEN3_5_4B_TARGETED_VISUAL_FALLBACK_BENCHMARK"
    )
    assert review["production_change"] is False


def test_qwen_visual_contract_is_targeted_and_fail_closed():
    contract = json.loads(
        (BENCH / "qwen35-visual-benchmark-contract.v1.json").read_text()
    )
    assert (
        contract["model"]["revision"]
        == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert contract["runtime"]["engine"] == "vLLM"
    assert contract["runtime"]["vllm_version"] == "0.30.0"
    assert contract["runtime"]["gpu_memory_utilization"] == 0.65
    assert contract["runtime"]["enforce_eager"] is False
    assert contract["inference_plan"]["candidate_inference_calls"] == 3
    assert contract["quality_boundary"]["fabricated_fact_tolerance"] == 0
    assert contract["quality_boundary"]["automatic_promotion"] is False
    assert (
        contract["quality_boundary"][
            "contemporaneous_codex_control_required_before_promotion"
        ]
        is True
    )
    assert [
        x["page_index_zero_based"]
        for x in contract["authoritative_source"]["targets"]
    ] == [7, 3]


def test_qwen_helper_forbids_numeric_curve_estimation():
    text = (BENCH / "run_qwen35_visual_benchmark.py").read_text()
    assert "Never estimate unprinted curve values" in text
    assert "curve_numeric_estimates must remain an empty list" in text
    assert "QWEN_CALL_START" in text
    assert "QWEN_CALL_COMPLETE" in text
    assert "gpu_memory_utilization=0.65" in text
    assert "enforce_eager=True" not in text
    assert 'limit_mm_per_prompt={"image": 1}' in text
    assert "AutoModel" not in text
    for forbidden in ["openai", "requests.get", "http://", "https://"]:
        assert forbidden not in text.lower()
    for forbidden in ["OPENAI_API_KEY", "responses.create(", "chat.completions", "api.openai.com"]:
        assert forbidden not in text


def test_qwen_runner_keeps_daemon_stopped_and_model_offline():
    text = (ROOT / "deploy" / "run-qwen35-visual-benchmark.sh").read_text()
    assert "vllm==0.30.0" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "SELECTED_GPU_HAS_ACTIVE_COMPUTE_PROCESS" in text
    assert "figure6_cifar_training_chart" in text
    assert "figure3_network_architecture" in text
    for forbidden in [
        "systemctl --user start",
        "OPENAI_API_KEY",
        "Qwen3.5-8B",
    ]:
        assert forbidden not in text
