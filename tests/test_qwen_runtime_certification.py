from pathlib import Path
import json

from research_engine.qwen_runtime_certification import (
    RUNTIME_CONTRACT_V3,
    certification_transition,
    classify_compiled_cache,
    operation_identity,
    parse_vllm_engine_timings,
    probe_completed_operation,
)


ROOT = Path(__file__).parents[1]
BENCH = ROOT / "benchmarks" / "document_intelligence"
FIXTURES = ROOT / "tests" / "fixtures"


def test_typed_execution_contract_change_preserves_package_reuse():
    cached = {
        "runtime_contract_version": "qwen35-vllm-cu129-image-v2",
    }
    result = certification_transition(
        cached,
        requested_contract_identity=RUNTIME_CONTRACT_V3,
        package_environment_compatible=True,
    )
    assert result["status"] == "CERTIFICATION_MISS"
    assert result["reason"] == "EXECUTION_CONTRACT_CHANGED"
    assert result["cached_contract_identity"] == "qwen35-vllm-cu129-image-v2"
    assert result["requested_contract_identity"] == RUNTIME_CONTRACT_V3
    assert result["package_environment_compatible"] is True
    assert result["rebuild_install_required"] is False
    assert result["earliest_revalidation_stage"] == "runtime_multimodal_smoke"


def test_historical_flashinfer_failure_envelope_preserves_root_cause():
    envelope = json.loads(
        (BENCH / "qwen35-flashinfer-nvcc-failure-envelope.v1.json").read_text()
    )
    assert envelope["certification_stage"] == "engine_initialization"
    assert envelope["child"]["role"] == "flashinfer_sampling_jit"
    assert envelope["selected_backend"] == "flashinfer_top_k_top_p_sampler"
    assert envelope["missing_external_requirement"]["name"] == "nvcc"
    assert "/usr/local/cuda" in envelope["first_causal_exception"]["message"]
    assert envelope["completion"]["model_init_completed"] is False
    assert envelope["completion"]["inference_calls_started"] == 0
    assert envelope["completion"]["inference_calls_completed"] == 0
    assert envelope["scientific_state_materialized"] is False
    assert envelope["production_routing_changed"] is False
    assert envelope["artifacts"]["targeted_traceback_log"]
    assert envelope["artifacts"]["full_log"]


def test_idempotent_recovery_recognizes_success_without_expensive_work():
    path = FIXTURES / "qwen35-success-receipt.v1.json"
    receipt = json.loads(path.read_text())
    identity = operation_identity(receipt)
    result = probe_completed_operation(
        [path],
        expected_operation_identity=identity,
    )
    assert result["result"] == "COMPLETED_REUSE"
    assert result["no_rerun_required"] is True
    assert result["expensive_stages_repeated"] == []
    assert result["durable_artifacts_inspected"] == [str(path)]
    assert result["probe_wall_seconds"] >= 0


def test_compile_cache_is_not_called_a_hit_from_presence_alone():
    present_only = classify_compiled_cache(
        cache_present_before=True,
        log_text="Using cache directory: /tmp/cache",
    )
    assert present_only["state"] == "PRESENT_UNVERIFIED"

    cold = classify_compiled_cache(
        cache_present_before=False,
        log_text="Compiling a graph for compile range (1, 8192) takes 12.69 s",
    )
    assert cold["state"] == "MISS"


def test_successful_cold_v3_log_timing_parser():
    text = """
{"event":"QWEN_MODEL_INIT_COMPLETE","seconds":117.64840551599627}
INFO init engine (profile, create kv cache, warmup model) took 93.11 s (compilation: 18.17 s)
INFO torch.compile took 18.17 s in total
INFO Initial profiling/warmup run took 48.84 s
"""
    parsed = parse_vllm_engine_timings(text)
    assert parsed["model_engine_initialization_seconds"] == 117.64840551599627
    assert parsed["vllm_engine_init_seconds"] == 93.11
    assert parsed["torch_compile_seconds"] == 18.17
    assert parsed["profiling_warmup_seconds"] == 48.84


def test_c3_contract_is_separate_and_reuses_live_engine():
    contract = json.loads(
        (BENCH / "qwen35-runtime-certification-contract.v1.json").read_text()
    )
    assert contract["operation"] == "QWEN35_SAME_MODALITY_RUNTIME_CERTIFICATION_C3"
    assert contract["scientific_benchmark_call_budget_impact"] == 0
    assert contract["requires_separate_authorization"] is True
    assert contract["engine_lifecycle"]["boot_for_certification_only"] is False
    assert contract["engine_lifecycle"]["reuse_same_live_engine_for_domain_work"] is True
    assert contract["scientific_state_materialized"] is False
    assert contract["production_routing_changed"] is False


def test_wave3_runner_instrumentation_adds_no_scientific_calls():
    text = (ROOT / "deploy" / "run-qwen35-visual-benchmark.sh").read_text()
    assert "CERTIFICATION_TRANSITION=" in text
    assert "qwen35-runtime-reuse-evidence.json" in text
    assert "VLLM_COMPILE_CACHE_PRESENT_BEFORE=" in text
    assert "scientific_call_count_added_for_wave3" in text
    assert "make_runtime_reuse_report" in text
    assert text.count("QWEN_CALL_START") == 0  # helper owns the exact three calls
    helper = (BENCH / "run_qwen35_visual_benchmark.py").read_text()
    assert helper.count('"QWEN_CALL_START"') == 1
    assert '("figure6_first", args.figure6, FIGURE6_PROMPT, False)' in helper
    assert '("figure3_warm", args.figure3, FIGURE3_PROMPT, True)' in helper
    assert '("figure6_warm_repeat", args.figure6, FIGURE6_PROMPT, True)' in helper


def test_completed_operation_probe_is_dry_run_only():
    text = (BENCH / "probe_qwen35_completed_operation.py").read_text()
    assert '"NON_DESTRUCTIVE_DRY_RUN"' in text
    assert '"benchmark_launch_attempted"] = False' in text
    assert '"package_install_or_download_attempted"] = False' in text
    assert '"model_initialization_attempted"] = False' in text
    assert '"inference_attempted"] = False' in text
    for forbidden in [
        "subprocess.run",
        "pip install",
        "LLM(",
        "run-qwen35-visual-benchmark.sh",
    ]:
        assert forbidden not in text
