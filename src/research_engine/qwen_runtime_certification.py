from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

RUNTIME_CONTRACT_V3 = "qwen35-vllm-cu129-image-native-sampler-v3"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

COLD_V3_BASELINE = {
    "model_engine_initialization_seconds": 117.64840551599627,
    "vllm_engine_init_seconds": 93.11,
    "torch_compile_seconds": 18.17,
    "profiling_warmup_seconds": 48.84,
}


def certification_transition(
    cached_receipt: dict[str, Any] | None,
    *,
    requested_contract_identity: str,
    package_environment_compatible: bool,
) -> dict[str, Any]:
    cached_contract = (
        str(cached_receipt.get("runtime_contract_version"))
        if isinstance(cached_receipt, dict)
        and cached_receipt.get("runtime_contract_version") is not None
        else None
    )
    if cached_contract == requested_contract_identity:
        return {
            "status": "CERTIFICATION_HIT",
            "reason": "EXECUTION_CONTRACT_MATCH",
            "cached_contract_identity": cached_contract,
            "requested_contract_identity": requested_contract_identity,
            "package_environment_compatible": package_environment_compatible,
            "rebuild_install_required": not package_environment_compatible,
            "earliest_revalidation_stage": (
                "package_reconcile"
                if not package_environment_compatible
                else "none"
            ),
        }

    reason = (
        "NO_CACHED_CERTIFICATION"
        if cached_contract is None
        else "EXECUTION_CONTRACT_CHANGED"
    )
    return {
        "status": "CERTIFICATION_MISS",
        "reason": reason,
        "cached_contract_identity": cached_contract,
        "requested_contract_identity": requested_contract_identity,
        "package_environment_compatible": package_environment_compatible,
        "rebuild_install_required": not package_environment_compatible,
        "earliest_revalidation_stage": (
            "package_reconcile"
            if not package_environment_compatible
            else "runtime_multimodal_smoke"
        ),
    }


def parse_vllm_engine_timings(log_text: str) -> dict[str, float | None]:
    def last_float(pattern: str) -> float | None:
        matches = re.findall(pattern, log_text, flags=re.MULTILINE)
        return float(matches[-1]) if matches else None

    return {
        "model_engine_initialization_seconds": last_float(
            r'"event":\s*"QWEN_MODEL_INIT_COMPLETE".*?"seconds":\s*([0-9.]+)'
        ),
        "vllm_engine_init_seconds": last_float(
            r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s"
        ),
        "torch_compile_seconds": last_float(
            r"torch\.compile took ([0-9.]+) s in total"
        ),
        "profiling_warmup_seconds": last_float(
            r"Initial profiling/warmup run took ([0-9.]+) s"
        ),
    }


def classify_compiled_cache(
    *,
    cache_present_before: bool,
    log_text: str,
) -> dict[str, Any]:
    lowered = log_text.lower()
    explicit_hit_markers = (
        "directly load the compiled graph",
        "loaded compiled graph",
        "cache hit",
        "loading aot compiled",
    )
    explicit_miss_markers = (
        "cache miss",
        "compiling a graph for compile range",
        "saved aot compiled function",
    )
    hit = any(marker in lowered for marker in explicit_hit_markers)
    miss = any(marker in lowered for marker in explicit_miss_markers)

    if not cache_present_before:
        state = "MISS"
    elif hit and miss:
        state = "PARTIAL_HIT"
    elif hit:
        state = "VERIFIED_HIT"
    elif miss:
        state = "PRESENT_BUT_RECOMPILED"
    else:
        state = "PRESENT_UNVERIFIED"

    return {
        "state": state,
        "cache_present_before": cache_present_before,
        "explicit_reuse_evidence": hit,
        "explicit_compile_evidence": miss,
    }


def build_failure_envelope(
    *,
    certification_stage: str,
    parent_failure: str,
    child_role: str,
    first_causal_exception_type: str,
    first_causal_message: str,
    selected_backend: str,
    missing_external_requirement: dict[str, Any] | None,
    child_exit: dict[str, Any] | None,
    targeted_traceback_log_path: str | None,
    full_log_path: str | None,
    model_init_completed: bool,
    inference_calls_started: int,
    inference_calls_completed: int,
    scientific_state_materialized: bool,
    production_routing_changed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "certification_stage": certification_stage,
        "parent_failure": parent_failure,
        "child": {
            "role": child_role,
            "exit": child_exit,
        },
        "first_causal_exception": {
            "type": first_causal_exception_type,
            "message": first_causal_message,
        },
        "selected_backend": selected_backend,
        "missing_external_requirement": missing_external_requirement,
        "artifacts": {
            "targeted_traceback_log": targeted_traceback_log_path,
            "full_log": full_log_path,
        },
        "completion": {
            "model_init_completed": model_init_completed,
            "inference_calls_started": inference_calls_started,
            "inference_calls_completed": inference_calls_completed,
        },
        "scientific_state_materialized": scientific_state_materialized,
        "production_routing_changed": production_routing_changed,
    }


def operation_identity(receipt: dict[str, Any]) -> str:
    render_targets = list((receipt.get("renders") or {}).get("targets") or [])
    identity = {
        "operation": receipt.get("operation"),
        "git_source_identity": receipt.get("git_source_identity"),
        "model_revision": (receipt.get("model") or {}).get("revision"),
        "runtime_contract": (receipt.get("runtime") or {}).get("contract_version"),
        "sampling_backend": (receipt.get("runtime") or {}).get("sampling_backend"),
        "target_crop_sha256": sorted(
            str(item.get("crop_sha256"))
            for item in render_targets
            if item.get("crop_sha256")
        ),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def probe_completed_operation(
    receipt_paths: Iterable[str | Path],
    *,
    expected_operation_identity: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    inspected: list[str] = []
    for raw in receipt_paths:
        path = Path(raw)
        inspected.append(str(path))
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            receipt.get("status") == "SUCCESS"
            and receipt.get("terminal_stage") == "complete"
            and int(receipt.get("qwen_calls_started", -1)) == 3
            and int(receipt.get("qwen_calls_completed", -1)) == 3
            and operation_identity(receipt) == expected_operation_identity
        ):
            return {
                "schema_version": 1,
                "result": "COMPLETED_REUSE",
                "no_rerun_required": True,
                "probe_wall_seconds": time.perf_counter() - started,
                "durable_artifacts_inspected": inspected,
                "expensive_stages_repeated": [],
                "operation_identity": expected_operation_identity,
            }
    return {
        "schema_version": 1,
        "result": "NO_MATCH",
        "no_rerun_required": False,
        "probe_wall_seconds": time.perf_counter() - started,
        "durable_artifacts_inspected": inspected,
        "expensive_stages_repeated": [],
        "operation_identity": expected_operation_identity,
    }


def make_runtime_reuse_report(
    *,
    log_text: str,
    cache_present_before: bool,
    package_environment_reused: bool,
    execution_certification_reused: bool,
    live_engine_reused_for_domain_calls: bool,
) -> dict[str, Any]:
    timings = parse_vllm_engine_timings(log_text)
    compiled = classify_compiled_cache(
        cache_present_before=cache_present_before,
        log_text=log_text,
    )
    comparisons: dict[str, Any] = {}
    for key, baseline in COLD_V3_BASELINE.items():
        current = timings.get(key)
        comparisons[key] = {
            "cold_v3_baseline_seconds": baseline,
            "current_seconds": current,
            "delta_seconds": None if current is None else current - baseline,
            "speedup": None if current in (None, 0) else baseline / current,
        }
    return {
        "schema_version": 1,
        "reuse_layers": {
            "R0_package_environment": (
                "REUSED" if package_environment_reused else "REBUILT"
            ),
            "R1_execution_certification": (
                "VERIFIED_HIT"
                if execution_certification_reused
                else "REVALIDATED"
            ),
            "R2_compiled_jit_runtime_cache": compiled,
            "R3_live_initialized_engine_session": (
                "REUSED"
                if live_engine_reused_for_domain_calls
                else "NOT_REUSED"
            ),
        },
        "timings_seconds": timings,
        "comparison_to_successful_cold_v3": comparisons,
    }
