#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


CANARY_EXPECTED = ["synthetic:beta"]
MODEL_MAX_TOKENS = 8192
DOMAIN_MAX_OUTPUT_TOKENS = 1024
DOMAIN_INPUT_TOKEN_LIMIT = 7000


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    value = text.strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        return None, "no_json_object"
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (parsed, None) if isinstance(parsed, dict) else (None, "json_not_object")


class GpuMonitor:
    def __init__(self, gpu_uuid: str, baseline_used_mib: float, interval: float = 0.2):
        self.gpu_uuid = gpu_uuid
        self.baseline_used_mib = baseline_used_mib
        self.peak_total_used_mib = baseline_used_mib
        self.peak_tree_rss_mib = 0.0
        self.samples = 0
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        import psutil
        root = psutil.Process(os.getpid())
        rss = 0
        for proc in [root] + root.children(recursive=True):
            try:
                rss += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_tree_rss_mib = max(self.peak_tree_rss_mib, rss / (1024**2))
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        for raw in cp.stdout.splitlines():
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) == 2 and parts[0] == self.gpu_uuid:
                try:
                    self.peak_total_used_mib = max(
                        self.peak_total_used_mib, float(parts[1])
                    )
                except ValueError:
                    pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample()
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        try:
            self._sample()
        except Exception:
            pass

    @property
    def peak_delta_mib(self):
        return max(0.0, self.peak_total_used_mib - self.baseline_used_mib)


def canary_prompt():
    candidates = [
        {"candidate_id": "synthetic:alpha", "text": "The alpha record says seven."},
        {"candidate_id": "synthetic:beta", "text": "The beta record says eleven."},
        {"candidate_id": "synthetic:gamma", "text": "The gamma record says thirteen."},
    ]
    return [
        {
            "role": "system",
            "content": (
                "Synthetic non-scientific runtime canary. Select the candidate whose "
                "text says eleven. Return exactly one JSON object with key "
                "selected_candidate_ids and no other keys."
            ),
        },
        {"role": "user", "content": json.dumps({"candidates": candidates}, sort_keys=True)},
    ]


def _candidate_excerpt(text: str, cap: int | None) -> str:
    value = str(text)
    if cap is None or len(value) <= cap:
        return value
    if cap < 32:
        return value[:cap]
    left = (cap - 3) // 2
    right = cap - 3 - left
    return value[:left] + "..." + value[-right:]


def domain_prompt(plan, *, excerpt_cap: int | None = None):
    # Keep candidate IDs immutable and system-owned provenance out of the model's
    # output contract. The model only needs semantic content to select IDs.
    candidates = [
        [
            x["candidate_id"],
            _candidate_excerpt(x["display_text"], excerpt_cap),
        ]
        for x in plan["candidates"]
    ]
    item = dict(plan["job"]["payload"].get("item") or {})
    return [
        {
            "role": "system",
            "content": (
                "Select semantic evidence from immutable grounded candidates. "
                "Each candidate row is [candidate_id, text]. Return exactly one JSON "
                "object with key selections. Each selection must contain candidate_id "
                "copied exactly from the supplied list, type, normalized_form, and "
                "confidence in [0,1]. Select the smallest set covering important methods, "
                "datasets, formulas or parameters, results, and limitations. Do not "
                "invent or output source quotes, page numbers, bboxes, or source SHA."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "paper": [
                        item.get("id"),
                        item.get("title"),
                        item.get("rationale"),
                    ],
                    "max_selections": 12,
                    "candidates": candidates,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def _chat_token_count(tokenizer, messages) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(encoded)


def fit_domain_prompt(plan, tokenizer):
    # Token-aware deterministic preflight. Preserve every candidate ID and only
    # shorten model-visible excerpts if the compact full-text representation does
    # not fit. Immutable provenance/full source text remains in the plan.
    caps = [None, 480, 360, 280, 220, 180, 140, 110, 80]
    original_ids = [str(x["candidate_id"]) for x in plan["candidates"]]
    for cap in caps:
        messages = domain_prompt(plan, excerpt_cap=cap)
        token_count = _chat_token_count(tokenizer, messages)
        if token_count <= DOMAIN_INPUT_TOKEN_LIMIT:
            visible_ids = [
                str(row[0])
                for row in json.loads(messages[1]["content"])["candidates"]
            ]
            if visible_ids != original_ids:
                raise RuntimeError("candidate identity changed during prompt fit")
            return messages, {
                "input_tokens": token_count,
                "input_token_limit": DOMAIN_INPUT_TOKEN_LIMIT,
                "model_max_tokens": MODEL_MAX_TOKENS,
                "reserved_output_tokens": DOMAIN_MAX_OUTPUT_TOKENS,
                "candidate_count": len(original_ids),
                "all_candidate_ids_preserved": True,
                "excerpt_cap_chars": cap,
            }
    raise RuntimeError(
        "C4 prompt cannot fit token budget while preserving all candidate IDs"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--baseline-gpu-used-mib", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    monitor = GpuMonitor(args.gpu_uuid, args.baseline_gpu_used_mib)
    monitor.start()
    report = {
        "schema_version": 1,
        "operation": "QWEN35_GROUNDED_CANDIDATE_EXTRACT_CALIBRATION",
        "candidate_family": "qwen3.5-4b",
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "model_init_seconds": None,
        "c3_canary_calls": 0,
        "scientific_inference_calls": 0,
        "scientific_inference_calls_started": 0,
        "scientific_inference_calls_completed": 0,
        "prompt_fit": None,
        "failure_stage": None,
        "canary": None,
        "domain": None,
        "failure": None,
        "production_routing_changed": False,
        "scientific_state_materialized": False,
    }

    current_stage = "C0_STATIC"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        domain_messages, prompt_fit = fit_domain_prompt(plan, tokenizer)
        report["prompt_fit"] = prompt_fit
        print(
            json.dumps({"event": "C4_PROMPT_PREFLIGHT_PASS", **prompt_fit}, sort_keys=True),
            flush=True,
        )

        current_stage = "C2B_ENGINE_BOOT"
        print(json.dumps({"event": "QWEN_MODEL_INIT_START"}, sort_keys=True), flush=True)
        started = time.perf_counter()
        llm = LLM(
            model=str(args.model_dir),
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=1,
            gpu_memory_utilization=0.65,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=False,
            disable_log_stats=True,
            mm_processor_cache_gb=0.25,
        )
        init = time.perf_counter() - started
        report["model_init_seconds"] = init
        print(
            json.dumps({"event": "QWEN_MODEL_INIT_COMPLETE", "seconds": init}, sort_keys=True),
            flush=True,
        )
        canary_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_tokens=64,
        )
        domain_sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_tokens=DOMAIN_MAX_OUTPUT_TOKENS,
        )

        current_stage = "C3_CANARY"
        print(json.dumps({"event": "C3_CANARY_START"}, sort_keys=True), flush=True)
        start = time.perf_counter()
        out = llm.chat(
            canary_prompt(),
            sampling_params=canary_sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )[0].outputs[0]
        canary_seconds = time.perf_counter() - start
        parsed, error = parse_json_object(out.text)
        selected = (
            list(parsed.get("selected_candidate_ids") or [])
            if parsed is not None
            else []
        )
        canary_pass = selected == CANARY_EXPECTED
        report["c3_canary_calls"] = 1
        report["canary"] = {
            "seconds": canary_seconds,
            "parsed": parsed,
            "parse_error": error,
            "pass": canary_pass,
            "expected_selected_candidate_ids": CANARY_EXPECTED,
            "synthetic_non_scientific": True,
        }
        print(
            json.dumps(
                {"event": "C3_CANARY_COMPLETE", "seconds": canary_seconds, "pass": canary_pass},
                sort_keys=True,
            ),
            flush=True,
        )
        if not canary_pass:
            raise RuntimeError("C3 same-modality canary failed")

        current_stage = "C4_DOMAIN"
        print(json.dumps({"event": "C4_DOMAIN_START"}, sort_keys=True), flush=True)
        report["scientific_inference_calls_started"] = 1
        start = time.perf_counter()
        out = llm.chat(
            domain_messages,
            sampling_params=domain_sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )[0].outputs[0]
        domain_seconds = time.perf_counter() - start
        parsed, error = parse_json_object(out.text)
        report["scientific_inference_calls"] = 1
        report["scientific_inference_calls_completed"] = 1
        report["domain"] = {
            "seconds": domain_seconds,
            "generated_text": out.text,
            "parsed": parsed,
            "parse_error": error,
            "structured_output_valid": parsed is not None,
            "output_token_count": len(out.token_ids),
        }
        print(
            json.dumps(
                {
                    "event": "C4_DOMAIN_COMPLETE",
                    "seconds": domain_seconds,
                    "structured_output_valid": parsed is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:
        report["failure_stage"] = current_stage
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor.stop()
        report["gpu_monitor_samples"] = monitor.samples
        report["peak_total_gpu_used_mib"] = monitor.peak_total_used_mib
        report["peak_gpu_delta_mib"] = monitor.peak_delta_mib
        report["peak_tree_rss_mib"] = monitor.peak_tree_rss_mib

    checks = {
        "runtime_completed": report["failure"] is None,
        "model_init": (report.get("model_init_seconds") or float("inf")) <= 120,
        "c3_canary": bool((report.get("canary") or {}).get("pass")),
        "exactly_one_scientific_call": (
            report["scientific_inference_calls_started"] == 1
            and report["scientific_inference_calls_completed"] == 1
            and report["scientific_inference_calls"] == 1
        ),
        "domain_structured_json": bool(
            (report.get("domain") or {}).get("structured_output_valid")
        ),
        "peak_vram": report["peak_gpu_delta_mib"] <= 16384,
        "peak_rss": report["peak_tree_rss_mib"] <= 32768,
    }
    report["checks"] = checks
    report["measurement_decision"] = (
        "MEASURED_PENDING_GUARD_AND_CODEX_CONTROL"
        if all(checks.values())
        else "REJECT_RESOURCE_RUNTIME_OR_STRUCTURE"
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "measurement_decision": report["measurement_decision"],
                "model_init_seconds": report.get("model_init_seconds"),
                "c3_canary_pass": bool((report.get("canary") or {}).get("pass")),
                "scientific_inference_calls": report["scientific_inference_calls"],
                "scientific_inference_calls_started": report["scientific_inference_calls_started"],
                "scientific_inference_calls_completed": report["scientific_inference_calls_completed"],
                "prompt_fit": report["prompt_fit"],
                "failure_stage": report["failure_stage"],
                "domain_seconds": (report.get("domain") or {}).get("seconds"),
                "peak_gpu_delta_mib": report["peak_gpu_delta_mib"],
                "peak_tree_rss_mib": report["peak_tree_rss_mib"],
                "failure": report["failure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["measurement_decision"] == "MEASURED_PENDING_GUARD_AND_CODEX_CONTROL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
