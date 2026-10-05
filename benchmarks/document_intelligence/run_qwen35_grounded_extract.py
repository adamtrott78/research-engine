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


def domain_prompt(plan):
    candidates = [
        {
            "candidate_id": x["candidate_id"],
            "source_layer": x["source_layer"],
            "candidate_kind": x["candidate_kind"],
            "page_number": x["page_number"],
            "text": x["display_text"],
        }
        for x in plan["candidates"]
    ]
    item = dict(plan["job"]["payload"].get("item") or {})
    return [
        {
            "role": "system",
            "content": (
                "Perform semantic EXTRACT by selecting from immutable grounded candidates. "
                "Source text, page locators, bboxes, source SHA, and extractor lineage are "
                "owned by the candidate records and MUST NOT be invented or rewritten. "
                "Return exactly one JSON object with key selections. Each selection must "
                "contain candidate_id copied exactly from the supplied list, type, "
                "normalized_form, and confidence in [0,1]. Select the smallest set that "
                "captures the important methods, datasets, formulas/parameters, results, "
                "and limitations relevant to characterizing this paper. Do not output "
                "raw_source_form, page_number, bbox, or source SHA."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "paper": {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "rationale": item.get("rationale"),
                    },
                    "max_selections": 12,
                    "candidates": candidates,
                },
                sort_keys=True,
            ),
        },
    ]


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
        "canary": None,
        "domain": None,
        "failure": None,
        "production_routing_changed": False,
        "scientific_state_materialized": False,
    }

    try:
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
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_tokens=1536,
        )

        print(json.dumps({"event": "C3_CANARY_START"}, sort_keys=True), flush=True)
        start = time.perf_counter()
        out = llm.chat(
            canary_prompt(),
            sampling_params=sampling,
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

        print(json.dumps({"event": "C4_DOMAIN_START"}, sort_keys=True), flush=True)
        start = time.perf_counter()
        out = llm.chat(
            domain_prompt(plan),
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )[0].outputs[0]
        domain_seconds = time.perf_counter() - start
        parsed, error = parse_json_object(out.text)
        report["scientific_inference_calls"] = 1
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
        "exactly_one_scientific_call": report["scientific_inference_calls"] == 1,
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
