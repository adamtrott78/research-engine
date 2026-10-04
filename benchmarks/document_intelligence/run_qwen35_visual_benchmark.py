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

SYSTEM_PROMPT = (
    "You are evaluating scientific evidence from one authoritative figure region "
    "plus supplied caption facts. Use only those inputs. Return one JSON object "
    "and nothing else. Never estimate unprinted curve values. Never introduce "
    "outside facts. If a requested relation is not supported, use null or an "
    "empty list and explain it only in uncertain_claims."
)

FIGURE6_PROMPT = """Caption context facts:
- This is CIFAR-10 training.
- Dashed curves are training error and bold curves are testing error.
- Left panel: plain networks. Middle panel: ResNets. Right panel: ResNets with 110 and 1202 layers.
- Plain-110 is not displayed because its error is above 60 percent.

Inspect the image and return exactly this JSON structure:
{
  "figure_id": "figure6",
  "panel_count": integer_or_null,
  "panels": [
    {"position":"left","family":string_or_null,"models":[strings]},
    {"position":"middle","family":string_or_null,"models":[strings]},
    {"position":"right","family":string_or_null,"models":[strings]}
  ],
  "line_style_semantics":{"dashed":string_or_null,"bold":string_or_null},
  "plain_110_displayed": boolean_or_null,
  "curve_numeric_estimates": [],
  "uncertain_claims":[strings],
  "abstain": boolean
}
Do not estimate any plotted curve values. curve_numeric_estimates must remain an empty list."""

FIGURE3_PROMPT = """Caption context facts:
- The three columns are VGG-19, a 34-layer plain network, and a 34-layer residual network.
- Caption FLOPs: VGG-19 = 19.6 billion; 34-layer plain = 3.6 billion; 34-layer residual = 3.6 billion.
- The caption states that dotted shortcuts increase dimensions.

Inspect the image and return exactly this JSON structure:
{
  "figure_id":"figure3",
  "architecture_columns":[strings],
  "shortcut_presence":{"VGG-19":boolean_or_null,"34-layer plain":boolean_or_null,"34-layer residual":boolean_or_null},
  "solid_shortcut_relation":string_or_null,
  "dotted_shortcut_relation":string_or_null,
  "stage_channel_widths":[integers],
  "caption_flops_billion":{"VGG-19":number_or_null,"34-layer plain":number_or_null,"34-layer residual":number_or_null},
  "uncertain_claims":[strings],
  "abstain":boolean
}
Do not infer facts not supported by the image or caption context."""


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    value = text.strip()
    if value.startswith("~~~") or value.startswith("'''"):
        lines = value.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() in {"~~~", "'''"}:
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        return None, "no_json_object"
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, None


class GpuDeltaMonitor:
    def __init__(self, gpu_uuid: str, baseline_used_mib: float, interval: float = 0.2):
        self.gpu_uuid = gpu_uuid
        self.baseline_used_mib = baseline_used_mib
        self.interval = interval
        self.peak_total_used_mib = baseline_used_mib
        self.peak_tree_rss_mib = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        import psutil

        root = psutil.Process(os.getpid())
        processes = [root] + root.children(recursive=True)
        rss_bytes = 0
        for proc in processes:
            try:
                rss_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_tree_rss_mib = max(
            self.peak_tree_rss_mib, rss_bytes / (1024.0 * 1024.0)
        )

        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        for raw in cp.stdout.splitlines():
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) != 2 or parts[0] != self.gpu_uuid:
                continue
            try:
                used = float(parts[1])
            except ValueError:
                continue
            self.peak_total_used_mib = max(self.peak_total_used_mib, used)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        try:
            self._sample()
        except Exception:
            pass

    @property
    def peak_delta_mib(self) -> float:
        return max(0.0, self.peak_total_used_mib - self.baseline_used_mib)


def run_call(
    llm: Any,
    sampling: Any,
    image_path: Path,
    prompt: str,
    label: str,
    raw_path: Path,
) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    print(json.dumps({"event": "QWEN_CALL_START", "label": label}, sort_keys=True), flush=True)
    started = time.perf_counter()
    outputs = llm.chat(
        messages,
        sampling_params=sampling,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    seconds = time.perf_counter() - started
    generated = outputs[0].outputs[0]
    text = generated.text
    parsed, parse_error = parse_json_object(text)
    payload = {
        "label": label,
        "seconds": seconds,
        "generated_text": text,
        "structured_output_valid": parsed is not None,
        "parse_error": parse_error,
        "parsed": parsed,
        "output_token_count": len(generated.token_ids),
    }
    raw_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "QWEN_CALL_COMPLETE",
                "label": label,
                "seconds": seconds,
                "structured_output_valid": parsed is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--figure6", type=Path, required=True)
    p.add_argument("--figure3", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--baseline-gpu-used-mib", type=float, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor = GpuDeltaMonitor(args.gpu_uuid, args.baseline_gpu_used_mib)
    monitor.start()
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "qwen3_5_4b_targeted_visual_fallback_measurement",
        "candidate_family": "qwen3.5-4b",
        "candidate_inference_calls": 0,
        "scientific_state_materialized": False,
        "quality_status": "PENDING_HUMAN_AND_CODEX_CONTROL",
        "failure": None,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
    }

    try:
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
        init_seconds = time.perf_counter() - started
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            seed=0,
            max_tokens=512,
        )

        runs = []
        for label, image, prompt, warm in [
            ("figure6_first", args.figure6, FIGURE6_PROMPT, False),
            ("figure3_warm", args.figure3, FIGURE3_PROMPT, True),
            ("figure6_warm_repeat", args.figure6, FIGURE6_PROMPT, True),
        ]:
            payload = run_call(
                llm,
                sampling,
                image,
                prompt,
                label,
                args.output_dir / f"{label}-raw.json",
            )
            report["candidate_inference_calls"] += 1
            payload["warm"] = warm
            runs.append(payload)

        warm_seconds = [x["seconds"] for x in runs if x["warm"]]
        report.update(
            {
                "model_init_seconds": init_seconds,
                "runs": runs,
                "warm_region_seconds_max": max(warm_seconds),
                "warm_region_seconds_mean": sum(warm_seconds) / len(warm_seconds),
                "peak_tree_rss_mib": monitor.peak_tree_rss_mib,
                "all_structured_outputs_valid": all(
                    x["structured_output_valid"] for x in runs
                ),
            }
        )
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor.stop()
        report["gpu_monitor_samples"] = monitor.samples
        report["baseline_gpu_used_mib"] = monitor.baseline_used_mib
        report["peak_total_gpu_used_mib"] = monitor.peak_total_used_mib
        report["peak_gpu_delta_mib"] = monitor.peak_delta_mib
        report["peak_tree_rss_mib"] = monitor.peak_tree_rss_mib

    ceilings = {
        "peak_vram_mib_lte": 14336.0,
        "peak_rss_mib_lte": 24576.0,
        "model_init_seconds_lte": 120.0,
        "warm_region_seconds_lte": 45.0,
    }
    checks = {
        "runtime_completed": report["failure"] is None,
        "exactly_three_inference_calls": report["candidate_inference_calls"] == 3,
        "gpu_telemetry_valid": report["gpu_monitor_samples"] > 0,
        "peak_vram": report.get("peak_gpu_delta_mib", float("inf"))
        <= ceilings["peak_vram_mib_lte"],
        "peak_rss": report.get("peak_tree_rss_mib", float("inf"))
        <= ceilings["peak_rss_mib_lte"],
        "model_init": report.get("model_init_seconds", float("inf"))
        <= ceilings["model_init_seconds_lte"],
        "warm_region": report.get("warm_region_seconds_max", float("inf"))
        <= ceilings["warm_region_seconds_lte"],
        "structured_json": report.get("all_structured_outputs_valid", False),
    }
    report["resource_ceilings"] = ceilings
    report["checks"] = checks
    report["measurement_decision"] = (
        "MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL"
        if all(checks.values())
        else "REJECT_RESOURCE_RUNTIME_OR_STRUCTURE"
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "measurement_decision": report["measurement_decision"],
                "candidate_inference_calls": report["candidate_inference_calls"],
                "model_init_seconds": report.get("model_init_seconds"),
                "warm_region_seconds_max": report.get("warm_region_seconds_max"),
                "peak_gpu_delta_mib": report.get("peak_gpu_delta_mib"),
                "peak_tree_rss_mib": report.get("peak_tree_rss_mib"),
                "all_structured_outputs_valid": report.get(
                    "all_structured_outputs_valid"
                ),
                "failure": report["failure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if report["measurement_decision"]
        == "MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
