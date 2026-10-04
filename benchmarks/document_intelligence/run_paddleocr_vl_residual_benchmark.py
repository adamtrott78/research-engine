#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any




def install_transformers_causal_mask_compat() -> str:
    from transformers import masking_utils

    original = masking_utils.create_causal_mask
    params = set(inspect.signature(original).parameters)
    if "inputs_embeds" in params:
        return "native_inputs_embeds"
    if "input_embeds" not in params:
        raise RuntimeError(
            "Unexpected transformers create_causal_mask signature: "
            + str(inspect.signature(original))
        )

    def compat(*args: Any, **kwargs: Any) -> Any:
        if "inputs_embeds" in kwargs:
            if "input_embeds" in kwargs:
                raise TypeError(
                    "create_causal_mask received both inputs_embeds and input_embeds"
                )
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = compat
    return "inputs_embeds_to_input_embeds"


class GpuProcessMonitor:
    def __init__(self, gpu_uuid: str, interval: float = 0.2):
        self.gpu_uuid = gpu_uuid
        self.interval = interval
        self.pid = os.getpid()
        self.peak_mib = 0.0
        self.samples = 0
        self.process_samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cp = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                self.samples += 1
                for raw in cp.stdout.splitlines():
                    parts = [p.strip() for p in raw.split(",")]
                    if len(parts) != 3:
                        continue
                    try:
                        pid = int(parts[0])
                        used = float(parts[2])
                    except ValueError:
                        continue
                    if pid == self.pid and parts[1] == self.gpu_uuid:
                        self.process_samples += 1
                        self.peak_mib = max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval)


def run_generate(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt_label: str,
    raw_path: Path,
) -> tuple[float, dict[str, Any]]:
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_label},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to("cuda:0")
    input_len = inputs["input_ids"].shape[-1]

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            use_cache=False,
        )
    seconds = time.perf_counter() - started

    generated_only = generated[:, input_len:]
    decoded = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = {
        "image": str(image_path),
        "prompt_label": prompt_label,
        "generated_text": decoded,
        "generated_token_count": int(generated_only.shape[-1]),
        "seconds": seconds,
    }
    raw_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return seconds, payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--figure6", type=Path, required=True)
    p.add_argument("--figure3", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor = GpuProcessMonitor(args.gpu_uuid)
    monitor.start()
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "paddleocr_vl_1_6_targeted_visual_region_measurement",
        "candidate_family": "paddleocr-vl-1.6",
        "loader": "AutoModelForCausalLM+AutoProcessor",
        "candidate_inference_calls": 0,
        "scientific_state_materialized": False,
        "quality_status": "PENDING_HUMAN_AND_CODEX_CONTROL",
        "failure": None,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA unavailable")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("Torch sees no CUDA devices")

        mask_compat = install_transformers_causal_mask_compat()
        report["causal_mask_compat"] = mask_compat

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            str(args.model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).eval().to("cuda:0")
        init_seconds = time.perf_counter() - started

        runs = []
        for label, image, prompt_label, warm in [
            ("figure6_first", args.figure6, "Chart Recognition:", False),
            ("figure3_warm", args.figure3, "OCR:", True),
            ("figure6_warm_repeat", args.figure6, "Chart Recognition:", True),
        ]:
            seconds, payload = run_generate(
                model,
                processor,
                image,
                prompt_label,
                args.output_dir / f"{label}-raw.json",
            )
            report["candidate_inference_calls"] += 1
            payload.update({"label": label, "warm": warm})
            runs.append(payload)

        warm = [r["seconds"] for r in runs if r["warm"]]
        report.update(
            {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_device_count_visible": torch.cuda.device_count(),
                "model_init_seconds": init_seconds,
                "runs": runs,
                "warm_region_seconds_max": max(warm),
                "warm_region_seconds_mean": statistics.fmean(warm),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
            }
        )
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor.stop()
        report["gpu_monitor_samples"] = monitor.samples
        report["gpu_process_samples"] = monitor.process_samples
        report["peak_process_vram_mib"] = monitor.peak_mib

    ceilings = {
        "peak_vram_mib_lte": 14336.0,
        "peak_rss_mib_lte": 24576.0,
        "model_init_seconds_lte": 120.0,
        "warm_region_seconds_lte": 45.0,
    }
    checks = {
        "runtime_completed": report["failure"] is None,
        "exactly_three_inference_calls": report["candidate_inference_calls"] == 3,
        "gpu_telemetry_valid": report["gpu_process_samples"] > 0,
        "peak_vram": report.get("peak_process_vram_mib", float("inf"))
        <= ceilings["peak_vram_mib_lte"],
        "peak_rss": report.get("peak_rss_mib", float("inf"))
        <= ceilings["peak_rss_mib_lte"],
        "model_init": report.get("model_init_seconds", float("inf"))
        <= ceilings["model_init_seconds_lte"],
        "warm_region": report.get("warm_region_seconds_max", float("inf"))
        <= ceilings["warm_region_seconds_lte"],
    }
    report["resource_ceilings"] = ceilings
    report["checks"] = checks
    report["measurement_decision"] = (
        "MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL"
        if all(checks.values())
        else "REJECT_RESOURCE_OR_RUNTIME"
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "measurement_decision": report["measurement_decision"],
                "candidate_inference_calls": report["candidate_inference_calls"],
                "loader": report["loader"],
                "model_init_seconds": report.get("model_init_seconds"),
                "warm_region_seconds_max": report.get("warm_region_seconds_max"),
                "peak_process_vram_mib": report.get("peak_process_vram_mib"),
                "peak_rss_mib": report.get("peak_rss_mib"),
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
