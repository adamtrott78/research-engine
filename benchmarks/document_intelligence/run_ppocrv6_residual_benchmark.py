#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
        self._thread.join(timeout=2)

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


def extract_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected PaddleOCR result payload type: {type(payload)!r}")
    payload = jsonable(payload)
    if "res" in payload and isinstance(payload["res"], dict):
        return payload["res"]
    return payload


def run_predict(ocr: Any, image: Path) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    results = list(ocr.predict(str(image)))
    seconds = time.perf_counter() - started
    if len(results) != 1:
        raise RuntimeError(f"expected one OCR result for {image}, got {len(results)}")
    raw = extract_payload(results[0])
    texts = [str(x) for x in raw.get("rec_texts", [])]
    scores = [float(x) for x in raw.get("rec_scores", [])]
    boxes = jsonable(raw.get("rec_boxes", []))
    return seconds, {
        "image": str(image),
        "recognized_line_count": len(texts),
        "recognized_nonempty_line_count": sum(bool(x.strip()) for x in texts),
        "recognized_text": "\n".join(texts),
        "rec_texts": texts,
        "rec_scores": scores,
        "rec_boxes": boxes,
        "mean_rec_score": statistics.fmean(scores) if scores else None,
        "minimum_rec_score": min(scores) if scores else None,
        "maximum_rec_score": max(scores) if scores else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--det-model", type=Path, required=True)
    p.add_argument("--rec-model", type=Path, required=True)
    p.add_argument("--scan-image", type=Path, required=True)
    p.add_argument("--low-confidence-image", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from paddleocr import PaddleOCR
    import paddle

    monitor = GpuProcessMonitor(args.gpu_uuid)
    monitor.start()
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "ppocrv6_residual_scan_text_measurement",
        "candidate_family": "paddleocr-pp-ocrv6-medium",
        "candidate_inference_calls": 0,
        "scientific_state_materialized": False,
        "quality_status": "PENDING_HUMAN_REVIEW",
        "failure": None,
    }
    try:
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError("PaddlePaddle runtime is not CUDA-enabled")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("PaddlePaddle sees no CUDA devices")

        started = time.perf_counter()
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_detection_model_dir=str(args.det_model),
            text_recognition_model_name="PP-OCRv6_medium_rec",
            text_recognition_model_dir=str(args.rec_model),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0",
            engine="paddle_static",
            enable_hpi=False,
        )
        init_seconds = time.perf_counter() - started

        runs = []
        for label, image, warm in [
            ("scan_textless_first", args.scan_image, False),
            ("low_confidence_text_warm", args.low_confidence_image, True),
            ("scan_textless_warm_repeat", args.scan_image, True),
        ]:
            seconds, payload = run_predict(ocr, image)
            report["candidate_inference_calls"] += 1
            payload.update({"label": label, "seconds": seconds, "warm": warm})
            runs.append(payload)

        warm_seconds = [r["seconds"] for r in runs if r["warm"]]
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        report.update(
            {
                "paddle_version": paddle.__version__,
                "cuda_device_count_visible": paddle.device.cuda.device_count(),
                "model_init_seconds": init_seconds,
                "runs": runs,
                "warm_page_seconds_max": max(warm_seconds),
                "warm_page_seconds_mean": statistics.fmean(warm_seconds),
                "peak_rss_mib": rss_mib,
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
        "peak_vram_mib_lte": 8192.0,
        "peak_rss_mib_lte": 16384.0,
        "model_init_seconds_lte": 60.0,
        "warm_page_seconds_lte": 10.0,
    }
    checks = {
        "runtime_completed": report["failure"] is None,
        "exactly_three_inference_calls": report["candidate_inference_calls"] == 3,
        "gpu_telemetry_valid": report["gpu_process_samples"] > 0,
        "peak_vram": report.get("peak_process_vram_mib", float("inf")) <= ceilings["peak_vram_mib_lte"],
        "peak_rss": report.get("peak_rss_mib", float("inf")) <= ceilings["peak_rss_mib_lte"],
        "model_init": report.get("model_init_seconds", float("inf")) <= ceilings["model_init_seconds_lte"],
        "warm_page": report.get("warm_page_seconds_max", float("inf")) <= ceilings["warm_page_seconds_lte"],
    }
    report["resource_ceilings"] = ceilings
    report["checks"] = checks
    report["measurement_decision"] = (
        "MEASURED_PENDING_HUMAN_QUALITY"
        if all(checks.values())
        else "REJECT_RESOURCE_OR_RUNTIME"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "measurement_decision": report["measurement_decision"],
        "candidate_inference_calls": report["candidate_inference_calls"],
        "quality_status": report["quality_status"],
        "model_init_seconds": report.get("model_init_seconds"),
        "warm_page_seconds_max": report.get("warm_page_seconds_max"),
        "peak_process_vram_mib": report.get("peak_process_vram_mib"),
        "peak_rss_mib": report.get("peak_rss_mib"),
        "failure": report["failure"],
    }, indent=2, sort_keys=True))
    return 0 if report["measurement_decision"] == "MEASURED_PENDING_HUMAN_QUALITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
