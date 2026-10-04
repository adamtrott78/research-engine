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
        self.gpu_uuid=gpu_uuid
        self.interval=interval
        self.pid=os.getpid()
        self.peak_mib=0.0
        self.samples=0
        self.process_samples=0
        self._stop=threading.Event()
        self._thread=threading.Thread(target=self._run,daemon=True)

    def start(self)->None:
        self._thread.start()

    def stop(self)->None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self)->None:
        while not self._stop.is_set():
            try:
                cp=subprocess.run(
                    ["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader,nounits"],
                    check=False,capture_output=True,text=True,timeout=2,
                )
                self.samples += 1
                for raw in cp.stdout.splitlines():
                    parts=[p.strip() for p in raw.split(",")]
                    if len(parts)!=3:
                        continue
                    try:
                        pid=int(parts[0]); used=float(parts[2])
                    except ValueError:
                        continue
                    if pid==self.pid and parts[1]==self.gpu_uuid:
                        self.process_samples += 1
                        self.peak_mib=max(self.peak_mib,used)
            except Exception:
                pass
            self._stop.wait(self.interval)


def result_json(result: Any)->dict[str,Any]:
    payload=getattr(result,"json",None)
    if callable(payload):
        payload=payload()
    if isinstance(payload,str):
        payload=json.loads(payload)
    if not isinstance(payload,dict):
        raise RuntimeError(f"unexpected PaddleOCR-VL result payload: {type(payload)!r}")
    return jsonable(payload)


def flatten_strings(value: Any)->list[str]:
    out=[]
    if isinstance(value,str):
        if value.strip():
            out.append(value.strip())
    elif isinstance(value,dict):
        for v in value.values():
            out.extend(flatten_strings(v))
    elif isinstance(value,(list,tuple)):
        for v in value:
            out.extend(flatten_strings(v))
    return out


def run_predict(pipeline: Any, image: Path, raw_path: Path)->tuple[float,dict[str,Any]]:
    started=time.perf_counter()
    results=list(pipeline.predict(str(image)))
    seconds=time.perf_counter()-started
    if len(results)!=1:
        raise RuntimeError(f"expected one result for {image}, got {len(results)}")
    payload=result_json(results[0])
    raw_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    strings=flatten_strings(payload)
    return seconds,{
        "image":str(image),
        "seconds":seconds,
        "raw_json_path":str(raw_path),
        "string_fragment_count":len(strings),
        "text_projection":"\n".join(strings),
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--figure6",type=Path,required=True)
    p.add_argument("--figure3",type=Path,required=True)
    p.add_argument("--gpu-uuid",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()

    from paddleocr import PaddleOCRVL
    import torch
    import transformers

    args.output_dir.mkdir(parents=True,exist_ok=True)
    monitor=GpuProcessMonitor(args.gpu_uuid)
    monitor.start()
    report={
        "schema_version":1,
        "purpose":"paddleocr_vl_1_6_targeted_visual_region_measurement",
        "candidate_family":"paddleocr-vl-1.6",
        "candidate_inference_calls":0,
        "scientific_state_materialized":False,
        "quality_status":"PENDING_HUMAN_AND_CODEX_CONTROL",
        "failure":None,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA unavailable")
        if torch.cuda.device_count()<1:
            raise RuntimeError("Torch sees no CUDA devices")

        started=time.perf_counter()
        pipeline=PaddleOCRVL(
            pipeline_version="v1.6",
            engine="transformers",
            vl_rec_model_dir=str(args.model_dir),
            use_layout_detection=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_queues=False,
            device="gpu:0",
        )
        init_seconds=time.perf_counter()-started

        runs=[]
        for label,image,warm in [
            ("figure6_first",args.figure6,False),
            ("figure3_warm",args.figure3,True),
            ("figure6_warm_repeat",args.figure6,True),
        ]:
            seconds,payload=run_predict(
                pipeline,image,args.output_dir/f"{label}-raw.json"
            )
            report["candidate_inference_calls"] += 1
            payload.update({"label":label,"warm":warm})
            runs.append(payload)

        warm=[r["seconds"] for r in runs if r["warm"]]
        report.update({
            "torch_version":torch.__version__,
            "transformers_version":transformers.__version__,
            "cuda_device_count_visible":torch.cuda.device_count(),
            "model_init_seconds":init_seconds,
            "runs":runs,
            "warm_region_seconds_max":max(warm),
            "warm_region_seconds_mean":statistics.fmean(warm),
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,
        })
    except Exception as exc:
        report["failure"]=f"{type(exc).__name__}: {exc}"
    finally:
        monitor.stop()
        report["gpu_monitor_samples"]=monitor.samples
        report["gpu_process_samples"]=monitor.process_samples
        report["peak_process_vram_mib"]=monitor.peak_mib

    ceilings={
        "peak_vram_mib_lte":14336.0,
        "peak_rss_mib_lte":24576.0,
        "model_init_seconds_lte":120.0,
        "warm_region_seconds_lte":45.0,
    }
    checks={
        "runtime_completed":report["failure"] is None,
        "exactly_three_inference_calls":report["candidate_inference_calls"]==3,
        "gpu_telemetry_valid":report["gpu_process_samples"]>0,
        "peak_vram":report.get("peak_process_vram_mib",float("inf"))<=ceilings["peak_vram_mib_lte"],
        "peak_rss":report.get("peak_rss_mib",float("inf"))<=ceilings["peak_rss_mib_lte"],
        "model_init":report.get("model_init_seconds",float("inf"))<=ceilings["model_init_seconds_lte"],
        "warm_region":report.get("warm_region_seconds_max",float("inf"))<=ceilings["warm_region_seconds_lte"],
    }
    report["resource_ceilings"]=ceilings
    report["checks"]=checks
    report["measurement_decision"]=(
        "MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL"
        if all(checks.values())
        else "REJECT_RESOURCE_OR_RUNTIME"
    )
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "measurement_decision":report["measurement_decision"],
        "candidate_inference_calls":report["candidate_inference_calls"],
        "model_init_seconds":report.get("model_init_seconds"),
        "warm_region_seconds_max":report.get("warm_region_seconds_max"),
        "peak_process_vram_mib":report.get("peak_process_vram_mib"),
        "peak_rss_mib":report.get("peak_rss_mib"),
        "failure":report["failure"],
    },indent=2,sort_keys=True))
    return 0 if report["measurement_decision"]=="MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL" else 2


if __name__=="__main__":
    raise SystemExit(main())
