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

from research_engine.openai_executor import COMMON_INSTRUCTIONS, JOB_INSTRUCTIONS, result_schema
from research_engine.orchestration import JobType


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    value=text.strip(); start=value.find("{"); end=value.rfind("}")
    if start<0 or end<start: return None,"no_json_object"
    try: parsed=json.loads(value[start:end+1])
    except Exception as exc: return None,f"{type(exc).__name__}: {exc}"
    return (parsed,None) if isinstance(parsed,dict) else (None,"json_not_object")


class GpuDeltaMonitor:
    def __init__(self,gpu_uuid:str,baseline_used_mib:float,interval:float=0.2):
        self.gpu_uuid=gpu_uuid; self.baseline_used_mib=baseline_used_mib; self.interval=interval
        self.peak_total_used_mib=baseline_used_mib; self.peak_tree_rss_mib=0.0; self.samples=0
        self._stop=threading.Event(); self._thread=threading.Thread(target=self._run,daemon=True)
    def _sample(self):
        import psutil
        root=psutil.Process(os.getpid()); rss=0
        for proc in [root]+root.children(recursive=True):
            try: rss+=proc.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        self.peak_tree_rss_mib=max(self.peak_tree_rss_mib,rss/(1024**2))
        cp=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=False,timeout=2)
        for raw in cp.stdout.splitlines():
            parts=[x.strip() for x in raw.split(",")]
            if len(parts)==2 and parts[0]==self.gpu_uuid:
                try: self.peak_total_used_mib=max(self.peak_total_used_mib,float(parts[1]))
                except ValueError: pass
    def _run(self):
        while not self._stop.is_set():
            try: self._sample(); self.samples+=1
            except Exception: pass
            self._stop.wait(self.interval)
    def start(self): self._thread.start()
    def stop(self):
        self._stop.set(); self._thread.join(timeout=3)
        try: self._sample()
        except Exception: pass
    @property
    def peak_delta_mib(self): return max(0.0,self.peak_total_used_mib-self.baseline_used_mib)


def prompt_for(row:dict[str,Any]):
    record=dict(row["job"]); job_type=JobType(str(record["job_type"])); schema=result_schema(job_type)
    system=(COMMON_INSTRUCTIONS+"\n"+JOB_INSTRUCTIONS[job_type]+
      "\nThis is a bounded local calibration. Use ONLY payload._research_context. "
      "For EXTRACT, raw_source_form must be copied verbatim from the cited page. "
      "For VERIFY, accepted_evidence_ids must come only from authoritative_evidence. "
      "If uncertain, return empty evidence/accepted IDs or explicit issues rather than guessing. "
      "Return exactly one JSON object matching the supplied schema.")
    user=json.dumps({"job_id":record["id"],"job_type":str(job_type),"payload":record["payload"],"output_schema":schema},sort_keys=True)
    return [{"role":"system","content":system},{"role":"user","content":user}],job_type


def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True); p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--gpu-uuid",required=True); p.add_argument("--baseline-gpu-used-mib",type=float,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    from vllm import LLM,SamplingParams
    import vllm
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    if len(plan.get("jobs") or [])!=2: raise SystemExit("plan must contain exactly two jobs")
    if {row["job"]["job_type"] for row in plan["jobs"]}!={"EXTRACT","VERIFY"}: raise SystemExit("plan must contain exactly one EXTRACT and one VERIFY")
    monitor=GpuDeltaMonitor(args.gpu_uuid,args.baseline_gpu_used_mib); monitor.start()
    report={"schema_version":1,"operation":"QWEN35_GUARDED_EXTRACT_VERIFY_CALIBRATION","candidate_family":"qwen3.5-4b","candidate_inference_calls":0,"jobs":[],"failure":None,"vllm_version":getattr(vllm,"__version__","unknown"),"scientific_state_materialized":False,"production_routing_changed":False}
    try:
        print(json.dumps({"event":"QWEN_MODEL_INIT_START"},sort_keys=True),flush=True); start=time.perf_counter()
        llm=LLM(model=str(args.model_dir),tensor_parallel_size=1,dtype="bfloat16",max_model_len=8192,max_num_seqs=1,gpu_memory_utilization=0.65,limit_mm_per_prompt={"image":1},trust_remote_code=False,disable_log_stats=True,mm_processor_cache_gb=0.25)
        init=time.perf_counter()-start; print(json.dumps({"event":"QWEN_MODEL_INIT_COMPLETE","seconds":init},sort_keys=True),flush=True)
        sampling=SamplingParams(temperature=0.0,top_p=1.0,seed=0,max_tokens=2048)
        for row in plan["jobs"]:
            messages,job_type=prompt_for(row); job_id=str(row["job"]["id"])
            print(json.dumps({"event":"QWEN_CALL_START","job_id":job_id,"job_type":str(job_type)},sort_keys=True),flush=True)
            started=time.perf_counter(); outputs=llm.chat(messages,sampling_params=sampling,use_tqdm=False,chat_template_kwargs={"enable_thinking":False}); seconds=time.perf_counter()-started
            generated=outputs[0].outputs[0]; parsed,error=parse_json_object(generated.text); report["candidate_inference_calls"]+=1
            report["jobs"].append({"job_id":job_id,"job_type":str(job_type),"seconds":seconds,"generated_text":generated.text,"parsed":parsed,"parse_error":error,"structured_output_valid":parsed is not None,"output_token_count":len(generated.token_ids)})
            print(json.dumps({"event":"QWEN_CALL_COMPLETE","job_id":job_id,"job_type":str(job_type),"seconds":seconds,"structured_output_valid":parsed is not None},sort_keys=True),flush=True)
        report["model_init_seconds"]=init
    except Exception as exc: report["failure"]=f"{type(exc).__name__}: {exc}"
    finally:
        monitor.stop(); report["gpu_monitor_samples"]=monitor.samples; report["baseline_gpu_used_mib"]=monitor.baseline_used_mib; report["peak_total_gpu_used_mib"]=monitor.peak_total_used_mib; report["peak_gpu_delta_mib"]=monitor.peak_delta_mib; report["peak_tree_rss_mib"]=monitor.peak_tree_rss_mib
    by_type={row["job_type"]:row["seconds"] for row in report["jobs"]}
    checks={"runtime_completed":report["failure"] is None,"exactly_two_inference_calls":report["candidate_inference_calls"]==2,"structured_json":len(report["jobs"])==2 and all(row["structured_output_valid"] for row in report["jobs"]),"gpu_telemetry_valid":report["gpu_monitor_samples"]>0,"peak_vram":report["peak_gpu_delta_mib"]<=16384,"peak_rss":report["peak_tree_rss_mib"]<=32768,"model_init":report.get("model_init_seconds",float("inf"))<=120,"extract_latency":by_type.get("EXTRACT",float("inf"))<=60,"verify_latency":by_type.get("VERIFY",float("inf"))<=45}
    report["checks"]=checks; report["resource_ceilings"]={"peak_vram_mib_lte":16384,"peak_rss_mib_lte":32768,"model_init_seconds_lte":120,"extract_job_seconds_lte":60,"verify_job_seconds_lte":45}
    report["measurement_decision"]="MEASURED_PENDING_GUARD_AND_CODEX_CONTROL" if all(checks.values()) else "REJECT_RESOURCE_RUNTIME_OR_STRUCTURE"
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"measurement_decision":report["measurement_decision"],"candidate_inference_calls":report["candidate_inference_calls"],"model_init_seconds":report.get("model_init_seconds"),"peak_gpu_delta_mib":report["peak_gpu_delta_mib"],"peak_tree_rss_mib":report["peak_tree_rss_mib"],"jobs":[{"job_id":x["job_id"],"job_type":x["job_type"],"seconds":x["seconds"],"structured_output_valid":x["structured_output_valid"]} for x in report["jobs"]],"failure":report["failure"]},indent=2,sort_keys=True))
    return 0 if report["measurement_decision"]=="MEASURED_PENDING_GUARD_AND_CODEX_CONTROL" else 2

if __name__=="__main__": raise SystemExit(main())
