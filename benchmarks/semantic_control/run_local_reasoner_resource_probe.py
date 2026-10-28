#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, resource, subprocess, threading, time, urllib.request

MIB=1024*1024

def gpu_memory(uuid:str)->int:
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],text=True,timeout=5)
    for line in raw.splitlines():
        u,m=[x.strip() for x in line.split(",",1)]
        if u==uuid:
            return int(m)
    raise RuntimeError(f"GPU UUID not found: {uuid}")

def download(url:str,dst:pathlib.Path,expected_sha:str,expected_bytes:int|None):
    h=hashlib.sha256(); total=0
    req=urllib.request.Request(url,headers={"User-Agent":"research-engine-resource-probe/1"})
    with urllib.request.urlopen(req,timeout=60) as src, dst.open("wb") as out:
        while True:
            chunk=src.read(8*MIB)
            if not chunk: break
            out.write(chunk); h.update(chunk); total+=len(chunk)
    got=h.hexdigest()
    if got!=expected_sha: raise RuntimeError(f"SHA256 mismatch {got} != {expected_sha}")
    if expected_bytes is not None and total!=expected_bytes: raise RuntimeError(f"byte count mismatch {total} != {expected_bytes}")
    return total

def run_envelope(llama:pathlib.Path, model:pathlib.Path, ctx:int, max_out:int, gpu_uuid:str, baseline:int, tmp:pathlib.Path):
    stop=threading.Event(); peak={"mib":baseline}
    def sample():
        while not stop.is_set():
            try: peak["mib"]=max(peak["mib"],gpu_memory(gpu_uuid))
            except Exception: pass
            stop.wait(.20)
    t=threading.Thread(target=sample,daemon=True); t.start()
    tag=f"ctx-{ctx}"
    timefile=tmp/f"{tag}.time.txt"
    cmd=["/usr/bin/time","-v","-o",str(timefile),str(llama),"-m",str(model),"-c",str(ctx),"-ngl","999","-n",str(max_out),"--temp","0","-p","Return exactly the token sequence RESOURCE_PROBE_OK and nothing else.","--no-display-prompt"]
    started=time.perf_counter()
    proc=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
    wall=time.perf_counter()-started
    stop.set(); t.join(timeout=2)
    text=timefile.read_text(encoding="utf-8",errors="replace") if timefile.exists() else ""
    m=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",text)
    rss_mib=(int(m.group(1))/1024.0) if m else None
    return {
      "context_tokens":ctx,"returncode":proc.returncode,"wall_seconds":wall,
      "peak_gpu_memory_mib":peak["mib"],"peak_gpu_delta_mib":max(0,peak["mib"]-baseline),
      "peak_rss_mib":rss_mib,"stdout_tail":proc.stdout[-4000:],"stderr_tail":proc.stderr[-6000:]
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--contract",type=pathlib.Path,required=True)
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--llama-cli",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    a=ap.parse_args()
    c=json.loads(a.contract.read_text())
    cand=next((x for x in c["candidates"] if x["id"]==a.candidate),None)
    if cand is None: raise SystemExit("unknown candidate")
    gpu_uuid=os.environ["P5_GPU_TARGET_UUID"]
    tmp=pathlib.Path(os.environ["RUNNER_TEMP"])
    model=tmp/cand["probe_artifact"]["file"]
    record={"schema_version":1,"experiment_id":c["experiment_id"],"candidate_id":a.candidate,"decision":"FAIL","failure_stage":"preflight","scientific_input_used":False,"scientific_state_materialized":False,"production_routing_changed":False,"external_llm_calls":0,"cloud_model_fallbacks":0}
    try:
        visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
        if visible!=gpu_uuid: raise RuntimeError(f"CUDA_VISIBLE_DEVICES mismatch: {visible}")
        baseline=gpu_memory(gpu_uuid); record["baseline_gpu_memory_mib"]=baseline
        if baseline>c["resource_gates"]["baseline_gpu_memory_mib_lte"]:
            record["decision"]="INVALID_SUBSTRATE"; record["failure_stage"]="baseline-gpu-residency"; return
        art=cand["probe_artifact"]
        url=f'https://huggingface.co/{art["repository"]}/resolve/{art["revision"]}/{art["file"]}?download=true'
        record["failure_stage"]="download"
        record["download_bytes"]=download(url,model,art["sha256"],art.get("bytes"))
        record["artifact_sha256"]=art["sha256"]
        rows=[]
        for env in c["inference_envelopes"]:
            record["failure_stage"]=env["id"]
            rows.append(run_envelope(a.llama_cli,model,int(env["context_tokens"]),int(env["max_output_tokens"]),gpu_uuid,baseline,tmp))
            if rows[-1]["returncode"]!=0: raise RuntimeError(f'{env["id"]} llama-cli rc={rows[-1]["returncode"]}')
            if rows[-1]["wall_seconds"]>c["resource_gates"]["hard_each_envelope_wall_seconds_lte"]: raise RuntimeError(f'{env["id"]} wall ceiling')
            if rows[-1]["peak_gpu_delta_mib"]>c["resource_gates"]["hard_peak_gpu_delta_mib_lte"]: raise RuntimeError(f'{env["id"]} GPU ceiling')
            if rows[-1]["peak_rss_mib"] is not None and rows[-1]["peak_rss_mib"]>c["resource_gates"]["hard_peak_rss_mib_lte"]: raise RuntimeError(f'{env["id"]} RSS ceiling')
        record["envelopes"]=rows
        record["decision"]="RESOURCE_FEASIBLE"; record["failure_stage"]=None
    except BaseException as e:
        record["failure_reason"]=f"{type(e).__name__}: {e}"
        raise
    finally:
        try:
            if model.exists(): model.unlink()
        finally:
            a.output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            print(json.dumps(record,indent=2,sort_keys=True))
if __name__=="__main__":
    main()
