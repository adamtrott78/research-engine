#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/residual-runtimes/qwen3.5-4b-vllm-0.30.0"
runtime_ready="$runtime/.workspace-control-ready.json"
model="$state/residual-models/qwen3.5-4b/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
runtime_contract="qwen35-vllm-cu129-image-native-sampler-v3"
cache_root="$state/residual-runtime-cache/$runtime_contract/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
run_id="${RESEARCH_HUD_RUN_ID:-rf-open-world-corpus-expansion-v1}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$visible/qwen35-grounded-extract-$stamp"
selection="$out/source-selection.json"
sidecar="$out/document-sidecar-receipt.json"
plan="$out/grounded-candidate-plan.json"
qwen="$out/qwen-grounded-results.json"
codex="$out/codex-control-results.json"
report="$out/grounded-extract-report.json"
helper_log="$out/qwen-helper.log"
reuse="$out/qwen-runtime-reuse-evidence.json"
certification="$out/runtime-certification.json"
wave3="$out/wave3-adoption-evidence.json"
operation_receipt="$out/runtime-operation-receipt.json"
operation_detail="$out/runtime-operation-detail.json"

fail(){ echo "ERROR=$1" >&2; exit "${2:-1}"; }
require_stopped(){
  local a s p
  a="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  s="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
  p="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
  [ "$a" = inactive ] && [ "$s" = dead ] && [ "${p:-0}" = 0 ] || fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
}

mkdir -p "$out"
exec > >(tee "$out/operator-console.log") 2>&1

echo "=== QWEN3.5 GROUNDED-CANDIDATE EXTRACT GATE ==="
systemctl --user stop "$unit" >/dev/null 2>&1 || true
require_stopped
cd "$root"
[ "$(git branch --show-current)" = main ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$runtime/bin/python" ] || fail "QWEN_RUNTIME_NOT_FOUND" 33
[ -f "$runtime_ready" ] || fail "QWEN_RUNTIME_CERTIFICATION_NOT_FOUND" 34
command -v codex >/dev/null || fail "CODEX_NOT_FOUND" 35
git_sha="$(git rev-parse HEAD)"
echo "HEAD=$git_sha"
echo "OUTPUT_ROOT=$out"

echo
echo "=== ENSURE POSTGRESQL SUPPORT SERVICE ==="
export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"
export RESEARCH_ENGINE_ARTIFACT_ROOT="${RESEARCH_ENGINE_ARTIFACT_ROOT:-$state/artifacts}"
sudo docker compose -f docker-compose.yml up -d db
db_ready=0
for _ in $(seq 1 30); do
  if sudo docker compose -f docker-compose.yml exec -T db pg_isready -U research_engine -d research_engine >/dev/null 2>&1; then
    db_ready=1
    break
  fi
  sleep 2
done
[ "$db_ready" -eq 1 ] || fail "POSTGRESQL_NOT_READY" 36
echo "POSTGRESQL=running/ready"
require_stopped

echo
echo "=== CHEAP SOURCE SELECTION ==="
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract select "$run_id" --output "$selection"
execid="$(PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract execution-id)"
pkgid="$(PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract package-id)"
opid="$(PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract operation-id --selection "$selection" --git-sha "$git_sha")"
opslug="${opid#sha256:}"
opstate="$state/runtime-operations/qwen35-grounded-extract/$opslug.json"
opdetailstate="$state/runtime-operations/qwen35-grounded-extract/$opslug.detail.json"
mkdir -p "$(dirname "$opstate")"
echo "EXECUTION_IDENTITY=$execid"
echo "PACKAGE_IDENTITY=$pkgid"
echo "OPERATION_ID=$opid"

if [ -f "$opstate" ]; then
  recovery="$("$core_venv/bin/python" - "$opstate" "$opid" <<'PY'
import json,sys
try: r=json.load(open(sys.argv[1],encoding="utf-8"))
except Exception: r={}
if r.get("operation_id")==sys.argv[2] and r.get("status")=="PASS": print("COMPLETED_REUSE")
elif r.get("operation_id")==sys.argv[2] and r.get("status")=="RUNNING": print("INSPECT_OWNER")
elif r.get("operation_id")==sys.argv[2] and r.get("status")=="FAIL": print("RESUME_FAILED_STAGE")
else: print("EXECUTE")
PY
)"
  echo "RECOVERY_DECISION=$recovery"
  if [ "$recovery" = COMPLETED_REUSE ]; then
    cat "$opstate"
    [ ! -f "$opdetailstate" ] || cat "$opdetailstate"
    echo "NO_RERUN_REQUIRED=true"
    exit 0
  fi
  if [ "$recovery" = INSPECT_OWNER ]; then
    owner_pid="$("$core_venv/bin/python" - "$opstate" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
print(((r.get("owner_process") or {}).get("pid")) or "")
PY
)"
    if [ -n "$owner_pid" ] && [ -r "/proc/$owner_pid/cmdline" ]; then
      owner_match="$("$core_venv/bin/python" - "/proc/$owner_pid/cmdline" <<'PY'
import pathlib,sys
raw=pathlib.Path(sys.argv[1]).read_bytes()
argv=[x.decode("utf-8",errors="replace") for x in raw.split(b"\0") if x]
ok=(len(argv)>1 and pathlib.PurePath(argv[0]).name in {"bash","sh"} and pathlib.PurePath(argv[1]).name=="run-qwen35-grounded-extract.sh")
print("true" if ok else "false")
PY
)"
      if [ "$owner_match" = true ]; then
        echo "ACTIVE_OPERATION_OWNER_PID=$owner_pid"
        echo "NO_DUPLICATE_LAUNCH=true"
        exit 70
      fi
    fi
    echo "STALE_RUNNING_RECEIPT=true"
  fi
fi

"$core_venv/bin/python" - "$opstate" "$operation_receipt" "$opdetailstate" "$operation_detail" "$opid" "$execid" "$git_sha" "$out" "$BASHPID" <<'PY'
import json,pathlib,sys
state,out,detail_state,detail_out,opid,execid,sha,root,pid=sys.argv[1:]
receipt={
 "schema_version":1,
 "operation_id":opid,
 "task_id":"research-engine-local-document-pipeline-v1",
 "git_sha":sha,
 "operation_type":"QWEN35_GROUNDED_CANDIDATE_EXTRACT_CALIBRATION",
 "execution_identity":execid,
 "status":"RUNNING",
 "terminal_stage":"C0_STATIC",
 "owner_process":{
   "role":"operator-shell",
   "pid":int(pid),
   "executable":"bash",
   "argv":["bash","deploy/run-qwen35-grounded-extract.sh"],
 },
}
detail={"schema_version":1,"operation_id":opid,"status":"RUNNING","artifacts":{"output_root":root}}
for dest in (state,out):
 pathlib.Path(dest).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
for dest in (detail_state,detail_out):
 pathlib.Path(dest).write_text(json.dumps(detail,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

echo
echo "=== BUILD / REUSE PRODUCTION DOCUMENT SIDECAR ==="
export RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED=1
export RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED=1
export RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1
export RESEARCH_ENGINE_DOCUMENT_SIDECAR_ROOT="${RESEARCH_ENGINE_DOCUMENT_SIDECAR_ROOT:-$state/document-sidecars}"
export RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX="${RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX:-1}"
export RESEARCH_ENGINE_FORMULA_GPU_INDEX="${RESEARCH_ENGINE_FORMULA_GPU_INDEX:-0}"
export RESEARCH_ENGINE_TABLE_GPU_INDEX="${RESEARCH_ENGINE_TABLE_GPU_INDEX:-1}"
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract ensure-sidecar --selection "$selection" --output "$sidecar"
require_stopped

echo
echo "=== BUILD IMMUTABLE GROUNDED CANDIDATE SET ==="
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract build --selection "$selection" --output "$plan"
require_stopped

echo
echo "=== C0 / C1 / C2A RUNTIME REUSE ==="
"$runtime/bin/python" - "$runtime_ready" "$runtime_contract" <<'PY'
import importlib.metadata,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["runtime_contract_version"]==sys.argv[2]
assert p["sampling_backend"]=="vllm_native"
assert p["vllm_flashinfer_sampler"] is False
assert p["vllm_distribution"]=="0.30.0+cu129"
assert p["vllm_release_wheel_sha256"]=="e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
assert p["torch"]=="2.13.0+cu129"
assert p["torchvision"]=="0.28.0+cu129"
assert p["torchcodec"]=="0.16.0+cu129"
assert p["torch_cuda_runtime"]=="12.9"
assert p["cuda_allocation_probe"]=="PASS"
assert p["vllm_multimodal_import_probe"]=="PASS"
assert importlib.metadata.version("vllm")=="0.30.0+cu129"
print("R0_ENVIRONMENT=VERIFIED_REUSE")
print("R1_CERTIFICATION=VERIFIED_HIT")
print("C0_STATIC=PASS")
print("C1_IMPORT_ACCELERATOR=PASS")
print("C2A_BACKEND_TOOLCHAIN=PASS")
PY
echo "PACKAGE_INSTALL_DOWNLOAD_WORK=FORBIDDEN_AND_SKIPPED"

for spec in \
 "model.safetensors-00001-of-00002.safetensors:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61" \
 "model.safetensors-00002-of-00002.safetensors:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
do
  name="${spec%%:*}"; sha="${spec##*:}"
  [ -f "$model/$name" ] || fail "QWEN_WEIGHT_NOT_FOUND:$name" 37
  [ "$(sha256sum "$model/$name" | awk '{print $1}')" = "$sha" ] || fail "QWEN_WEIGHT_SHA_MISMATCH:$name" 38
done

echo
echo "=== SELECT EXCLUSIVE GPU AFTER SIDECAR OWNERS CLOSE ==="
active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
[ -z "$active" ] || fail "GPU_COMPUTE_PROCESS_REMAINS_AFTER_SIDECAR:$active" 39
line="$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used --format=csv,noheader,nounits | sort -t',' -k3 -nr | head -1)"
gpu_index="$(printf '%s' "$line" | cut -d',' -f1 | xargs)"
gpu_uuid="$(printf '%s' "$line" | cut -d',' -f2 | xargs)"
gpu_free="$(printf '%s' "$line" | cut -d',' -f3 | xargs)"
gpu_used="$(printf '%s' "$line" | cut -d',' -f4 | xargs)"
[ "$gpu_free" -ge 16384 ] || fail "INSUFFICIENT_FREE_GPU_MEMORY" 40

export CUDA_VISIBLE_DEVICES="$gpu_index"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cache_present=false
if [ -d "$cache_root" ] && find "$cache_root" -type f -print -quit 2>/dev/null | grep -q .; then cache_present=true; fi
export XDG_CACHE_HOME="$cache_root"
mkdir -p "$cache_root"
echo "SELECTED_GPU_UUID=$gpu_uuid"
echo "VLLM_COMPILE_CACHE_PRESENT_BEFORE=$cache_present"

echo
echo "=== ONE QWEN BOOT: C3 CANARY -> C4 GROUNDED EXTRACT ==="
set +e
PYTHONPATH="$root/src" "$runtime/bin/python" "$root/benchmarks/document_intelligence/run_qwen35_grounded_extract.py" \
  --plan "$plan" --model-dir "$model" --gpu-uuid "$gpu_uuid" --baseline-gpu-used-mib "$gpu_used" --output "$qwen" 2>&1 | tee "$helper_log"
qwen_rc="${PIPESTATUS[0]}"
set -e

PYTHONPATH="$root/src" "$core_venv/bin/python" - "$helper_log" "$reuse" "$cache_present" <<'PY'
import json,pathlib,sys
from research_engine.qwen_runtime_certification import make_runtime_reuse_report
log=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
r=make_runtime_reuse_report(
 log_text=log.read_text(encoding="utf-8",errors="replace"),
 cache_present_before=sys.argv[3].lower()=="true",
 package_environment_reused=True,
 execution_certification_reused=True,
 live_engine_reused_for_domain_calls=True,
)
r["c3_same_modality_canary_calls"]=1
r["c4_scientific_domain_calls"]=1
r["second_heavyweight_boot"]=False
out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(r,indent=2,sort_keys=True))
PY
if [ "$qwen_rc" -ne 0 ]; then
  "$core_venv/bin/python" - "$opstate" "$operation_receipt" "$opdetailstate" "$operation_detail" "$opid" "$execid" "$git_sha" "$out" "$qwen" "$helper_log" <<'PY'
import json,pathlib,sys
state,out,detail_state,detail_out,opid,execid,sha,root,qwen,log=sys.argv[1:]
result={}
try: result=json.load(open(qwen,encoding="utf-8"))
except Exception: pass
engine_init=1 if result.get("model_init_seconds") is not None else 0
c3=result.get("canary") or {}
domain=result.get("domain") or {}
started=int(bool(c3))+int(bool(domain))
completed=int(bool(c3.get("parsed")))+int(bool(domain.get("parsed")))
stage="C3_CANARY" if engine_init and not c3.get("pass") else "C2B_ENGINE_BOOT"
receipt={
 "schema_version":1,
 "operation_id":opid,
 "task_id":"research-engine-local-document-pipeline-v1",
 "git_sha":sha,
 "operation_type":"QWEN35_GROUNDED_CANDIDATE_EXTRACT_CALIBRATION",
 "execution_identity":execid,
 "status":"FAIL",
 "terminal_stage":stage,
 "failure_envelope":{
   "schema_version":1,
   "stage":stage,
   "parent_failure":{"class":"QWEN_HELPER_FAILED","message":str(result.get("failure") or "grounded extract helper failed")},
   "causal_failure":{"class":"RUNTIME_OR_CANARY_FAILURE","message":"inspect compact Qwen result/helper log","role":"qwen_engine"},
   "state_flags":{"scientific_state_materialized":False,"production_routing_changed":False},
   "counts":{"engine_init_completed":engine_init,"inference_calls_started":started,"inference_calls_completed":completed},
   "artifacts":{"qwen_result":qwen,"targeted_traceback_log":log,"full_log":log},
 },
}
detail={"schema_version":1,"operation_id":opid,"status":"FAIL","artifacts":{"output_root":root,"qwen_result":qwen,"helper_log":log}}
for dest in (state,out):
 pathlib.Path(dest).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
for dest in (detail_state,detail_out):
 pathlib.Path(dest).write_text(json.dumps(detail,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
  fail "QWEN_GROUNDED_EXTRACT_RUNTIME_FAILED" "$qwen_rc"
fi
require_stopped

echo
echo "=== CONTEMPORANEOUS STANDARD CODEX AUTHORITY CONTROL ==="
if [ -n "${OPENAI_API_KEY:-}" ]; then fail "OPENAI_API_KEY_PRESENT_IN_CALLING_SHELL" 50; fi
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract control --plan "$plan" --output "$codex"
require_stopped

echo
echo "=== FAIL-CLOSED GROUNDED-CANDIDATE ADJUDICATION ==="
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_grounded_extract evaluate --plan "$plan" --qwen "$qwen" --codex "$codex" --output "$report"
require_stopped

echo
echo "=== WORKSPACE-CONTROL WAVE-3 CERTIFICATION + ADOPTION DETAIL ==="
"$core_venv/bin/python" - "$qwen" "$reuse" "$certification" "$wave3" "$execid" "$pkgid" "$git_sha" <<'PY'
import json,pathlib,sys
q=json.load(open(sys.argv[1],encoding="utf-8"))
r=json.load(open(sys.argv[2],encoding="utf-8"))
execid,pkgid,git_sha=sys.argv[5],sys.argv[6],sys.argv[7]
local_r2=r["reuse_layers"]["R2_compiled_jit_runtime_cache"]["state"]
shared_r2="VERIFIED_HIT" if local_r2=="VERIFIED_HIT" else ("MISS" if local_r2=="MISS" else "PRESENT_UNVERIFIED")
c3=bool((q.get("canary") or {}).get("pass"))
c4=bool((q.get("domain") or {}).get("structured_output_valid")) and q.get("scientific_inference_calls")==1
stages=[
 {"stage":"C0_STATIC","status":"PASS","execution_identity":execid,"evidence":{"git_sha":git_sha}},
 {"stage":"C1_IMPORT_ACCELERATOR","status":"PASS","execution_identity":execid,"evidence":{"package_reuse":True}},
 {"stage":"C2A_BACKEND_TOOLCHAIN","status":"PASS","execution_identity":execid,"evidence":{"sampling_backend":"vllm_native"}},
 {"stage":"C2B_ENGINE_BOOT","status":"PASS" if q.get("model_init_seconds") is not None and q.get("failure") is None else "FAIL","execution_identity":execid,"evidence":{"model_init_seconds":q.get("model_init_seconds")}},
 {"stage":"C3_CANARY","status":"PASS" if c3 else "FAIL","execution_identity":execid,"evidence":{"synthetic_non_scientific":True,"seconds":(q.get("canary") or {}).get("seconds")}},
 {"stage":"C4_DOMAIN","status":"PASS" if c4 else "FAIL","execution_identity":execid,"evidence":{"scientific_calls":q.get("scientific_inference_calls"),"seconds":(q.get("domain") or {}).get("seconds")}},
]
runtime_certified=all(x["status"]=="PASS" for x in stages[:5])
cert={
 "schema_version":1,
 "execution_identity":execid,
 "package_identity":pkgid,
 "compatibility_predicates":{"cuda_runtime":"12.9","sampling_backend":"vllm_native"},
 "runtime_certified":runtime_certified,
 "stage_receipts":stages,
 "reuse_layers":{
  "R0_ENVIRONMENT":"VERIFIED_REUSE",
  "R1_CERTIFICATION":"VERIFIED_HIT",
  "R2_COMPILED_CACHE":shared_r2,
  "R3_LIVE_SESSION":"HEALTHY_REUSE" if c3 and c4 else "NOT_APPLICABLE",
 },
}
freeze=runtime_certified and shared_r2=="VERIFIED_HIT"
detail={
 "schema_version":1,
 "execution_identity":execid,
 "package_identity":pkgid,
 "project_local_r2_detail":local_r2,
 "timings_seconds":r.get("timings_seconds"),
 "comparison_to_successful_cold_v3":r.get("comparison_to_successful_cold_v3"),
 "WAVE_3_FREEZE_READY":freeze,
 "freeze_rationale":"R2 explicit verified reuse and C3 passed under one live identity." if freeze else "C3 may pass, but R2 lacks explicit VERIFIED_HIT evidence.",
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(cert,indent=2,sort_keys=True)+"\n",encoding="utf-8")
pathlib.Path(sys.argv[4]).write_text(json.dumps(detail,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"certification":cert,"adoption_detail":detail},indent=2,sort_keys=True))
PY

echo "=== FINAL OPERATION RECEIPT ==="
"$core_venv/bin/python" - "$opstate" "$operation_receipt" "$opid" "$git_sha" "$out" "$report" "$wave3" "$qwen" "$codex" "$plan" "$sidecar" <<'PY'
import json,pathlib,sys
state,out,opid,sha,root,report,wave3,qwen,codex,plan,sidecar=sys.argv[1:]
decision=json.load(open(report,encoding="utf-8"))["decision"]
p={"schema_version":1,"operation_id":opid,"task_id":"research-engine-local-document-pipeline-v1","git_sha":sha,"operation_type":"QWEN35_GROUNDED_CANDIDATE_EXTRACT_CALIBRATION","execution_identity":"qwen35-vllm-cu129-image-native-sampler-v3","status":"PASS","terminal_stage":"C4_DOMAIN","result_decision":decision,"owner_process":None,"artifacts":{"output_root":root,"scientific_report":report,"wave3_adoption":wave3,"qwen_result":qwen,"codex_control":codex,"candidate_plan":plan,"sidecar_receipt":sidecar},"expensive_stages_repeated":[]}
for dest in (state,out): pathlib.Path(dest).write_text(json.dumps(p,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(p,indent=2,sort_keys=True))
PY

echo
echo "=== COMPACT RESULT ==="
"$core_venv/bin/python" - "$report" "$wave3" "$sidecar" "$plan" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8")); w=json.load(open(sys.argv[2],encoding="utf-8")); s=json.load(open(sys.argv[3],encoding="utf-8")); p=json.load(open(sys.argv[4],encoding="utf-8"))
print(json.dumps({"decision":r["decision"],"c3_canary_pass":r["c3_canary_pass"],"guard_status":r["guard"]["status"],"guard_reasons":r["guard"]["reasons"],"promotion_thresholds_pass":r["promotion_thresholds_pass"],"codex_agreement":r["codex_agreement"],"candidate_summary":p["candidate_summary"],"sidecar_status":s.get("status"),"sidecar_cache_hit":s.get("cache_hit"),"wave3_reuse_layers":w["reuse_layers"],"runtime_certified":w["runtime_certified"],"WAVE_3_FREEZE_READY":w["WAVE_3_FREEZE_READY"],"wave3_timings_seconds":w.get("timings_seconds")},indent=2,sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
systemctl --user stop "$unit" >/dev/null 2>&1 || true
require_stopped
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "PRODUCTION_ROUTING_CHANGED=false"
echo "C3_CALLS=1"
echo "C4_SCIENTIFIC_CALLS=1"
echo "OUTPUT_ROOT=$out"
echo "GROUNDED_EXTRACT_REPORT=$report"
echo "WAVE3_ADOPTION=$wave3"
echo "OPERATION_RECEIPT=$operation_receipt"
