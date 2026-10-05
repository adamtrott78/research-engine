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
out="$visible/qwen35-guarded-semantic-calibration-$stamp"
plan="$out/calibration-plan.json"; qwen="$out/qwen-results.json"; codex="$out/codex-control-results.json"; report="$out/calibration-report.json"
helper_log="$out/qwen-helper.log"; reuse="$out/qwen-runtime-reuse-evidence.json"

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

echo "=== QWEN3.5 GUARDED EXTRACT / VERIFY CALIBRATION ==="
systemctl --user stop "$unit" >/dev/null 2>&1 || true
require_stopped
cd "$root"
[ "$(git branch --show-current)" = main ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$runtime/bin/python" ] || fail "QWEN_RUNTIME_NOT_FOUND" 33
[ -f "$runtime_ready" ] || fail "QWEN_RUNTIME_CERTIFICATION_NOT_FOUND" 34
command -v codex >/dev/null || fail "CODEX_NOT_FOUND" 35
echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$out"
echo "SOURCE_RUN_ID=$run_id"

echo
echo "=== R0 / R1 EXACT RUNTIME REUSE ==="
"$runtime/bin/python" - "$runtime_ready" "$runtime_contract" <<'PY'
import importlib.metadata, json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["runtime_contract_version"] == sys.argv[2]
assert p["sampling_backend"] == "vllm_native"
assert p["vllm_flashinfer_sampler"] is False
assert p["vllm_distribution"] == "0.30.0+cu129"
assert p["vllm_release_wheel_sha256"] == "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
assert p["torch"] == "2.13.0+cu129"
assert p["torchvision"] == "0.28.0+cu129"
assert p["torchcodec"] == "0.16.0+cu129"
assert p["torch_cuda_runtime"] == "12.9"
assert p["cuda_allocation_probe"] == "PASS"
assert p["vllm_multimodal_import_probe"] == "PASS"
assert importlib.metadata.version("vllm") == "0.30.0+cu129"
assert importlib.metadata.version("torchcodec") == "0.16.0+cu129"
print("R0_PACKAGE_ENVIRONMENT=VERIFIED_REUSE")
print("R1_EXECUTION_CERTIFICATION=VERIFIED_HIT")
PY
echo "PACKAGE_INSTALL_DOWNLOAD_WORK=FORBIDDEN_AND_SKIPPED"

for spec in \
 "model.safetensors-00001-of-00002.safetensors:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61" \
 "model.safetensors-00002-of-00002.safetensors:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
do
  name="${spec%%:*}"; sha="${spec##*:}"
  [ -f "$model/$name" ] || fail "QWEN_WEIGHT_NOT_FOUND:$name" 36
  [ "$(sha256sum "$model/$name" | awk '{print $1}')" = "$sha" ] || fail "QWEN_WEIGHT_SHA_MISMATCH:$name" 37
done

echo
echo "=== POSTGRESQL READ-ONLY CALIBRATION SOURCE ==="
export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"
export RESEARCH_ENGINE_ARTIFACT_ROOT="${RESEARCH_ENGINE_ARTIFACT_ROOT:-$state/artifacts}"
sudo docker compose -f docker-compose.yml exec -T db pg_isready -U research_engine -d research_engine
require_stopped

echo
echo "=== BUILD BOUNDED TWO-JOB PLAN ==="
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_guarded_semantic plan "$run_id" --output "$plan" --max-context-chars 26000
require_stopped

echo
echo "=== SELECT EXCLUSIVE GPU ==="
line="$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used --format=csv,noheader,nounits | sort -t',' -k3 -nr | head -1)"
gpu_index="$(printf '%s' "$line" | cut -d',' -f1 | xargs)"; gpu_uuid="$(printf '%s' "$line" | cut -d',' -f2 | xargs)"
gpu_free="$(printf '%s' "$line" | cut -d',' -f3 | xargs)"; gpu_used="$(printf '%s' "$line" | cut -d',' -f4 | xargs)"
[ "$gpu_free" -ge 16384 ] || fail "INSUFFICIENT_FREE_GPU_MEMORY" 40
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$gpu_uuid"; then fail "SELECTED_GPU_HAS_ACTIVE_COMPUTE_PROCESS" 41; fi
export CUDA_VISIBLE_DEVICES="$gpu_index"; export VLLM_USE_FLASHINFER_SAMPLER=0; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1
cache_present=false
if [ -d "$cache_root" ] && find "$cache_root" -type f -print -quit 2>/dev/null | grep -q .; then cache_present=true; fi
export XDG_CACHE_HOME="$cache_root"; mkdir -p "$cache_root"
echo "SELECTED_GPU_UUID=$gpu_uuid"; echo "VLLM_COMPILE_CACHE_PRESENT_BEFORE=$cache_present"; echo "VLLM_PERSISTENT_COMPILE_CACHE=$cache_root"

echo
echo "=== TWO QWEN DOMAIN CALLS / ONE ENGINE BOOT ==="
set +e
PYTHONPATH="$root/src" "$runtime/bin/python" "$root/benchmarks/document_intelligence/run_qwen35_guarded_semantic_calibration.py" --plan "$plan" --model-dir "$model" --gpu-uuid "$gpu_uuid" --baseline-gpu-used-mib "$gpu_used" --output "$qwen" 2>&1 | tee "$helper_log"
qwen_rc="${PIPESTATUS[0]}"
set -e

PYTHONPATH="$root/src" "$core_venv/bin/python" - "$helper_log" "$reuse" "$cache_present" <<'PY'
import json,pathlib,sys
from research_engine.qwen_runtime_certification import make_runtime_reuse_report
log=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
report=make_runtime_reuse_report(log_text=log.read_text(encoding="utf-8",errors="replace"),cache_present_before=sys.argv[3].lower()=="true",package_environment_reused=True,execution_certification_reused=True,live_engine_reused_for_domain_calls=True)
report["scientific_call_count_added_for_wave3"]=0
report["domain_calls_in_same_engine_session"]=2
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2,sort_keys=True))
PY

[ "$qwen_rc" -eq 0 ] || fail "QWEN_GUARDED_CALIBRATION_RUNTIME_FAILED" "$qwen_rc"
require_stopped

echo
echo "=== CONTEMPORANEOUS STANDARD CODEX CONTROLS ==="
if [ -n "${OPENAI_API_KEY:-}" ]; then fail "OPENAI_API_KEY_PRESENT_IN_CALLING_SHELL" 50; fi
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_guarded_semantic control --plan "$plan" --output "$codex"
require_stopped

echo
echo "=== DETERMINISTIC GUARD + QWEN VS CODEX ADJUDICATION ==="
PYTHONPATH="$root/src" "$core_venv/bin/python" -m research_engine.qwen_guarded_semantic evaluate --plan "$plan" --qwen "$qwen" --codex "$codex" --output "$report"
require_stopped

echo
echo "=== COMPACT RESULT ==="
"$core_venv/bin/python" - "$report" "$reuse" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8")); w=json.load(open(sys.argv[2],encoding="utf-8"))
print(json.dumps({"decision":r["decision"],"guarded_jobs_passed":r["guarded_jobs_passed"],"codex_agreement_jobs_passed":r["codex_agreement_jobs_passed"],"jobs":[{"job_id":x["job_id"],"job_type":x["job_type"],"guard_status":x["guard"]["status"],"guard_reasons":x["guard"]["reasons"],"codex_agreement_pass":x["qwen_vs_codex_agreement_pass"],"fallthrough_required":x["fallthrough_required"]} for x in r["jobs"]],"wave3_reuse_layers":w["reuse_layers"],"wave3_timings_seconds":w["timings_seconds"],"wave3_comparison_to_successful_cold_v3":w["comparison_to_successful_cold_v3"]},indent=2,sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
systemctl --user stop "$unit" >/dev/null 2>&1 || true
require_stopped
echo "DAEMON=inactive/dead PID=0"; echo "SCIENTIFIC_STATE_MATERIALIZED=false"; echo "PRODUCTION_ROUTING_CHANGED=false"; echo "QWEN_CALLS=2"; echo "C3_CALLS=0"
echo "OUTPUT_ROOT=$out"; echo "CALIBRATION_REPORT=$report"; echo "WAVE3_RUNTIME_REUSE=$reuse"
