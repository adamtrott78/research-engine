#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
legacy_tmp_root="$adam/.workspace-control/tmp"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
formula_runtime="$state/document-runtimes/docling-formula-enrichment"
formula_cache="$state/document-models/docling-formula-enrichment-2.131.0"
table_runtime="$state/document-runtimes/paddle-table-recognition-v2"
table_cache="$state/document-models/paddle-table-recognition-v2-3.7.0"
legacy_standard_raw="$legacy_tmp_root/document-docling-standard-no-ocr-20261002T035925Z"
formula_plan="$legacy_standard_raw/hybrid-routing-followup-v1/formula-target-plan.json"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/local-inference-concurrency-$stamp"
report="$output_root/local-inference-concurrency-report.json"

fail() {
    echo "ERROR=$1" >&2
    exit "${2:-1}"
}

require_stopped() {
    local active sub pid
    active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
    pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
    if [ "$active" != "inactive" ] || [ "$sub" != "dead" ] || [ "${pid:-0}" != "0" ]; then
        fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
    fi
}

find_padataset() {
    local candidate
    for candidate in "$adam/varMax/PADataset" "$adam/PADataset"; do
        if [ -d "$candidate/papers/milcom2026/reference_notes/papers" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    candidate="$(
        find "$adam" -maxdepth 5 -type d \
            -path '*/PADataset/papers/milcom2026/reference_notes/papers' \
            -print -quit 2>/dev/null \
        | sed 's#/papers/milcom2026/reference_notes/papers$##'
    )"
    [ -n "$candidate" ] || return 1
    printf '%s\n' "$candidate"
}

probe_formula_gpu() {
    local index="$1"
    CUDA_VISIBLE_DEVICES="$index" "$formula_runtime/bin/python" - <<'PY'
import json
import torch
result = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if not result["cuda_available"] or result["device_count"] < 1:
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(2)
device = torch.device("cuda:0")
probe = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
probe.fill_(1)
torch.cuda.synchronize(device)
result["device_name"] = torch.cuda.get_device_name(device)
result["allocation_probe_mib"] = 256
print(json.dumps(result, sort_keys=True))
PY
}

probe_table_gpu() {
    local index="$1"
    CUDA_VISIBLE_DEVICES="$index" "$table_runtime/bin/python" - <<'PY'
import json
import paddle
result = {
    "compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
    "device_count": paddle.device.cuda.device_count(),
}
if not result["compiled_with_cuda"] or result["device_count"] < 1:
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(2)
paddle.set_device("gpu:0")
probe = paddle.ones([64, 1024, 1024], dtype="float32")
result["allocation_probe_mib"] = int(probe.numel() * 4 / 1024 / 1024)
result["mean"] = float(probe.mean())
print(json.dumps(result, sort_keys=True))
PY
}

echo "=== SHARED LOCAL INFERENCE CONCURRENCY GATE ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$formula_runtime/bin/python" ] || fail "FORMULA_RUNTIME_NOT_FOUND" 33
[ -x "$table_runtime/bin/python" ] || fail "TABLE_RUNTIME_NOT_FOUND" 34
[ -d "$formula_cache" ] || fail "FORMULA_MODEL_CACHE_NOT_FOUND" 35
[ -d "$table_cache" ] || fail "TABLE_MODEL_CACHE_NOT_FOUND" 36
[ -f "$formula_plan" ] || fail "FORMULA_TARGET_PLAN_NOT_FOUND" 37

shopt -s nullglob
table_plans=("$visible_tmp_root"/paddle-table-fallback-*/paddle-table-target-plan.json)
shopt -u nullglob
[ "${#table_plans[@]}" -gt 0 ] || fail "VISIBLE_PADDLE_TABLE_TARGET_PLAN_NOT_FOUND" 38
table_plan="$(
    ls -1t "${table_plans[@]}" | head -n 1
)"
[ -f "$table_plan" ] || fail "TABLE_TARGET_PLAN_NOT_FOUND" 39

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

mkdir -p "$visible_tmp_root" "$output_root"

echo "HEAD=$(git rev-parse HEAD)"
echo "VISIBLE_TMP_ROOT=$visible_tmp_root"
echo "OUTPUT_ROOT=$output_root"
echo "FORMULA_PLAN=$formula_plan"
echo "TABLE_PLAN=$table_plan"
echo "FORMULA_RUNTIME=$formula_runtime"
echo "TABLE_RUNTIME=$table_runtime"
echo "FORMULA_CACHE_SIZE=$(du -sh "$formula_cache" | awk '{print $1}')"
echo "TABLE_CACHE_SIZE=$(du -sh "$table_cache" | awk '{print $1}')"

echo
echo "=== REFRESH CORE PACKAGE ONLY ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA_SMI_NOT_FOUND" 45
echo
echo "=== GPU INVENTORY ==="
nvidia-smi \
    --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits

mapfile -t gpu_indices < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | sed 's/[[:space:]]//g'
)
[ "${#gpu_indices[@]}" -ge 2 ] || fail "CONCURRENCY_GATE_REQUIRES_TWO_GPUS" 46

formula_gpu=""
for index in "${gpu_indices[@]}"; do
    echo "FORMULA_GPU_PROBE=$index"
    if probe_formula_gpu "$index"; then
        formula_gpu="$index"
        break
    fi
done
[ -n "$formula_gpu" ] || fail "NO_USABLE_FORMULA_GPU" 47

table_gpu=""
for index in "${gpu_indices[@]}"; do
    [ "$index" = "$formula_gpu" ] && continue
    echo "TABLE_GPU_PROBE=$index"
    if probe_table_gpu "$index"; then
        table_gpu="$index"
        break
    fi
done
[ -n "$table_gpu" ] || fail "NO_DISTINCT_USABLE_TABLE_GPU" 48

echo "FORMULA_PHYSICAL_GPU_INDEX=$formula_gpu"
echo "TABLE_PHYSICAL_GPU_INDEX=$table_gpu"
[ "$formula_gpu" != "$table_gpu" ] || fail "MODEL_FAMILIES_MUST_USE_DISTINCT_GPUS" 49
require_stopped

export RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1
export RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING=8
export RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS=900
export RESEARCH_ENGINE_FORMULA_RUNTIME="$formula_runtime"
export RESEARCH_ENGINE_FORMULA_MODEL_CACHE="$formula_cache"
export RESEARCH_ENGINE_FORMULA_GPU_INDEX="$formula_gpu"
export RESEARCH_ENGINE_TABLE_RUNTIME="$table_runtime"
export RESEARCH_ENGINE_TABLE_MODEL_CACHE="$table_cache"
export RESEARCH_ENGINE_TABLE_GPU_INDEX="$table_gpu"

echo
echo "=== CONCURRENT COLD/WARM MODEL-OWNER TEST ==="
"$core_venv/bin/python" -m research_engine.local_inference_benchmark \
    --bundle-root "$bundle_root" \
    --formula-plan "$formula_plan" \
    --table-plan "$table_plan" \
    --output-root "$output_root"
require_stopped

[ -f "$report" ] || fail "CONCURRENCY_REPORT_NOT_FOUND" 60

echo
echo "=== COMPACT LOCAL INFERENCE CONCURRENCY SUMMARY ==="
"$core_venv/bin/python" - "$report" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "decision": d["decision"],
    "first_wave_seconds": d["first_wave_seconds"],
    "warm_mixed_wave_seconds": d["warm_mixed_wave_seconds"],
    "formula_server_pids": d["formula_server_pids"],
    "table_server_pids": d["table_server_pids"],
    "formula_model_load_counts": d["formula_model_load_counts"],
    "table_model_load_counts": d["table_model_load_counts"],
    "checks": d["checks"],
    "service_health_before_close": d["service_health_before_close"],
    "resources": d["resources"],
}, indent=2, sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "FORMULA_PHYSICAL_GPU_INDEX=$formula_gpu"
echo "TABLE_PHYSICAL_GPU_INDEX=$table_gpu"
echo "VISIBLE_TMP_ROOT=$visible_tmp_root"
echo "OUTPUT_ROOT=$output_root"
echo "REPORT=$report"
echo "LOCAL_INFERENCE_CONCURRENCY_GATE=PASS"
