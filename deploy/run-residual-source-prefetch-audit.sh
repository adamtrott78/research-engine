#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
native_runtime="$adam/.document-runtimes/docling-native"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
standard_cache="$state/document-models/docling-standard-no-ocr-2.131.0"
native_script="$root/benchmarks/document_intelligence/runtimes/docling_native.py"
standard_script="$root/benchmarks/document_intelligence/runtimes/docling_standard_no_ocr.py"
case_dir="$root/benchmarks/document_intelligence/cases"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/residual-source-prefetch-audit-$stamp"
source_root="$output_root/sources"
raw_root="$output_root/raw"
package_root="$output_root/packages"
fetch_report="$output_root/source-fetch-report.json"
audit_report="$output_root/residual-source-audit-report.json"

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

cleanup() {
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

select_structure_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA_SMI_NOT_FOUND" 40

    mapfile -t candidates < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' '{
            gsub(/^ +| +$/,"",$1);
            gsub(/^ +| +$/,"",$2);
            print $2 "|" $1
        }' \
        | sort -t'|' -k1,1nr
    )
    [ "${#candidates[@]}" -gt 0 ] || fail "NO_GPU_CANDIDATES" 41

    local row free_mib index
    for row in "${candidates[@]}"; do
        IFS='|' read -r free_mib index <<< "$row"
        echo "STRUCTURE_GPU_PROBE_INDEX=$index" >&2
        echo "STRUCTURE_GPU_PROBE_FREE_MIB=$free_mib" >&2
        if CUDA_VISIBLE_DEVICES="$index" "$standard_runtime/bin/python" - <<'PY'
import json
import sys
import torch

payload = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if not payload["cuda_available"] or payload["device_count"] < 1:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    raise SystemExit(2)
device = torch.device("cuda:0")
probe = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
probe.fill_(1)
torch.cuda.synchronize(device)
payload["device_name"] = torch.cuda.get_device_name(device)
payload["allocation_probe_mib"] = 128
print(json.dumps(payload, sort_keys=True), file=sys.stderr)
PY
        then
            printf '%s\n' "$index"
            return 0
        fi
    done
    return 1
}

single_json() {
    local directory="$1"
    local -a matches=()
    mapfile -t matches < <(find "$directory" -maxdepth 1 -type f -name '*.json' -print | sort)
    [ "${#matches[@]}" -eq 1 ] || fail "EXPECTED_EXACTLY_ONE_RUNTIME_JSON:$directory:${#matches[@]}" 50
    printf '%s\n' "${matches[0]}"
}

echo "=== RESIDUAL SOURCE PREFETCH / DETERMINISTIC AUDIT ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$native_runtime/bin/python" ] || fail "NATIVE_RUNTIME_NOT_FOUND" 33
[ -x "$standard_runtime/bin/python" ] || fail "STANDARD_RUNTIME_NOT_FOUND" 34
[ -d "$standard_cache" ] || fail "STANDARD_MODEL_CACHE_NOT_FOUND" 35
[ -f "$native_script" ] || fail "NATIVE_SCRIPT_NOT_FOUND" 36
[ -f "$standard_script" ] || fail "STANDARD_SCRIPT_NOT_FOUND" 37

for case_id in scan-mixed-quality-v1 semantic-visuals-v1 malformed-layout-v1; do
    [ -f "$case_dir/$case_id.json" ] || fail "CASE_MANIFEST_NOT_FOUND:$case_id" 38
done

mkdir -p "$output_root" "$source_root" "$raw_root" "$package_root"

echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "NATIVE_RUNTIME=$native_runtime"
echo "STANDARD_RUNTIME=$standard_runtime"
echo "STANDARD_MODEL_CACHE=$standard_cache"
echo "CANDIDATE_MODEL_DOWNLOADS=0"

echo
echo "=== REFRESH CORE PACKAGE ONLY ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

echo
echo "=== FETCH + HASH REGISTERED SOURCES ==="
"$core_venv/bin/python" -m research_engine.residual_source_audit fetch \
    --case-dir "$case_dir" \
    --source-root "$source_root" \
    --output "$fetch_report" \
    > "$output_root/source-fetch-summary.json"
cat "$fetch_report"
require_stopped

echo
echo "=== SELECT EXISTING STRUCTURE GPU ==="
echo "GPU_INVENTORY_BEGIN"
nvidia-smi \
    --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits
echo "GPU_INVENTORY_END"
structure_gpu="$(select_structure_gpu)" || fail "NO_USABLE_STRUCTURE_GPU" 42
[[ "$structure_gpu" =~ ^[0-9]+$ ]] || fail "INVALID_STRUCTURE_GPU_INDEX:$structure_gpu" 43
echo "SELECTED_STRUCTURE_GPU_INDEX=$structure_gpu"
require_stopped

export RESEARCH_ENGINE_DOCLING_DEVICE="cuda"
export RESEARCH_ENGINE_DOCLING_ARTIFACTS="$standard_cache"
export RESEARCH_ENGINE_DOCLING_THREADS="${RESEARCH_ENGINE_DOCLING_THREADS:-$(nproc 2>/dev/null || echo 4)}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo
echo "=== EXISTING RUNTIME SELF-TESTS ==="
"$native_runtime/bin/python" "$native_script" --self-test
CUDA_VISIBLE_DEVICES="$structure_gpu" \
    "$standard_runtime/bin/python" "$standard_script" --self-test
require_stopped

for case_id in scan-mixed-quality-v1 semantic-visuals-v1 malformed-layout-v1; do
    echo
    echo "=== CASE $case_id ==="
    source_pdf="$source_root/$case_id.pdf"
    [ -f "$source_pdf" ] || fail "SOURCE_NOT_FOUND_AFTER_FETCH:$case_id" 60

    native_out="$raw_root/$case_id/native"
    standard_out="$raw_root/$case_id/standard"
    mkdir -p "$native_out" "$standard_out"

    echo "--- native ---"
    "$native_runtime/bin/python" "$native_script" \
        "$source_pdf" "$native_out" \
        > "$output_root/$case_id-native.log" 2>&1
    require_stopped

    echo "--- standard_no_ocr (offline existing cache) ---"
    CUDA_VISIBLE_DEVICES="$structure_gpu" \
        "$standard_runtime/bin/python" "$standard_script" \
        "$source_pdf" "$standard_out" \
        > "$output_root/$case_id-standard.log" 2>&1
    require_stopped

    native_json="$(single_json "$native_out")"
    standard_json="$(single_json "$standard_out")"

    "$core_venv/bin/python" -m research_engine.residual_source_audit normalize \
        --case-id "$case_id" \
        --source-pdf "$source_pdf" \
        --native-json "$native_json" \
        --standard-json "$standard_json" \
        --output-root "$package_root" \
        > "$output_root/$case_id-normalize.json"
    require_stopped
done

echo
echo "=== DETERMINISTIC RESIDUAL OPENER AUDIT ==="
set +e
"$core_venv/bin/python" -m research_engine.residual_source_audit audit \
    --case-dir "$case_dir" \
    --source-root "$source_root" \
    --package-root "$package_root" \
    --output "$audit_report" \
    --fail-on-reject \
    > "$output_root/audit-summary.json"
audit_status=$?
set -e
cat "$audit_report"

echo
echo "=== AUDIT SAFETY / PREFETCH READINESS CHECK ==="
"$core_venv/bin/python" - "$audit_report" "$audit_status" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
audit_status = int(sys.argv[2])
checks = {
    "audit_command_status_zero": audit_status == 0,
    "decision_pass": report["decision"] == "PASS",
    "candidate_model_downloads_zero": report["candidate_model_downloads"] == 0,
    "candidate_model_inference_calls_zero": report["candidate_model_inference_calls"] == 0,
    "scientific_state_materialized_false": report["scientific_state_materialized"] is False,
    "model_prefetch_ready": report["model_prefetch_ready"] is True,
    "scan_case_accepted": report["cases"]["scan-mixed-quality-v1"]["accepted"] is True,
    "semantic_visual_case_accepted": report["cases"]["semantic-visuals-v1"]["accepted"] is True,
    "layout_case_accepted": report["cases"]["malformed-layout-v1"]["accepted"] is True,
}
payload = {
    "decision": "PASS" if all(checks.values()) else "REJECT",
    "checks": checks,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(2)
PY
require_stopped

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "CANDIDATE_MODEL_DOWNLOADS=0"
echo "CANDIDATE_MODEL_INFERENCE_CALLS=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "FETCH_REPORT=$fetch_report"
echo "AUDIT_REPORT=$audit_report"
echo "RESIDUAL_SOURCE_PREFETCH_AUDIT=PASS"
