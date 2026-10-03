#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
legacy_tmp_root="$adam/.workspace-control/tmp"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/document-runtimes/docling-standard-no-ocr"
models="$state/document-models/docling-standard-no-ocr-2.131.0"
profile_file="$runtime/.research-engine-profile"
expected_profile="docling-standard-no-ocr-core-cu126-opencv-v4"
manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"
shim="$root/benchmarks/document_intelligence/runtimes/docling_tableformer_accurate.py"
legacy_standard_raw="$legacy_tmp_root/document-docling-standard-no-ocr-20261002T035925Z"
standard_root="$legacy_standard_raw/docling-standard-no-ocr-paper-package-v1"
table_audit="$legacy_standard_raw/hybrid-routing-followup-v1/table-residual-audit.json"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/document-tableformer-accurate-residual-$stamp"
plan="$output_root/tableformer-accurate-target-plan.json"
accurate_root="$output_root/accurate-pages"
evaluation="$output_root/tableformer-accurate-evaluation.json"

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

select_usable_gpu() {
    local row gpu_index gpu_uuid free_mib
    local -a candidates=()

    command -v nvidia-smi >/dev/null 2>&1 || return 1
    mapfile -t candidates < <(
        nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits \
        | awk -F',' '{
            gsub(/^ +| +$/,"",$1);
            gsub(/^ +| +$/,"",$2);
            gsub(/^ +| +$/,"",$3);
            print $3 "|" $1 "|" $2
        }' \
        | sort -t'|' -k1,1nr
    )

    echo "=== GPU INVENTORY ==="
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
        --format=csv,noheader,nounits || true

    for row in "${candidates[@]}"; do
        IFS='|' read -r free_mib gpu_index gpu_uuid <<< "$row"
        echo "GPU_PROBE_INDEX=$gpu_index"
        echo "GPU_PROBE_UUID=$gpu_uuid"
        echo "GPU_PROBE_FREE_MIB=$free_mib"
        if CUDA_VISIBLE_DEVICES="$gpu_index" "$runtime/bin/python" - <<'PY'
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
        then
            SELECTED_GPU_INDEX="$gpu_index"
            SELECTED_GPU_UUID="$gpu_uuid"
            export SELECTED_GPU_INDEX SELECTED_GPU_UUID
            echo "SELECTED_PHYSICAL_GPU_INDEX=$SELECTED_GPU_INDEX"
            echo "SELECTED_PHYSICAL_GPU_UUID=$SELECTED_GPU_UUID"
            return 0
        fi
    done
    return 1
}

echo "=== TARGETED TABLEFORMER ACCURATE RESIDUAL GATE ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$runtime/bin/python" ] || fail "STANDARD_RUNTIME_NOT_FOUND" 33
[ "$(cat "$profile_file" 2>/dev/null || true)" = "$expected_profile" ] || fail "STANDARD_RUNTIME_PROFILE_MISMATCH" 34
[ -d "$models" ] || fail "STANDARD_MODEL_CACHE_NOT_FOUND" 35
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 36
[ -f "$shim" ] || fail "TABLEFORMER_ACCURATE_SHIM_NOT_FOUND" 37
[ -d "$standard_root" ] || fail "STANDARD_PACKAGE_ROOT_NOT_FOUND" 38
[ -f "$table_audit" ] || fail "TABLE_RESIDUAL_AUDIT_NOT_FOUND" 39

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

mkdir -p "$visible_tmp_root" "$output_root" "$accurate_root"

echo "HEAD=$(git rev-parse HEAD)"
echo "LEGACY_INPUT_ROOT=$legacy_standard_raw"
echo "VISIBLE_TMP_ROOT=$visible_tmp_root"
echo "OUTPUT_ROOT=$output_root"
echo "STANDARD_PACKAGE_ROOT=$standard_root"
echo "TABLE_AUDIT=$table_audit"

echo
echo "=== REFRESH CORE ANALYSIS MODULES ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

echo
echo "=== DERIVE EMPTY-GRID TARGETS ONLY ==="
"$core_venv/bin/python" -m research_engine.document_table_residual plan \
    "$manifest" \
    --standard-package-root "$standard_root" \
    --table-audit "$table_audit" \
    --output "$plan"

target_tables="$("$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["summary"]["target_tables"])
PY
)"
target_pages="$("$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["summary"]["target_pages"])
PY
)"
echo "TARGET_TABLES=$target_tables"
echo "TARGET_PAGES=$target_pages"
[ "$target_tables" -gt 0 ] || fail "NO_EMPTY_GRID_TABLE_TARGETS" 42

echo
echo "=== SELECT HEALTHY CUDA DEVICE ==="
select_usable_gpu || fail "NO_USABLE_CUDA_GPU_FOR_TABLEFORMER_ACCURATE" 50
require_stopped

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_INDEX"
export RESEARCH_ENGINE_DOCLING_DEVICE="cuda"
export RESEARCH_ENGINE_DOCLING_ARTIFACTS="$models"
export RESEARCH_ENGINE_DOCLING_THREADS="${RESEARCH_ENGINE_DOCLING_THREADS:-$(nproc 2>/dev/null || echo 4)}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo
echo "=== RUN TABLEFORMER ACCURATE ONLY ON TARGET PAGES ==="
while IFS=$'\t' read -r case_id source_location page_number; do
    [ -n "$case_id" ] || continue
    source_pdf="$bundle_root/$source_location"
    [ -f "$source_pdf" ] || fail "TARGET_SOURCE_NOT_FOUND:$source_pdf" 51
    page_dir="$accurate_root/$case_id/page-$(printf '%04d' "$page_number")"
    mkdir -p "$page_dir"
    echo "TABLEFORMER_ACCURATE_CASE=$case_id PAGE=$page_number"
    "$runtime/bin/python" "$shim" \
        "$source_pdf" "$page_dir" --page "$page_number"
    require_stopped
done < <(
    "$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
seen=set()
for row in d["targets"]:
    key=(row["case_id"], row["source_location"], row["page_number"])
    if key in seen:
        continue
    seen.add(key)
    print(f"{key[0]}\t{key[1]}\t{key[2]}")
PY
)

echo
echo "=== EVALUATE ACCURATE MODE AGAINST TARGET REFERENCES ==="
"$core_venv/bin/python" -m research_engine.document_table_residual evaluate \
    "$manifest" \
    --bundle-root "$bundle_root" \
    --target-plan "$plan" \
    --accurate-root "$accurate_root" \
    --output "$evaluation"

require_stopped

echo
echo "=== COMPACT TABLEFORMER ACCURATE SUMMARY ==="
"$core_venv/bin/python" - "$plan" "$evaluation" <<'PY'
import json, sys
plan, evaluation = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]
print(json.dumps({
    "target_plan": plan["summary"],
    "evaluation": evaluation["summary"],
}, indent=2, sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "VISIBLE_TMP_ROOT=$visible_tmp_root"
echo "OUTPUT_ROOT=$output_root"
echo "TARGET_PLAN=$plan"
echo "EVALUATION=$evaluation"
echo "TABLEFORMER_ACCURATE_RESIDUAL_GATE=PASS"
