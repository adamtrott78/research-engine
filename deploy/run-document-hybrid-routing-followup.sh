#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
tmp_root="$adam/.workspace-control/tmp"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/document-runtimes/docling-formula-enrichment"
models="$state/document-models/docling-formula-enrichment-2.131.0"
profile_id="docling-formula-enrichment-codeformulav2-cu126-v1"
profile_file="$runtime/.research-engine-profile"
manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"
shim="$root/benchmarks/document_intelligence/runtimes/docling_formula_enrichment.py"
default_raw_root="$tmp_root/document-docling-standard-no-ocr-20261002T035925Z"

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
    echo "=== GPU INVENTORY ==="
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
        --format=csv,noheader,nounits || true

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

raw_root="$(readlink -f "${1:-$default_raw_root}")"
standard_root="$raw_root/docling-standard-no-ocr-paper-package-v1"
followup_root="$raw_root/hybrid-routing-followup-v1"
formula_root="$raw_root/docling-formula-enrichment-targeted-v1"
routing_report="$followup_root/hybrid-routing-report.json"
table_audit="$followup_root/table-residual-audit.json"
target_plan="$followup_root/formula-target-plan.json"
formula_eval="$followup_root/formula-enrichment-evaluation.json"

echo "=== HYBRID DOCUMENT ROUTING + TARGETED FORMULA FOLLOWUP ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 33
[ -f "$shim" ] || fail "FORMULA_SHIM_NOT_FOUND" 34
[ -d "$raw_root" ] || fail "PRESERVED_STANDARD_OUTPUT_NOT_FOUND:$raw_root" 35
[ -d "$standard_root" ] || fail "NORMALIZED_STANDARD_ROOT_NOT_FOUND:$standard_root" 36
[ ! -e "$followup_root" ] || fail "FOLLOWUP_OUTPUT_ALREADY_EXISTS:$followup_root" 37
[ ! -e "$formula_root" ] || fail "FORMULA_OUTPUT_ALREADY_EXISTS:$formula_root" 38

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

mapfile -t native_summaries < <(
    find "$tmp_root" -maxdepth 6 -type f \
        -path '*/docling-paper-package-v1/docling-normalization-summary.json' \
        -print | sort
)
printf 'FOUND_NATIVE_NORMALIZATION_SUMMARIES=%s\n' "${#native_summaries[@]}"
printf '%s\n' "${native_summaries[@]}"
[ "${#native_summaries[@]}" -eq 1 ] || \
    fail "EXPECTED_EXACTLY_ONE_EXISTING_NORMALIZED_NATIVE_CORPUS" 42
native_root="$(dirname "${native_summaries[0]}")"

echo "HEAD=$(git rev-parse HEAD)"
echo "PRESERVED_STANDARD_ROOT=$raw_root"
echo "STANDARD_PACKAGE_ROOT=$standard_root"
echo "NATIVE_PACKAGE_ROOT=$native_root"
echo "FOLLOWUP_ROOT=$followup_root"
echo "FORMULA_OUTPUT_ROOT=$formula_root"
echo "BUNDLE_ROOT=$bundle_root"

mkdir -p "$followup_root"

echo
echo "=== REFRESH CORE CLI / ANALYSIS MODULES ONLY ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

echo
echo "=== 1/4 HYBRID ROUTING CONTRACT ==="
"$core_venv/bin/python" -m research_engine.document_routing \
    "$manifest" \
    --native-package-root "$native_root" \
    --standard-package-root "$standard_root" \
    --output "$routing_report"

echo
echo "=== 2/4 COMPACT TABLE RESIDUAL AUDIT ==="
"$core_venv/bin/python" -m research_engine.document_structure_audit \
    "$manifest" \
    --bundle-root "$bundle_root" \
    --standard-package-root "$standard_root" \
    --output "$table_audit"

echo
echo "=== 3/4 DERIVE BOUNDED FORMULA TARGET PAGES ==="
"$core_venv/bin/python" -m research_engine.document_formula_eval plan \
    "$manifest" \
    --raw-standard-root "$raw_root" \
    --output "$target_plan"

target_pages="$(
    "$core_venv/bin/python" - "$target_plan" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["summary"]["target_pages"])
PY
)"
echo "FORMULA_TARGET_PAGES=$target_pages"
[ "$target_pages" -gt 0 ] || fail "NO_FORMULA_TARGET_PAGES" 43

require_stopped

echo
echo "=== PREPARE ISOLATED FORMULA-ONLY RUNTIME ==="
existing_profile="$(cat "$profile_file" 2>/dev/null || true)"
if [ -d "$runtime" ] && [ "$existing_profile" != "$profile_id" ]; then
    echo "RECREATE_FORMULA_RUNTIME_PROFILE=$existing_profile->$profile_id"
    rm -rf "$runtime"
fi
if [ ! -x "$runtime/bin/python" ]; then
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126
    "$runtime/bin/python" -m pip install \
        "docling-slim[convert-core,format-pdf,models-local,models-vlm-inline]==2.131.0" \
        "docling-ibm-models[opencv-python-headless]>=4.0.2,<5"
    printf '%s\n' "$profile_id" > "$profile_file"
fi

echo "DOCLING_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("docling-slim"))')"
echo "TORCH_VERSION=$("$runtime/bin/python" -c 'import torch; print(torch.__version__)')"
echo "TRANSFORMERS_VERSION=$("$runtime/bin/python" -c 'import transformers; print(transformers.__version__)')"

select_usable_gpu || fail "NO_USABLE_CUDA_GPU_FOR_FORMULA_ENRICHMENT" 50
require_stopped

echo
echo "=== PREFETCH ONLY LAYOUT + CODEFORMULAV2 ==="
mkdir -p "$models"
CUDA_VISIBLE_DEVICES="$SELECTED_GPU_INDEX" \
RESEARCH_ENGINE_DOCLING_DEVICE=cuda \
RESEARCH_ENGINE_DOCLING_ARTIFACTS="$models" \
"$runtime/bin/python" "$shim" --prefetch-models "$models"
require_stopped

echo
echo "=== 4/4 FORMULA-ONLY PAGE EXECUTION ==="
mkdir -p "$formula_root"
while IFS=$'\t' read -r case_id source_location page_number; do
    [ -n "$case_id" ] || continue
    source_pdf="$bundle_root/$source_location"
    [ -f "$source_pdf" ] || fail "FORMULA_SOURCE_NOT_FOUND:$source_pdf" 51
    page_dir="$formula_root/$case_id/page-$(printf '%04d' "$page_number")"
    mkdir -p "$page_dir"
    echo "FORMULA_CASE=$case_id PAGE=$page_number"
    CUDA_VISIBLE_DEVICES="$SELECTED_GPU_INDEX" \
    RESEARCH_ENGINE_DOCLING_DEVICE=cuda \
    RESEARCH_ENGINE_DOCLING_ARTIFACTS="$models" \
    "$runtime/bin/python" "$shim" \
        "$source_pdf" "$page_dir" --page "$page_number"
    require_stopped
done < <(
    "$core_venv/bin/python" - "$target_plan" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for row in data["cases"]:
    for page in row["target_pages"]:
        print(f"{row['case_id']}\t{row['source_location']}\t{page}")
PY
)

echo
echo "=== FORMULA CONTENT EVALUATION ==="
"$core_venv/bin/python" -m research_engine.document_formula_eval evaluate \
    "$manifest" \
    --bundle-root "$bundle_root" \
    --formula-root "$formula_root" \
    --baseline-summary "$standard_root/docling-normalization-summary.json" \
    --output "$formula_eval"

require_stopped

echo
echo "=== COMPACT FOLLOWUP SUMMARY ==="
"$core_venv/bin/python" - "$routing_report" "$table_audit" "$target_plan" "$formula_eval" <<'PY'
import json
import sys

routing, tables, plan, formulas = [
    json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]
]
print(json.dumps({
    "routing": routing["summary"],
    "table_audit": tables["summary"],
    "formula_target_plan": plan["summary"],
    "formula_evaluation": formulas["summary"],
}, indent=2, sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "HYBRID_ROUTING_REPORT=$routing_report"
echo "TABLE_RESIDUAL_AUDIT=$table_audit"
echo "FORMULA_TARGET_PLAN=$target_plan"
echo "FORMULA_EVALUATION=$formula_eval"
echo "DOCUMENT_HYBRID_ROUTING_FOLLOWUP=PASS"
