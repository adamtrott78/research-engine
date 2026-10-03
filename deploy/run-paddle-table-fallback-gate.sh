#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
legacy_tmp_root="$adam/.workspace-control/tmp"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/document-runtimes/paddle-table-recognition-v2"
model_cache="$state/document-models/paddle-table-recognition-v2-3.7.0"
profile_id="paddle-table-recognition-v2-paddleocr-3.7.0-paddle-3.3.0-cu126-v1"
profile_file="$runtime/.research-engine-profile"
manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"
shim="$root/benchmarks/document_intelligence/runtimes/paddle_table_recognition_v2.py"
legacy_standard_raw="$legacy_tmp_root/document-docling-standard-no-ocr-20261002T035925Z"
standard_root="$legacy_standard_raw/docling-standard-no-ocr-paper-package-v1"
table_audit="$legacy_standard_raw/hybrid-routing-followup-v1/table-residual-audit.json"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/paddle-table-fallback-$stamp"
target_plan="$output_root/paddle-table-target-plan.json"
paddle_output="$output_root/paddle-output"
evaluation="$output_root/paddle-table-evaluation.json"

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

echo "=== TARGETED PADDLE TABLE RECOGNITION V2 FALLBACK ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 33
[ -f "$shim" ] || fail "PADDLE_TABLE_SHIM_NOT_FOUND" 34
[ -d "$standard_root" ] || fail "STANDARD_PACKAGE_ROOT_NOT_FOUND" 35
[ -f "$table_audit" ] || fail "TABLE_RESIDUAL_AUDIT_NOT_FOUND" 36

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

mkdir -p "$visible_tmp_root" "$output_root" "$paddle_output" "$(dirname "$runtime")" "$model_cache"

echo "HEAD=$(git rev-parse HEAD)"
echo "VISIBLE_TMP_ROOT=$visible_tmp_root"
echo "OUTPUT_ROOT=$output_root"
echo "BUNDLE_ROOT=$bundle_root"
echo "PADDLE_RUNTIME=$runtime"
echo "PADDLE_MODEL_CACHE=$model_cache"

echo
echo "=== REFRESH CORE ANALYSIS MODULES ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

echo
echo "=== DERIVE THE SAME THREE PERSISTENT EMPTY-GRID TARGETS ==="
"$core_venv/bin/python" -m research_engine.document_table_residual plan \
    "$manifest" \
    --standard-package-root "$standard_root" \
    --table-audit "$table_audit" \
    --output "$target_plan"

target_tables="$("$core_venv/bin/python" - "$target_plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["summary"]["target_tables"])
PY
)"
target_pages="$("$core_venv/bin/python" - "$target_plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["summary"]["target_pages"])
PY
)"
echo "TARGET_TABLES=$target_tables"
echo "TARGET_PAGES=$target_pages"
[ "$target_tables" -eq 3 ] || fail "EXPECTED_EXACTLY_THREE_PERSISTENT_TABLE_TARGETS" 42
[ "$target_pages" -eq 3 ] || fail "EXPECTED_EXACTLY_THREE_PERSISTENT_TABLE_PAGES" 43

echo
echo "=== PREPARE ISOLATED PADDLE TABLE RUNTIME ==="
existing_profile="$(cat "$profile_file" 2>/dev/null || true)"
if [ -d "$runtime" ] && [ "$existing_profile" != "$profile_id" ]; then
    echo "RECREATE_PADDLE_RUNTIME_PROFILE=$existing_profile->$profile_id"
    rm -rf "$runtime"
fi
if [ ! -x "$runtime/bin/python" ]; then
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    "$runtime/bin/python" -m pip install \
        "paddleocr[doc-parser]==3.7.0" \
        pypdfium2
    printf '%s\n' "$profile_id" > "$profile_file"
fi

echo "PADDLEOCR_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("paddleocr"))')"
echo "PADDLEPADDLE_VERSION=$("$runtime/bin/python" -c 'import paddle; print(paddle.__version__)')"
echo "PYPDFIUM2_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("pypdfium2"))')"
echo "RUNTIME_SIZE=$(du -sh "$runtime" | awk '{print $1}')"

select_usable_gpu || fail "NO_USABLE_PADDLE_CUDA_GPU" 50
require_stopped

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_INDEX"
export PADDLE_PDX_CACHE_HOME="$model_cache"
export PADDLE_PDX_MODEL_SOURCE="BOS"

echo
echo "=== RUN ONE RESIDENT PP-TABLEMAGIC PIPELINE ACROSS THREE CROPS ==="
"$runtime/bin/python" "$shim" \
    "$target_plan" \
    --bundle-root "$bundle_root" \
    --output-root "$paddle_output" \
    --model-cache "$model_cache" \
    --device gpu:0 \
    --dpi 300 \
    --padding-pt 6
require_stopped

echo "MODEL_CACHE_SIZE=$(du -sh "$model_cache" | awk '{print $1}')"

echo
echo "=== EVALUATE ONLY THE THREE FAILED TABLE REFERENCES ==="
"$core_venv/bin/python" -m research_engine.document_paddle_table_eval \
    "$manifest" \
    --bundle-root "$bundle_root" \
    --target-plan "$target_plan" \
    --paddle-output-root "$paddle_output" \
    --output "$evaluation"
require_stopped

echo
echo "=== COMPACT PADDLE TABLE FALLBACK SUMMARY ==="
"$core_venv/bin/python" - "$target_plan" "$evaluation" <<'PY'
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
echo "TARGET_PLAN=$target_plan"
echo "PADDLE_OUTPUT=$paddle_output"
echo "EVALUATION=$evaluation"
echo "PADDLE_TABLE_FALLBACK_GATE=PASS"
