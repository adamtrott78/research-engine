#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/document-runtimes/docling-standard-no-ocr"
models="$state/document-models/docling-standard-no-ocr-2.131.0"
profile_id="docling-standard-no-ocr-core-cu126-opencv-v4"
previous_profile_id="docling-standard-no-ocr-core-cu126-v3"
profile_file="$runtime/.research-engine-profile"
manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"
shim="$root/benchmarks/document_intelligence/runtimes/docling_standard_no_ocr.py"
tmp_root="$adam/.workspace-control/tmp"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$tmp_root/document-docling-standard-no-ocr-$stamp"

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

selected_gpu_index=""
selected_gpu_uuid=""

run_cuda_probe() {
    "$runtime/bin/python" - <<'PY'
import json
import torch

result = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
}
if not result["cuda_available"] or result["device_count"] < 1:
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(2)

device = torch.device("cuda:0")
result["device_name"] = torch.cuda.get_device_name(device)
before_free, before_total = torch.cuda.mem_get_info(device)

probe = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
probe.fill_(1)
torch.cuda.synchronize(device)

after_free, _ = torch.cuda.mem_get_info(device)
result["free_before_mib"] = round(before_free / 1024 / 1024, 1)
result["free_after_mib"] = round(after_free / 1024 / 1024, 1)
result["total_mib"] = round(before_total / 1024 / 1024, 1)
result["allocation_probe_mib"] = 256
print(json.dumps(result, sort_keys=True))

del probe
torch.cuda.empty_cache()
torch.cuda.synchronize(device)
PY
}

select_usable_gpu() {
    local row gpu_index gpu_uuid free_mib
    local -a candidates=()

    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "GPU_SELECTION_ERROR=nvidia-smi unavailable" >&2
        return 1
    }

    echo "=== GPU INVENTORY ==="
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode,persistence_mode \
        --format=csv,noheader,nounits || true

    echo
    echo "=== ACTIVE GPU COMPUTE PROCESSES ==="
    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits || true

    echo
    echo "=== CUDA ENVIRONMENT ==="
    echo "INHERITED_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "INHERITED_NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-<unset>}"
    echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER-<unset>}"

    mapfile -t candidates < <(
        nvidia-smi \
            --query-gpu=index,uuid,memory.free \
            --format=csv,noheader,nounits \
        | awk -F',' '{
            gsub(/^ +| +$/,"",$1);
            gsub(/^ +| +$/,"",$2);
            gsub(/^ +| +$/,"",$3);
            print $3 "|" $1 "|" $2
        }' \
        | sort -t'|' -k1,1nr
    )

    [ "${#candidates[@]}" -gt 0 ] || {
        echo "GPU_SELECTION_ERROR=no CUDA devices enumerated" >&2
        return 1
    }

    echo
    echo "=== UNMASKED CUDA ALLOCATION CONTROL ==="
    if env -u CUDA_VISIBLE_DEVICES "$runtime/bin/python" - <<'PY'
import json
import os
import torch

result = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
if not result["cuda_available"] or result["device_count"] < 1:
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(2)

device = torch.device("cuda:0")
result["device_name"] = torch.cuda.get_device_name(device)
probe = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
probe.fill_(1)
torch.cuda.synchronize(device)
result["allocation_probe_mib"] = 256
print(json.dumps(result, sort_keys=True))
PY
    then
        echo "UNMASKED_CUDA_ALLOCATION=PASS"
    else
        echo "UNMASKED_CUDA_ALLOCATION=FAIL"
    fi

    echo
    echo "=== PER-GPU INDEX CUDA ALLOCATION PROBES ==="
    for row in "${candidates[@]}"; do
        IFS='|' read -r free_mib gpu_index gpu_uuid <<< "$row"
        echo "GPU_PROBE_INDEX=$gpu_index"
        echo "GPU_PROBE_UUID=$gpu_uuid"
        echo "GPU_PROBE_FREE_MIB=$free_mib"

        if CUDA_VISIBLE_DEVICES="$gpu_index" run_cuda_probe; then
            selected_gpu_index="$gpu_index"
            selected_gpu_uuid="$gpu_uuid"
            echo "SELECTED_PHYSICAL_GPU_INDEX=$selected_gpu_index"
            echo "SELECTED_PHYSICAL_GPU_UUID=$selected_gpu_uuid"
            return 0
        fi

        echo "GPU_PROBE_FAILED_INDEX=$gpu_index"
        echo "GPU_PROBE_FAILED_UUID=$gpu_uuid"
    done

    echo "GPU_SELECTION_ERROR=no GPU passed numeric-index CUDA allocation probe" >&2
    return 1
}

find_padataset() {
    local candidate
    for candidate in \
        "$adam/varMax/PADataset" \
        "$adam/PADataset"
    do
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

echo "=== DOCLING STANDARD / NO-OCR STRUCTURE BAKEOFF ==="

systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 33
[ -f "$shim" ] || fail "DOCLING_STANDARD_NO_OCR_SHIM_NOT_FOUND" 34

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

mkdir -p "$tmp_root" "$(dirname "$runtime")" "$(dirname "$models")"

echo "HEAD=$(git rev-parse HEAD)"
echo "BUNDLE_ROOT=$bundle_root"
echo "OUTPUT_ROOT=$output_root"
echo "RUNTIME=$runtime"
echo "MODEL_ARTIFACTS=$models"
echo "DEVICE=${RESEARCH_ENGINE_DOCLING_DEVICE:-auto}"

echo
echo "=== HARDWARE SNAPSHOT ==="
echo "CPU_COUNT=$(nproc 2>/dev/null || true)"
free -h || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
else
    echo "NVIDIA_SMI=unavailable"
fi

echo
echo "=== PREPARE ISOLATED STRUCTURE RUNTIME ==="
existing_profile="$(cat "$profile_file" 2>/dev/null || true)"
if [ -d "$runtime" ] && [ "$existing_profile" = "$previous_profile_id" ]; then
    echo "MIGRATE_RUNTIME_PROFILE=$existing_profile->$profile_id"
    "$runtime/bin/python" -m pip install \
        "docling-ibm-models[opencv-python-headless]>=4.0.2,<5"
    printf '%s\n' "$profile_id" > "$profile_file"
elif [ -d "$runtime" ] && [ "$existing_profile" != "$profile_id" ]; then
    echo "RECREATE_RUNTIME_PROFILE=$existing_profile->$profile_id"
    rm -rf "$runtime"
fi
if [ ! -x "$runtime/bin/python" ]; then
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126
    "$runtime/bin/python" -m pip install \
        "docling-slim[convert-core,format-pdf,models-local]==2.131.0" \
        "docling-ibm-models[opencv-python-headless]>=4.0.2,<5"
    printf '%s\n' "$profile_id" > "$profile_file"
fi

echo "DOCLING_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("docling-slim"))')"
echo "DOCLING_IBM_MODELS_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("docling-ibm-models"))')"
echo "TORCH_VERSION=$("$runtime/bin/python" -c 'import torch; print(torch.__version__)')"
echo "TORCHVISION_VERSION=$("$runtime/bin/python" -c 'import torchvision; print(torchvision.__version__)')"
echo "SCIPY_VERSION=$("$runtime/bin/python" -c 'import scipy; print(scipy.__version__)')"
echo "OPENCV_VERSION=$("$runtime/bin/python" -c 'import cv2; print(cv2.__version__)')"
"$runtime/bin/python" - <<'PY'
import cv2
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
print("TABLEFORMER_IMPORT_PREFLIGHT=PASS")
PY
echo "RUNTIME_SIZE=$(du -sh "$runtime" | awk '{print $1}')"

echo
echo "=== SELECT HEALTHY CUDA DEVICE ==="
select_usable_gpu || fail "NO_USABLE_CUDA_DEVICE" 42
export CUDA_VISIBLE_DEVICES="$selected_gpu_index"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SELECTED_PHYSICAL_GPU_INDEX=$selected_gpu_index"
echo "SELECTED_PHYSICAL_GPU_UUID=$selected_gpu_uuid"
echo "TORCH_CUDA_AVAILABLE=$("$runtime/bin/python" -c 'import torch; print(torch.cuda.is_available())')"
echo "TORCH_CUDA_DEVICE=$("$runtime/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"

require_stopped

echo
echo "=== PREFETCH ONLY LAYOUT + TABLEFORMER MODELS ==="
mkdir -p "$models"
"$runtime/bin/python" "$shim" --prefetch-models "$models"
echo "MODEL_ARTIFACT_SIZE=$(du -sh "$models" | awk '{print $1}')"
find "$models" -maxdepth 2 -type d -print | sort | head -80

require_stopped

echo
echo "=== SDK PREFLIGHT ==="
export RESEARCH_ENGINE_DOCLING_BIN="$runtime/bin/python"
export RESEARCH_ENGINE_DOCLING_ARTIFACTS="$models"
export RESEARCH_ENGINE_DOCLING_DEVICE="cuda"
export RESEARCH_ENGINE_DOCLING_THREADS="${RESEARCH_ENGINE_DOCLING_THREADS:-$(nproc 2>/dev/null || echo 4)}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$runtime/bin/python" "$shim" --self-test

require_stopped

echo
echo "=== REFRESH CORE BENCHMARK CLI ==="
"$core_venv/bin/python" -m pip install -e "$root"

echo
echo "=== RUN 7-PAPER / 73-PAGE STRUCTURE BENCHMARK ==="
mkdir -p "$output_root"
"$core_venv/bin/research-engine" document-benchmark-external \
    "$manifest" \
    docling \
    standard_no_ocr \
    --bundle-root "$bundle_root" \
    --output-root "$output_root" \
    --timeout-seconds 3600 \
    | tee "$output_root/terminal-summary.json"

require_stopped

echo
echo "=== OUTPUT SUMMARY ==="
echo "RUNTIME_SIZE=$(du -sh "$runtime" | awk '{print $1}')"
echo "MODEL_ARTIFACT_SIZE=$(du -sh "$models" | awk '{print $1}')"
echo "OUTPUT_SIZE=$(du -sh "$output_root" | awk '{print $1}')"
echo "SUMMARY=$output_root/benchmark-summary-docling-standard_no_ocr.json"

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "OUTPUT_ROOT=$output_root"
echo "DOCLING_STANDARD_NO_OCR_STRUCTURE_BAKEOFF=PASS"
