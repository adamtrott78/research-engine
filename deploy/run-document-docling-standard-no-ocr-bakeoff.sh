#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
core_venv="$state/venv"
runtime="$state/document-runtimes/docling-standard-no-ocr"
models="$state/document-models/docling-standard-no-ocr-2.131.0"
profile_id="docling-standard-no-ocr-core-cu126-v3"
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
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo "NVIDIA_SMI=unavailable"
fi

echo
echo "=== PREPARE ISOLATED STRUCTURE RUNTIME ==="
if [ -d "$runtime" ]; then
    existing_profile="$(cat "$profile_file" 2>/dev/null || true)"
    if [ "$existing_profile" != "$profile_id" ]; then
        echo "RECREATE_RUNTIME_PROFILE=$existing_profile->$profile_id"
        rm -rf "$runtime"
    fi
fi
if [ ! -x "$runtime/bin/python" ]; then
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126
    "$runtime/bin/python" -m pip install \
        "docling-slim[convert-core,format-pdf,models-local]==2.131.0"
    printf '%s\n' "$profile_id" > "$profile_file"
fi

echo "DOCLING_VERSION=$("$runtime/bin/python" -c 'import importlib.metadata as m; print(m.version("docling-slim"))')"
echo "TORCH_VERSION=$("$runtime/bin/python" -c 'import torch; print(torch.__version__)')"
echo "TORCHVISION_VERSION=$("$runtime/bin/python" -c 'import torchvision; print(torchvision.__version__)')"
echo "SCIPY_VERSION=$("$runtime/bin/python" -c 'import scipy; print(scipy.__version__)')"
torch_cuda_available="$("$runtime/bin/python" -c 'import torch; print(torch.cuda.is_available())')"
echo "TORCH_CUDA_AVAILABLE=$torch_cuda_available"
[ "$torch_cuda_available" = "True" ] || fail "TORCH_CUDA_UNAVAILABLE" 42
echo "TORCH_CUDA_DEVICE=$("$runtime/bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "RUNTIME_SIZE=$(du -sh "$runtime" | awk '{print $1}')"

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
export RESEARCH_ENGINE_DOCLING_DEVICE="${RESEARCH_ENGINE_DOCLING_DEVICE:-auto}"
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
