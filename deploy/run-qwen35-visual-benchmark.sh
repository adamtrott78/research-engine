#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/qwen35-visual-benchmark-$stamp"
runtime="$state/residual-runtimes/qwen3.5-4b-vllm-0.30.0"
runtime_ready="$runtime/.workspace-control-ready.json"
model="$state/residual-models/qwen3.5-4b/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
expected_source_sha="1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a"
helper="$root/benchmarks/document_intelligence/run_qwen35_visual_benchmark.py"
report="$output_root/qwen35-visual-benchmark-report.json"
expected_vllm_dist="0.30.0+cu129"
expected_torch="2.13.0+cu129"
expected_torchcodec="0.15.0+cu129"
pytorch_cu129_index="https://download.pytorch.org/whl/cu129"
vllm_cu129_wheel="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

fail() {
    echo "ERROR=$1" >&2
    exit "${2:-1}"
}

require_stopped() {
    local active sub pid
    active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
    pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
    [ "$active" = "inactive" ] && [ "$sub" = "dead" ] && [ "${pid:-0}" = "0" ] \
        || fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
}

cleanup() {
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

systemctl --user stop "$unit" || true
require_stopped
cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$standard_runtime/bin/python" ] || fail "STANDARD_RUNTIME_NOT_FOUND" 32
[ -f "$helper" ] || fail "QWEN_HELPER_NOT_FOUND" 33
mkdir -p "$output_root"

echo "=== QWEN3.5-4B TARGETED VISUAL FALLBACK BENCHMARK ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"

echo
echo "=== VERIFY PREFETCHED QWEN ==="
for spec in \
  "model.safetensors-00001-of-00002.safetensors:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61" \
  "model.safetensors-00002-of-00002.safetensors:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
do
    name="${spec%%:*}"
    sha="${spec##*:}"
    [ -f "$model/$name" ] || fail "QWEN_WEIGHT_NOT_FOUND:$name" 40
    [ "$(sha256sum "$model/$name" | awk '{print $1}')" = "$sha" ] \
        || fail "QWEN_WEIGHT_SHA_MISMATCH:$name" 41
done
require_stopped

echo
echo "=== SELECT EXCLUSIVE GPU ==="
selected_line="$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used --format=csv,noheader,nounits | sort -t',' -k3 -nr | head -1)"
physical_gpu_index="$(printf '%s' "$selected_line" | cut -d',' -f1 | xargs)"
gpu_uuid="$(printf '%s' "$selected_line" | cut -d',' -f2 | xargs)"
gpu_free_mib="$(printf '%s' "$selected_line" | cut -d',' -f3 | xargs)"
gpu_used_mib="$(printf '%s' "$selected_line" | cut -d',' -f4 | xargs)"
[[ "$physical_gpu_index" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_INDEX" 42
[[ "$gpu_free_mib" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_FREE_MEMORY" 43
[ "$gpu_free_mib" -ge 16384 ] || fail "INSUFFICIENT_FREE_GPU_MEMORY" 44
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$gpu_uuid"; then
    fail "SELECTED_GPU_HAS_ACTIVE_COMPUTE_PROCESS" 45
fi
export CUDA_VISIBLE_DEVICES="$physical_gpu_index"
echo "SELECTED_PHYSICAL_GPU_INDEX=$physical_gpu_index"
echo "SELECTED_GPU_UUID=$gpu_uuid"
echo "SELECTED_GPU_FREE_MIB=$gpu_free_mib"
echo "BASELINE_GPU_USED_MIB=$gpu_used_mib"
require_stopped

echo
echo "=== LOCATE AUTHORITATIVE SEMANTIC-VISUAL PDF ==="
source_pdf=""
while IFS= read -r candidate; do
    if [ "$(sha256sum "$candidate" | awk '{print $1}')" = "$expected_source_sha" ]; then
        source_pdf="$candidate"
        break
    fi
done < <(
    find "$visible_tmp_root" -type f \
      -path '*/residual-source-prefetch-audit-*/sources/semantic-visuals-v1.pdf' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
)
[ -n "$source_pdf" ] || fail "PINNED_SEMANTIC_VISUAL_SOURCE_NOT_FOUND" 46
echo "SOURCE_PDF=$source_pdf"

echo
echo "=== RENDER EXACT VALIDATED FIGURE REGIONS ==="
"$standard_runtime/bin/python" - "$source_pdf" "$output_root" <<'PY'
import json
import pathlib
import sys

import pypdfium2 as pdfium

source = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
pdf = pdfium.PdfDocument(str(source))
if len(pdf) != 12:
    raise SystemExit(f"expected 12 pages, got {len(pdf)}")

targets = [
    (
        "figure6_cifar_training_chart",
        7,
        [99.19828033447266, 72.1703872680664, 495.0450134277344, 171.62620544433594],
    ),
    (
        "figure3_network_architecture",
        3,
        [50.5037956237793, 76.09317016601562, 285.709716796875, 618.7928466796875],
    ),
]
dpi = 300
scale = dpi / 72.0
receipt = {"render_dpi": dpi, "bbox_origin": "TOPLEFT_PDF_POINTS", "targets": []}
for label, index, bbox in targets:
    image = pdf[index].render(scale=scale).to_pil()
    left, top, right, bottom = [round(v * scale) for v in bbox]
    crop = image.crop((left, top, right, bottom))
    path = out / f"{label}-page-zb{index}.png"
    crop.save(path)
    receipt["targets"].append(
        {
            "id": label,
            "page_index_zero_based": index,
            "pdf_bbox": bbox,
            "path": str(path),
            "width_px": crop.width,
            "height_px": crop.height,
        }
    )
(out / "qwen35-render-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
require_stopped

echo
echo "=== RECONCILE ISOLATED VLLM CUDA-12.9 RUNTIME ==="
[ "$(uname -m)" = "x86_64" ] || fail "QWEN_VLLM_CU129_REQUIRES_X86_64" 47

runtime_rebuild=0
if [ ! -x "$runtime/bin/python" ]; then
    runtime_rebuild=1
else
    if ! "$runtime/bin/python" - "$expected_vllm_dist" "$expected_torch" <<'PY'
import importlib.metadata
import sys

import torch

assert importlib.metadata.version("vllm") == sys.argv[1]
assert torch.__version__ == sys.argv[2], torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
PY
    then
        runtime_rebuild=1
    fi
fi

if [ "$runtime_rebuild" -eq 1 ]; then
    rm -rf "$runtime"
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        "$vllm_cu129_wheel" \
        --extra-index-url "$pytorch_cu129_index"
fi

installed_torchcodec="$("$runtime/bin/python" - <<'PY'
import importlib.metadata
try:
    print(importlib.metadata.version("torchcodec"))
except importlib.metadata.PackageNotFoundError:
    print("")
PY
)"
if [ "$installed_torchcodec" != "$expected_torchcodec" ]; then
    echo "TORCHCODEC_RECONCILE=${installed_torchcodec:-missing}->$expected_torchcodec"
    "$runtime/bin/python" -m pip install --force-reinstall --no-deps \
        "torchcodec==$expected_torchcodec" \
        --index-url "$pytorch_cu129_index"
fi

"$runtime/bin/python" -m pip freeze | sort > "$output_root/qwen35-vllm-runtime-freeze.txt"
rm -f "$runtime_ready"

echo
echo "=== FULL VLLM MULTIMODAL IMPORT + CUDA PREFLIGHT ==="
"$runtime/bin/python" - "$runtime_ready" "$expected_vllm_dist" "$expected_torch" "$expected_torchcodec" <<'PY'
import importlib.metadata
import json
import pathlib
import sys

import torch
import torchcodec
from vllm import LLM, SamplingParams
import vllm

vllm_dist = importlib.metadata.version("vllm")
torchcodec_dist = importlib.metadata.version("torchcodec")
assert vllm_dist == sys.argv[2], vllm_dist
assert torch.__version__ == sys.argv[3], torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torchcodec_dist == sys.argv[4], torchcodec_dist
assert vllm.__version__ == "0.30.0", vllm.__version__
assert LLM is not None
assert SamplingParams is not None
assert torch.cuda.is_available(), "vLLM runtime Torch cannot access CUDA"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
probe = torch.empty((32, 32), device="cuda:0")
del probe
payload = {
    "vllm": vllm.__version__,
    "vllm_distribution": vllm_dist,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "torchcodec": torchcodec_dist,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_allocation_probe": "PASS",
    "vllm_multimodal_import_probe": "PASS",
    "engine": "native_qwen3.5_multimodal",
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
require_stopped

echo
echo "=== RUN EXACTLY THREE TARGETED QWEN VISUAL-SEMANTIC CALLS ==="
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME="$output_root/xdg-cache"
mkdir -p "$XDG_CACHE_HOME"

"$runtime/bin/python" "$helper" \
    --model-dir "$model" \
    --figure6 "$output_root/figure6_cifar_training_chart-page-zb7.png" \
    --figure3 "$output_root/figure3_network_architecture-page-zb3.png" \
    --gpu-uuid "$gpu_uuid" \
    --baseline-gpu-used-mib "$gpu_used_mib" \
    --output-dir "$output_root" \
    --output "$report"
require_stopped

echo
echo "=== FINAL SAFETY STATE ==="
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "BENCHMARK_REPORT=$report"
echo "QWEN35_VISUAL_MEASUREMENT=COMPLETE"
