#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/paddleocr-vl-residual-benchmark-$stamp"
runtime="$state/residual-runtimes/paddleocr-vl-1.6-transformers-4.55.0"
runtime_ready="$runtime/.workspace-control-ready.json"
model="$state/residual-models/paddleocr-vl-1.6/c5630abae1d940eafe0697512a0325494b02ab42"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
expected_source_sha="1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a"
expected_model_sha="85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
helper="$root/benchmarks/document_intelligence/run_paddleocr_vl_residual_benchmark.py"
report="$output_root/paddleocr-vl-benchmark-report.json"

fail() { echo "ERROR=$1" >&2; exit "${2:-1}"; }

require_stopped() {
    local active sub pid
    active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
    pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
    [ "$active" = "inactive" ] && [ "$sub" = "dead" ] && [ "${pid:-0}" = "0" ] \
        || fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
}

cleanup() { systemctl --user stop "$unit" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

systemctl --user stop "$unit" || true
require_stopped
cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$standard_runtime/bin/python" ] || fail "STANDARD_RUNTIME_NOT_FOUND" 32
[ -f "$helper" ] || fail "VLM_HELPER_NOT_FOUND" 33
mkdir -p "$output_root"

echo "=== PADDLEOCR-VL 1.6 TARGETED VISUAL BENCHMARK ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"

echo
echo "=== VERIFY PREFETCHED VLM ==="
[ -f "$model/model.safetensors" ] || fail "VLM_MODEL_NOT_FOUND" 40
[ "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" = "$expected_model_sha" ] \
    || fail "VLM_MODEL_SHA_MISMATCH" 41
require_stopped

echo
echo "=== SELECT GPU ==="
selected_line="$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | sort -t',' -k3 -nr | head -1)"
physical_gpu_index="$(printf '%s' "$selected_line" | cut -d',' -f1 | xargs)"
gpu_uuid="$(printf '%s' "$selected_line" | cut -d',' -f2 | xargs)"
gpu_free_mib="$(printf '%s' "$selected_line" | cut -d',' -f3 | xargs)"
[[ "$physical_gpu_index" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_INDEX" 44
[[ "$gpu_free_mib" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_FREE_MEMORY" 45
[ "$gpu_free_mib" -ge 14336 ] || fail "INSUFFICIENT_FREE_GPU_MEMORY" 46
export CUDA_VISIBLE_DEVICES="$physical_gpu_index"
echo "SELECTED_PHYSICAL_GPU_INDEX=$physical_gpu_index"
echo "SELECTED_GPU_UUID=$gpu_uuid"
echo "SELECTED_GPU_FREE_MIB=$gpu_free_mib"
require_stopped

echo
echo "=== LOCATE AUTHORITATIVE SEMANTIC-VISUAL PDF ==="
source_pdf=""
while IFS= read -r candidate; do
    if [ "$(sha256sum "$candidate" | awk '{print $1}')" = "$expected_source_sha" ]; then
        source_pdf="$candidate"; break
    fi
done < <(
    find "$visible_tmp_root" -type f \
      -path '*/residual-source-prefetch-audit-*/sources/semantic-visuals-v1.pdf' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
)
[ -n "$source_pdf" ] || fail "PINNED_SEMANTIC_VISUAL_SOURCE_NOT_FOUND" 47
echo "SOURCE_PDF=$source_pdf"

echo
echo "=== RENDER EXACT VALIDATED FIGURE REGIONS ==="
"$standard_runtime/bin/python" - "$source_pdf" "$output_root" <<'PY'
import json, pathlib, sys
import pypdfium2 as pdfium

source=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
pdf=pdfium.PdfDocument(str(source))
if len(pdf)!=12:
    raise SystemExit(f"expected 12 pages, got {len(pdf)}")
targets=[
    ("figure6_cifar_training_chart",7,[99.19828033447266,72.1703872680664,495.0450134277344,171.62620544433594]),
    ("figure3_network_architecture",3,[50.5037956237793,76.09317016601562,285.709716796875,618.7928466796875]),
]
dpi=300
scale=dpi/72.0
receipt={"render_dpi":dpi,"bbox_origin":"TOPLEFT_PDF_POINTS","targets":[]}
for label,index,bbox in targets:
    page=pdf[index]
    image=page.render(scale=scale).to_pil()
    left,top,right,bottom=[round(v*scale) for v in bbox]
    crop=image.crop((left,top,right,bottom))
    path=out/f"{label}-page-zb{index}.png"
    crop.save(path)
    receipt["targets"].append({
        "id":label,"page_index_zero_based":index,"pdf_bbox":bbox,
        "pixel_crop":[left,top,right,bottom],"path":str(path),
        "width_px":crop.width,"height_px":crop.height,
    })
(out/"visual-region-render-receipt.json").write_text(
    json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8"
)
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== BUILD ISOLATED TRANSFORMERS VLM RUNTIME ==="
if [ -f "$runtime_ready" ]; then
    if ! "$runtime/bin/python" - <<'PY'
import torch, torchvision, transformers
from google.protobuf import __version__ as protobuf_version
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
assert transformers.__version__ == "4.55.0", transformers.__version__
assert protobuf_version == "5.29.5", protobuf_version
PY
    then
        echo "STALE_VLM_RUNTIME_SENTINEL=1"
        rm -f "$runtime_ready"
    fi
fi
if [ ! -f "$runtime_ready" ]; then
    rm -rf "$runtime"
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
      --index-url https://download.pytorch.org/whl/cu126 \
      "torch==2.8.0" "torchvision==0.23.0"
    "$runtime/bin/python" -m pip install \
      "transformers==4.55.0" "accelerate>=1.6,<2" "protobuf==5.29.5"
fi
"$runtime/bin/python" - "$runtime_ready" "$model" <<'PY'
import json, pathlib, sys
import torch, torchvision, transformers
from google.protobuf import __version__ as protobuf_version
from transformers import AutoConfig, AutoProcessor
assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
assert transformers.__version__ == "4.55.0", transformers.__version__
assert protobuf_version == "5.29.5", protobuf_version
assert torch.cuda.is_available()
model_dir=sys.argv[2]
config=AutoConfig.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
auto_map=getattr(config,"auto_map",{}) or {}
assert auto_map.get("AutoModelForCausalLM") == "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration", auto_map
processor=AutoProcessor.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
assert processor.__class__.__name__ == "PaddleOCRVLProcessor", processor.__class__.__name__
payload={
    "torch":torch.__version__,
    "torchvision":torchvision.__version__,
    "transformers":transformers.__version__,
    "protobuf":protobuf_version,
    "cuda":True,
    "loader":"AutoModelForCausalLM+AutoProcessor",
    "declared_auto_model_for_causal_lm":auto_map["AutoModelForCausalLM"],
    "processor_class":processor.__class__.__name__,
    "trust_remote_code":"PINNED_LOCAL_MODEL_ONLY",
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
PY
"$runtime/bin/python" -m pip freeze | sort > "$output_root/paddleocr-vl-runtime-freeze.txt"
require_stopped

echo
echo "=== RUN EXACTLY THREE TARGETED VLM INFERENCES VIA DECLARED TRANSFORMERS INTERFACE ==="
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$output_root/xdg-cache"
mkdir -p "$XDG_CACHE_HOME"

"$runtime/bin/python" "$helper" \
  --model-dir "$model" \
  --figure6 "$output_root/figure6_cifar_training_chart-page-zb7.png" \
  --figure3 "$output_root/figure3_network_architecture-page-zb3.png" \
  --gpu-uuid "$gpu_uuid" \
  --output-dir "$output_root" \
  --output "$report"
require_stopped

echo
echo "=== FINAL VLM SAFETY CHECK ==="
python3 - "$report" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["measurement_decision"]=="MEASURED_PENDING_HUMAN_AND_CODEX_CONTROL",r
assert r["candidate_inference_calls"]==3
assert r["scientific_state_materialized"] is False
assert all(r["checks"].values()),r["checks"]
print(json.dumps({
    "measurement_decision":r["measurement_decision"],
    "candidate_inference_calls":r["candidate_inference_calls"],
    "model_init_seconds":r["model_init_seconds"],
    "warm_region_seconds_max":r["warm_region_seconds_max"],
    "peak_process_vram_mib":r["peak_process_vram_mib"],
    "peak_rss_mib":r["peak_rss_mib"],
},indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== FINAL SAFETY STATE ==="
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "BENCHMARK_REPORT=$report"
echo "HUMAN_REVIEW_CROPS=$output_root/figure6_cifar_training_chart-page-zb7.png,$output_root/figure3_network_architecture-page-zb3.png"
echo "PADDLEOCR_VL_MEASUREMENT=READY_FOR_HUMAN_REVIEW"
