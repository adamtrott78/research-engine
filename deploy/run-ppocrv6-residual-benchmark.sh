#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/ppocrv6-residual-benchmark-$stamp"
runtime="$state/residual-runtimes/ppocrv6-medium-paddle-3.2.0"
runtime_ready="$runtime/.workspace-control-ready.json"
wheelhouse="$state/residual-runtime-wheelhouse"
models="$state/residual-models"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
det_revision="8e0f56fb2ef86b461d99cfc7ac5c137738985f61"
rec_revision="e5a92bcbc5cc1b494628e458d267778f0704fd7c"
det_model="$models/pp-ocrv6-medium-det/$det_revision"
rec_model="$models/pp-ocrv6-medium-rec/$rec_revision"
expected_source_sha="3948255a723380d8844bab31c60298cee00da9334efa377f1d596b306792b6a1"
report="$output_root/ppocrv6-benchmark-report.json"
helper="$root/benchmarks/document_intelligence/run_ppocrv6_residual_benchmark.py"

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
[ -f "$helper" ] || fail "BENCHMARK_HELPER_NOT_FOUND" 33
mkdir -p "$output_root"

echo "=== PP-OCRV6 BOUNDED RESIDUAL BENCHMARK ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "CANDIDATE_FAMILY=paddleocr-pp-ocrv6-medium"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"

echo
echo "=== VERIFY PREFETCHED MODELS ==="
[ -f "$det_model/inference.pdiparams" ] || fail "DET_MODEL_NOT_FOUND" 40
[ -f "$rec_model/inference.pdiparams" ] || fail "REC_MODEL_NOT_FOUND" 41
[ "$(sha256sum "$det_model/inference.pdiparams" | awk '{print $1}')" = "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960" ] \
    || fail "DET_WEIGHT_SHA_MISMATCH" 42
[ "$(sha256sum "$rec_model/inference.pdiparams" | awk '{print $1}')" = "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319" ] \
    || fail "REC_WEIGHT_SHA_MISMATCH" 43

wheel="$(find "$wheelhouse" -maxdepth 1 -type f -name 'paddleocr-3.7.0-*.whl' -print)"
[ "$(printf '%s\n' "$wheel" | sed '/^$/d' | wc -l)" -eq 1 ] || fail "EXPECTED_ONE_PADDLEOCR_WHEEL" 44
[ "$(sha256sum "$wheel" | awk '{print $1}')" = "c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338" ] \
    || fail "PADDLEOCR_WHEEL_SHA_MISMATCH" 45
require_stopped

echo
echo "=== SELECT GPU ==="
selected_line="$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | sort -t',' -k3 -nr | head -1)"
physical_gpu_index="$(printf '%s' "$selected_line" | cut -d',' -f1 | xargs)"
gpu_uuid="$(printf '%s' "$selected_line" | cut -d',' -f2 | xargs)"
gpu_free_mib="$(printf '%s' "$selected_line" | cut -d',' -f3 | xargs)"
[[ "$physical_gpu_index" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_INDEX" 46
[[ "$gpu_free_mib" =~ ^[0-9]+$ ]] || fail "INVALID_GPU_FREE_MEMORY" 47
[ "$gpu_free_mib" -ge 8192 ] || fail "INSUFFICIENT_FREE_GPU_MEMORY" 48
echo "SELECTED_PHYSICAL_GPU_INDEX=$physical_gpu_index"
echo "SELECTED_GPU_UUID=$gpu_uuid"
echo "SELECTED_GPU_FREE_MIB=$gpu_free_mib"
export CUDA_VISIBLE_DEVICES="$physical_gpu_index"
require_stopped

echo
echo "=== LOCATE AUTHORITATIVE SOURCE PDF ==="
source_pdf=""
while IFS= read -r candidate; do
    if [ "$(sha256sum "$candidate" | awk '{print $1}')" = "$expected_source_sha" ]; then
        source_pdf="$candidate"
        break
    fi
done < <(
    find "$visible_tmp_root" -type f \
        -path '*/residual-source-prefetch-audit-*/sources/scan-mixed-quality-v1.pdf' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
)
[ -n "$source_pdf" ] || fail "PINNED_SCAN_SOURCE_PDF_NOT_FOUND" 49
echo "SOURCE_PDF=$source_pdf"
echo "SOURCE_SHA256=$expected_source_sha"

echo
echo "=== RENDER ONLY PROMOTED TARGET PAGES ==="
"$standard_runtime/bin/python" - "$source_pdf" "$output_root" <<'PY'
import json, pathlib, sys
import pypdfium2 as pdfium

source=pathlib.Path(sys.argv[1])
out=pathlib.Path(sys.argv[2])
pdf=pdfium.PdfDocument(str(source))
if len(pdf) != 28:
    raise SystemExit(f"expected 28 pages, got {len(pdf)}")
targets=[("scan_textless",3),("low_confidence_text",16)]
receipt={"render_dpi":200,"targets":[]}
for label,index in targets:
    page=pdf[index]
    bitmap=page.render(scale=200/72)
    image=bitmap.to_pil()
    path=out/f"{label}-page-zb{index}.png"
    image.save(path)
    receipt["targets"].append({
        "id":label,
        "page_index_zero_based":index,
        "path":str(path),
        "width_px":image.width,
        "height_px":image.height,
    })
(out/"render-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== BUILD ISOLATED PP-OCRV6 RUNTIME ==="
if [ ! -f "$runtime_ready" ]; then
    rm -rf "$runtime"
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        "paddlepaddle-gpu==3.2.0" \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    "$runtime/bin/python" -m pip install "$wheel"
fi
"$runtime/bin/python" - "$runtime_ready" <<'PY'
import json, pathlib, sys
import paddle
import paddleocr
assert paddle.__version__ == "3.2.0", paddle.__version__
assert paddle.is_compiled_with_cuda()
assert paddle.device.cuda.device_count() >= 1
payload={
    "paddle_version":paddle.__version__,
    "paddleocr_import":"PASS",
    "cuda_compiled":True,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("PADDLE_VERSION="+paddle.__version__)
print("PADDLEOCR_IMPORT=PASS")
print("CUDA_DEVICE_COUNT="+str(paddle.device.cuda.device_count()))
PY
"$runtime/bin/python" -m pip freeze | sort > "$output_root/ppocrv6-runtime-freeze.txt"
require_stopped

echo
echo "=== RUN EXACTLY THREE BOUNDED OCR INFERENCES ==="
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PADDLEX_HOME="$output_root/paddlex-home"
export XDG_CACHE_HOME="$output_root/xdg-cache"
mkdir -p "$PADDLEX_HOME" "$XDG_CACHE_HOME"

"$runtime/bin/python" "$helper" \
    --det-model "$det_model" \
    --rec-model "$rec_model" \
    --scan-image "$output_root/scan_textless-page-zb3.png" \
    --low-confidence-image "$output_root/low_confidence_text-page-zb16.png" \
    --gpu-uuid "$gpu_uuid" \
    --output "$report"
require_stopped

if find "$PADDLEX_HOME" -type f -size +1M -print -quit | grep -q .; then
    fail "UNEXPECTED_LARGE_PADDLEX_CACHE_DURING_LOCAL_MODEL_INFERENCE" 50
fi

echo
echo "=== FINAL BENCHMARK SAFETY CHECK ==="
python3 - "$report" <<'PY'
import json, sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["measurement_decision"] == "MEASURED_PENDING_HUMAN_QUALITY", r
assert r["candidate_inference_calls"] == 3
assert r["scientific_state_materialized"] is False
assert r["quality_status"] == "PENDING_HUMAN_REVIEW"
assert all(r["checks"].values()), r["checks"]
print(json.dumps({
    "measurement_decision":r["measurement_decision"],
    "candidate_inference_calls":r["candidate_inference_calls"],
    "model_init_seconds":r["model_init_seconds"],
    "warm_page_seconds_max":r["warm_page_seconds_max"],
    "peak_process_vram_mib":r["peak_process_vram_mib"],
    "peak_rss_mib":r["peak_rss_mib"],
    "quality_status":r["quality_status"],
},indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== FINAL SAFETY STATE ==="
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "BENCHMARK_REPORT=$report"
echo "HUMAN_REVIEW_IMAGES=$output_root/scan_textless-page-zb3.png,$output_root/low_confidence_text-page-zb16.png"
echo "PP_OCRV6_MEASUREMENT=READY_FOR_HUMAN_REVIEW"
