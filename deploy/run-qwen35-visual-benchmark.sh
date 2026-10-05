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
receipt="$output_root/qwen35-benchmark-receipt.json"
host_receipt="$output_root/qwen35-host-build-preflight.json"
runtime_log="$output_root/qwen35-runtime-preflight.log"
helper_log="$output_root/qwen35-helper.log"
console_log="$output_root/qwen35-benchmark-console.log"
render_receipt="$output_root/qwen35-render-receipt.json"
render_cache_root="$state/residual-render-cache/qwen35-visual-v2"
runtime_contract_version="qwen35-vllm-cu129-image-native-sampler-v3"
render_contract_version="qwen35-validated-region-render-v2"
model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
vllm_cache_root="$state/residual-runtime-cache/$runtime_contract_version/$model_revision"
expected_vllm_dist="0.30.0+cu129"
expected_vllm_wheel_sha="e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
expected_torch="2.13.0+cu129"
expected_torchvision="0.28.0+cu129"
expected_torchcodec="0.16.0+cu129"
pytorch_cu129_index="https://download.pytorch.org/whl/cu129"
vllm_cu129_wheel="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
export VLLM_USE_FLASHINFER_SAMPLER=0

terminal_stage="bootstrap"
status="FAILED"
failure_class="UNCLASSIFIED_FAILURE"
failure_message="benchmark exited before success"
runtime_setup_mode="not_started"
runtime_identity_reused="false"
runtime_reuse_check_seconds=""
runtime_setup_seconds=""
runtime_smoke_seconds=""

now_ns() {
    python3 - <<'PY'
import time
print(time.time_ns())
PY
}

elapsed_seconds() {
    python3 - "$1" "$2" <<'PY'
import sys
print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000:.6f}")
PY
}

fail() {
    failure_class="$1"
    failure_message="${3:-$1}"
    echo "ERROR=$failure_class" >&2
    [ "$failure_message" = "$failure_class" ] || echo "ERROR_MESSAGE=$failure_message" >&2
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

mkdir -p "$output_root"
exec > >(tee "$console_log") 2>&1

write_compact_receipt() {
    local rc="$1"
    local started=0 completed=0 initialized=false
    if [ -f "$helper_log" ]; then
        started="$(grep -c 'QWEN_CALL_START' "$helper_log" 2>/dev/null || true)"
        completed="$(grep -c 'QWEN_CALL_COMPLETE' "$helper_log" 2>/dev/null || true)"
        grep -q 'QWEN_MODEL_INIT_COMPLETE' "$helper_log" 2>/dev/null && initialized=true || true
    fi
    RECEIPT_STATUS="$status" RECEIPT_STAGE="$terminal_stage" \
    RECEIPT_FAILURE_CLASS="$failure_class" RECEIPT_FAILURE_MESSAGE="$failure_message" \
    RECEIPT_RUNTIME_SETUP_MODE="$runtime_setup_mode" \
    RECEIPT_RUNTIME_IDENTITY_REUSED="$runtime_identity_reused" \
    RECEIPT_RUNTIME_REUSE_CHECK_SECONDS="$runtime_reuse_check_seconds" \
    RECEIPT_RUNTIME_SETUP_SECONDS="$runtime_setup_seconds" \
    RECEIPT_RUNTIME_SMOKE_SECONDS="$runtime_smoke_seconds" \
    RECEIPT_RUNTIME_CONTRACT_VERSION="$runtime_contract_version" \
    RECEIPT_SAMPLER_BACKEND="vllm_native" \
    RECEIPT_VLLM_CACHE_ROOT="$vllm_cache_root" \
    RECEIPT_QWEN_STARTED="$started" RECEIPT_QWEN_COMPLETED="$completed" \
    RECEIPT_MODEL_INITIALIZED="$initialized" RECEIPT_EXIT_CODE="$rc" \
    python3 - "$receipt" "$root" "$output_root" "$report" "$runtime_ready" "$host_receipt" "$render_receipt" "$runtime_log" "$helper_log" "$console_log" <<'PY'
import json, os, pathlib, subprocess, sys
receipt, root, out, report, runtime, host, render, runtime_log, helper_log, console_log = map(pathlib.Path, sys.argv[1:])
def read_json(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
def size(path):
    try: return path.stat().st_size
    except OSError: return None
def fenv(name):
    value=os.environ.get(name,"")
    return float(value) if value else None
try:
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
except Exception:
    head=None
primary=read_json(report); render_obj=read_json(render)
targets=render_obj.get("targets",[]) if isinstance(render_obj,dict) else []
setup_mode=os.environ.get("RECEIPT_RUNTIME_SETUP_MODE")
payload={
  "schema_version":1,
  "operation":"qwen3.5-4b-targeted-visual-fallback-benchmark",
  "git_source_identity":head,
  "terminal_stage":os.environ.get("RECEIPT_STAGE"),
  "status":os.environ.get("RECEIPT_STATUS"),
  "failure":{"class":os.environ.get("RECEIPT_FAILURE_CLASS"),"message":os.environ.get("RECEIPT_FAILURE_MESSAGE")},
  "runtime":{
    "contract_version":os.environ.get("RECEIPT_RUNTIME_CONTRACT_VERSION"),
    "sampling_backend":os.environ.get("RECEIPT_SAMPLER_BACKEND"),
    "flashinfer_sampler_enabled":False,
    "persistent_compile_cache_root":os.environ.get("RECEIPT_VLLM_CACHE_ROOT"),
    "setup_mode":setup_mode,
    "verified_identity_reused":os.environ.get("RECEIPT_RUNTIME_IDENTITY_REUSED")=="true",
    "installation_or_download_performed":setup_mode in {"full_rebuild","torchcodec_repair"},
    "exact_execution_path_smoke_passed":isinstance(read_json(runtime),dict) and read_json(runtime).get("vllm_multimodal_import_probe")=="PASS" and read_json(runtime).get("cuda_allocation_probe")=="PASS",
    "host_build":read_json(host),
    "verified_receipt":read_json(runtime),
  },
  "model":{"family":"Qwen3.5-4B","revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a","initialization_occurred":os.environ.get("RECEIPT_MODEL_INITIALIZED")=="true"},
  "qwen_calls_started":int(os.environ.get("RECEIPT_QWEN_STARTED","0")),
  "qwen_calls_completed":int(os.environ.get("RECEIPT_QWEN_COMPLETED","0")),
  "qwen_quality_result":primary.get("measurement_decision") if isinstance(primary,dict) else None,
  "renders":{"generated":sum(not bool(x.get("cache_hit")) for x in targets),"reused":sum(bool(x.get("cache_hit")) for x in targets),"targets":targets},
  "timing_seconds":{"runtime_reuse_check":fenv("RECEIPT_RUNTIME_REUSE_CHECK_SECONDS"),"runtime_setup":fenv("RECEIPT_RUNTIME_SETUP_SECONDS"),"runtime_smoke":fenv("RECEIPT_RUNTIME_SMOKE_SECONDS"),"render_total":render_obj.get("render_total_seconds") if isinstance(render_obj,dict) else None},
  "artifacts":{"output_root":str(out),"primary_report":str(report) if report.exists() else None,"runtime_receipt":str(runtime) if runtime.exists() else None,"host_build_receipt":str(host) if host.exists() else None,"render_receipt":str(render) if render.exists() else None,"runtime_log":str(runtime_log) if runtime_log.exists() else None,"helper_log":str(helper_log) if helper_log.exists() else None,"console_log":str(console_log)},
  "artifact_bytes":{"runtime_log":size(runtime_log),"helper_log":size(helper_log),"console_log":size(console_log)},
  "exit_code":int(os.environ.get("RECEIPT_EXIT_CODE","1")),
  "scientific_state_materialized":False,
  "production_routing_changed":False,
}
receipt.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(f"BENCHMARK_RECEIPT={receipt}")
PY
}

finalize() {
    local rc=$?
    set +e
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
    write_compact_receipt "$rc" || true
    echo "FINAL_DAEMON_STATE=$(systemctl --user is-active "$unit" 2>/dev/null || true)/$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
    echo "SCIENTIFIC_STATE_MATERIALIZED=false"
    echo "OUTPUT_ROOT=$output_root"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

terminal_stage="repo_daemon_guards"
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

echo
echo "=== HOST / BUILD-SELECTION PREFLIGHT ==="
terminal_stage="host_build_preflight"
[ "$(uname -m)" = "x86_64" ] || fail "QWEN_VLLM_CU129_REQUIRES_X86_64" 47
gpu_name="$(nvidia-smi -i "$physical_gpu_index" --query-gpu=name --format=csv,noheader | head -1 | xargs)"
driver_version="$(nvidia-smi -i "$physical_gpu_index" --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)"
driver_major="${driver_version%%.*}"
[[ "$driver_major" =~ ^[0-9]+$ ]] || fail "INVALID_NVIDIA_DRIVER_VERSION" 48
[ "$driver_major" -ge 525 ] || fail "CUDA12_MINOR_COMPAT_DRIVER_TOO_OLD" 49 "CUDA 12.x runtime selected on NVIDIA driver older than 525: $driver_version"
driver_cuda_max="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
python3 - "$host_receipt" "$gpu_name" "$driver_version" "$driver_cuda_max" "$expected_vllm_dist" "$expected_vllm_wheel_sha" "$expected_torch" "$expected_torchvision" "$expected_torchcodec" "$runtime_contract_version" <<'PY'
import json, platform, shutil, sys, sysconfig
from pathlib import Path
payload={
 "schema_version":1,
 "architecture":platform.machine(),
 "python":{"version":platform.python_version(),"implementation":platform.python_implementation(),"soabi":sysconfig.get_config_var("SOABI")},
 "gpu":{"name":sys.argv[2],"driver_version":sys.argv[3],"driver_reported_cuda_max":sys.argv[4] or None},
 "requested_runtime":{"vllm_distribution":sys.argv[5],"vllm_release_wheel_sha256":sys.argv[6],"torch":sys.argv[7],"torchvision":sys.argv[8],"torchcodec":sys.argv[9],"runtime_contract_version":sys.argv[10],"cuda_build_family":"cu129","sampling_backend":"vllm_native","VLLM_USE_FLASHINFER_SAMPLER":"0"},
 "nvcc":{"path":shutil.which("nvcc"),"required_for_selected_sampler":False},
 "compatibility_predicates":{"x86_64":platform.machine()=="x86_64","cuda12_minor_compat_driver_family":int(sys.argv[3].split(".")[0])>=525},
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
PY
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
echo "=== EXACT RUNTIME / WHEEL IDENTITY CACHE ==="
terminal_stage="runtime_identity_cache"
runtime_reuse_start_ns="$(now_ns)"
runtime_ready_ok=0
if [ -f "$runtime_ready" ]; then
    if python3 - "$runtime_ready" "$runtime_contract_version" "$expected_vllm_dist" "$expected_vllm_wheel_sha" "$expected_torch" "$expected_torchvision" "$expected_torchcodec" <<'PY'
import json, platform, sys, sysconfig
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["runtime_contract_version"] == sys.argv[2]
assert p["vllm_distribution"] == sys.argv[3]
assert p["vllm_release_wheel_sha256"] == sys.argv[4]
assert p["torch"] == sys.argv[5]
assert p["torchvision"] == sys.argv[6]
assert p["torchcodec"] == sys.argv[7]
assert p["torch_cuda_runtime"] == "12.9"
assert p["python_version"] == platform.python_version()
assert p["python_soabi"] == sysconfig.get_config_var("SOABI")
assert p["vllm_multimodal_import_probe"] == "PASS"
assert p["cuda_allocation_probe"] == "PASS"
assert p["vllm_flashinfer_sampler"] is False
assert p["sampling_backend"] == "vllm_native"
PY
    then runtime_ready_ok=1; fi
fi
runtime_reuse_end_ns="$(now_ns)"
runtime_reuse_check_seconds="$(elapsed_seconds "$runtime_reuse_start_ns" "$runtime_reuse_end_ns")"

echo
echo "=== RECONCILE ISOLATED VLLM CUDA-12.9 RUNTIME ==="
terminal_stage="runtime_reconcile"
runtime_setup_start_ns="$(now_ns)"
runtime_rebuild=0
if [ ! -x "$runtime/bin/python" ]; then
    runtime_rebuild=1
else
    if ! "$runtime/bin/python" - "$expected_vllm_dist" "$expected_vllm_wheel_sha" "$expected_torch" "$expected_torchvision" <<'PY'
import importlib.metadata
import json
import sys

vllm_dist = importlib.metadata.distribution("vllm")
assert vllm_dist.version == sys.argv[1], vllm_dist.version
direct = json.loads(vllm_dist.read_text("direct_url.json") or "{}")
assert direct.get("archive_info",{}).get("hash") == f"sha256={sys.argv[2]}", direct
import torch
assert torch.__version__ == sys.argv[3], torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert importlib.metadata.version("torchvision") == sys.argv[4]
PY
    then
        runtime_rebuild=1
    fi
fi

if [ "$runtime_rebuild" -eq 1 ]; then
    runtime_setup_mode="full_rebuild"
    rm -rf "$runtime"
    python3 -m venv "$runtime"
    "$runtime/bin/python" -m pip install --upgrade pip
    "$runtime/bin/python" -m pip install \
        "$vllm_cu129_wheel" \
        "torch==$expected_torch" \
        "torchcodec==$expected_torchcodec" \
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
    [ "$runtime_setup_mode" = "not_started" ] && runtime_setup_mode="torchcodec_repair"
    echo "TORCHCODEC_RECONCILE=${installed_torchcodec:-missing}->$expected_torchcodec"
    "$runtime/bin/python" -m pip install --force-reinstall --no-deps \
        "torchcodec==$expected_torchcodec" \
        --index-url "$pytorch_cu129_index"
fi
if [ "$runtime_setup_mode" = "not_started" ]; then
    if [ "$runtime_ready_ok" -eq 1 ]; then
        runtime_setup_mode="verified_reuse"
        runtime_identity_reused="true"
        echo "RUNTIME_IDENTITY_CACHE=VERIFIED_REUSE"
    else
        runtime_setup_mode="identity_reuse_smoke_revalidation"
        echo "RUNTIME_IDENTITY_CACHE=IDENTITY_MATCH_REVALIDATE_SMOKE"
    fi
    echo "RUNTIME_INSTALL_DOWNLOAD_WORK=SKIPPED"
fi
runtime_setup_end_ns="$(now_ns)"
runtime_setup_seconds="$(elapsed_seconds "$runtime_setup_start_ns" "$runtime_setup_end_ns")"

"$runtime/bin/python" -m pip freeze | sort > "$output_root/qwen35-vllm-runtime-freeze.txt"
rm -f "$runtime_ready"

echo
echo "=== EXACT MULTIMODAL IMPORT / NATIVE-LIBRARY SMOKE ==="
terminal_stage="runtime_multimodal_smoke"
runtime_smoke_start_ns="$(now_ns)"
set +e
"$runtime/bin/python" - "$runtime_ready" "$expected_vllm_dist" "$expected_vllm_wheel_sha" "$expected_torch" "$expected_torchvision" "$expected_torchcodec" "$runtime_contract_version" >"$runtime_log" 2>&1 <<'PY'
import importlib.metadata
import json
import os
import pathlib
import platform
import sys
import sysconfig

vllm_distribution = importlib.metadata.distribution("vllm")
vllm_dist = vllm_distribution.version
direct = json.loads(vllm_distribution.read_text("direct_url.json") or "{}")
assert vllm_dist == sys.argv[2], vllm_dist
assert direct.get("archive_info",{}).get("hash") == f"sha256={sys.argv[3]}", direct

import torch
import torchcodec
from vllm import LLM, SamplingParams
from vllm import envs as vllm_envs
import vllm

torchcodec_dist = importlib.metadata.version("torchcodec")
assert torch.__version__ == sys.argv[4], torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert importlib.metadata.version("torchvision") == sys.argv[5]
assert torchcodec_dist == sys.argv[6], torchcodec_dist
assert vllm.__version__ == "0.30.0", vllm.__version__
assert LLM is not None
assert SamplingParams is not None
assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
assert vllm_envs.VLLM_USE_FLASHINFER_SAMPLER is False
assert torch.cuda.is_available(), "vLLM runtime Torch cannot access CUDA"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
probe = torch.empty((32, 32), device="cuda:0")
del probe
payload = {
    "schema_version": 2,
    "runtime_contract_version": sys.argv[7],
    "python_version": platform.python_version(),
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "vllm": vllm.__version__,
    "vllm_distribution": vllm_dist,
    "vllm_release_wheel_sha256": sys.argv[3],
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "torchvision": importlib.metadata.version("torchvision"),
    "torchcodec": torchcodec_dist,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_allocation_probe": "PASS",
    "vllm_multimodal_import_probe": "PASS",
    "sampling_backend": "vllm_native",
    "vllm_flashinfer_sampler": False,
    "engine": "native_qwen3.5_multimodal",
}
path = pathlib.Path(sys.argv[1])
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY
runtime_smoke_rc=$?
set -e
runtime_smoke_end_ns="$(now_ns)"
runtime_smoke_seconds="$(elapsed_seconds "$runtime_smoke_start_ns" "$runtime_smoke_end_ns")"
cat "$runtime_log"
if [ "$runtime_smoke_rc" -ne 0 ]; then
    if grep -Eq 'libav(util|device|codec|format|filter|resample|scale)\.so' "$runtime_log"; then
        fail "RUNTIME_MULTIMODAL_NATIVE_LIBRARY_FAILURE" 50 "TorchCodec/vLLM native-library smoke failed; inspect targeted runtime log"
    elif grep -q 'libnvrtc.so' "$runtime_log"; then
        fail "RUNTIME_CUDA_NVRTC_INCOMPATIBILITY" 51 "TorchCodec/vLLM requested an incompatible NVRTC shared library; inspect targeted runtime log"
    else
        fail "RUNTIME_MULTIMODAL_IMPORT_FAILURE" 52 "Exact multimodal execution-path smoke failed; inspect targeted runtime log"
    fi
fi
require_stopped

echo
echo "=== VERIFY PREFETCHED QWEN ==="
terminal_stage="model_weight_verification"
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
echo "=== CONTENT-ADDRESSED VALIDATED FIGURE RENDER CACHE ==="
terminal_stage="render_cache"
mkdir -p "$render_cache_root"
"$standard_runtime/bin/python" - "$source_pdf" "$output_root" "$render_cache_root" "$expected_source_sha" "$render_contract_version" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, shutil, sys, time
import pypdfium2 as pdfium
source=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); cache_root=pathlib.Path(sys.argv[3])
expected_source_sha=sys.argv[4]; contract=sys.argv[5]; started=time.perf_counter()
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != expected_source_sha: raise SystemExit(f"source SHA mismatch: {source_sha}")
pdf=pdfium.PdfDocument(str(source))
if len(pdf)!=12: raise SystemExit(f"expected 12 pages, got {len(pdf)}")
renderer_version=importlib.metadata.version("pypdfium2")
targets=[
 ("figure6_cifar_training_chart",7,[99.19828033447266,72.1703872680664,495.0450134277344,171.62620544433594]),
 ("figure3_network_architecture",3,[50.5037956237793,76.09317016601562,285.709716796875,618.7928466796875]),
]
dpi=300; scale=dpi/72.0
receipt={"schema_version":2,"render_contract_version":contract,"source_pdf_sha256":source_sha,"render_dpi":dpi,"bbox_origin":"TOPLEFT_PDF_POINTS","renderer":{"implementation":"pypdfium2","version":renderer_version},"targets":[]}
for label,index,bbox in targets:
    identity={"render_contract_version":contract,"source_pdf_sha256":source_sha,"page_index_zero_based":index,"pdf_bbox":bbox,"bbox_origin":"TOPLEFT_PDF_POINTS","dpi":dpi,"renderer":{"implementation":"pypdfium2","version":renderer_version},"scale_formula":"dpi/72.0","crop_rounding":"python_round(points*scale)","png_save_options":{}}
    cache_key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    cache_png=cache_root/f"{cache_key}.png"; cache_json=cache_root/f"{cache_key}.json"; hit=False
    target_started=time.perf_counter(); crop_sha=None; width=height=None
    if cache_png.exists() and cache_json.exists():
        try:
            prior=json.loads(cache_json.read_text(encoding="utf-8")); actual=hashlib.sha256(cache_png.read_bytes()).hexdigest()
            if prior.get("cache_key")==cache_key and prior.get("identity")==identity and prior.get("crop_sha256")==actual:
                hit=True; crop_sha=actual; width=int(prior["width_px"]); height=int(prior["height_px"])
        except Exception: hit=False
    if not hit:
        image=pdf[index].render(scale=scale).to_pil()
        left,top,right,bottom=[round(v*scale) for v in bbox]
        crop=image.crop((left,top,right,bottom)); tmp=cache_png.with_suffix(".png.tmp"); crop.save(tmp,format="PNG"); tmp.replace(cache_png)
        crop_sha=hashlib.sha256(cache_png.read_bytes()).hexdigest(); width,height=crop.width,crop.height
        payload={"schema_version":1,"cache_key":cache_key,"identity":identity,"crop_sha256":crop_sha,"width_px":width,"height_px":height}
        tmpj=cache_json.with_suffix(".json.tmp"); tmpj.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmpj.replace(cache_json)
    output_path=out/f"{label}-page-zb{index}.png"
    if output_path.exists(): output_path.unlink()
    materialization="hardlink"
    try: os.link(cache_png,output_path)
    except OSError: shutil.copy2(cache_png,output_path); materialization="copy2"
    receipt["targets"].append({"id":label,"page_index_zero_based":index,"pdf_bbox":bbox,"path":str(output_path),"cache_path":str(cache_png),"cache_key":cache_key,"cache_hit":hit,"crop_sha256":crop_sha,"width_px":width,"height_px":height,"materialization":materialization,"target_seconds":time.perf_counter()-target_started})
receipt["render_total_seconds"]=time.perf_counter()-started
receipt["generated_count"]=sum(not x["cache_hit"] for x in receipt["targets"]); receipt["reused_count"]=sum(x["cache_hit"] for x in receipt["targets"])
(out/"qwen35-render-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== RUN EXACTLY THREE TARGETED QWEN VISUAL-SEMANTIC CALLS ==="
terminal_stage="qwen_inference"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME="$vllm_cache_root"
mkdir -p "$XDG_CACHE_HOME"
echo "VLLM_SAMPLING_BACKEND=vllm_native"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "VLLM_PERSISTENT_COMPILE_CACHE=$XDG_CACHE_HOME"

set +e
"$runtime/bin/python" "$helper" \
    --model-dir "$model" \
    --figure6 "$output_root/figure6_cifar_training_chart-page-zb7.png" \
    --figure3 "$output_root/figure3_network_architecture-page-zb3.png" \
    --gpu-uuid "$gpu_uuid" \
    --baseline-gpu-used-mib "$gpu_used_mib" \
    --output-dir "$output_root" \
    --output "$report" 2>&1 | tee "$helper_log"
helper_rc=${PIPESTATUS[0]}
set -e
require_stopped
if [ "$helper_rc" -ne 0 ]; then
    if grep -q "Could not find nvcc" "$helper_log"; then
        fail "QWEN_ENGINE_SAMPLER_JIT_NVCC_MISSING" "$helper_rc" "Engine startup selected a sampler path requiring nvcc; inspect helper log"
    elif grep -q "Engine core initialization failed" "$helper_log"; then
        fail "QWEN_ENGINE_INITIALIZATION_FAILED" "$helper_rc" "vLLM engine initialization failed before Qwen inference; inspect helper log"
    else
        fail "QWEN_MEASUREMENT_REJECTED" "$helper_rc" "Qwen helper returned a non-passing measurement decision; inspect primary report"
    fi
fi
terminal_stage="complete"
status="SUCCESS"
failure_class="NONE"
failure_message=""

echo
echo "=== FINAL SAFETY STATE ==="
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "BENCHMARK_REPORT=$report"
echo "BENCHMARK_RECEIPT=$receipt"
echo "QWEN35_VISUAL_MEASUREMENT=COMPLETE"
