#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="$HOME/adamArchives/Adam/.research-engine"
runtime="$state/document-runtimes/docling-standard-no-ocr"
unit="research-engine-daemon.service"

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

systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$runtime/bin/python" ] || fail "DOCLING_RUNTIME_NOT_FOUND" 32

echo "=== CUDA HOST DIAGNOSTIC ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "USER=$(id -un)"
echo "GROUPS=$(id -Gn)"
echo "KERNEL=$(uname -srvo)"
echo "ARCH=$(uname -m)"
echo "CMDLINE=$(cat /proc/cmdline 2>/dev/null || true)"

echo
echo "=== NVIDIA DRIVER ==="
cat /proc/driver/nvidia/version 2>/dev/null || true
nvidia-smi || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,compute_mode --format=csv,noheader,nounits || true

echo
echo "=== NVIDIA KERNEL MODULES ==="
lsmod | grep -E '^(nvidia|nvidia_uvm|nvidia_drm|nvidia_modeset)' || true
echo "NVIDIA_UVM_PROC_DEVICE=$(grep -E 'nvidia-uvm' /proc/devices 2>/dev/null || true)"
if [ -d /sys/module/nvidia_uvm ]; then
    echo "NVIDIA_UVM_MODULE=loaded"
    if [ -r /sys/module/nvidia_uvm/parameters/uvm_disable_hmm ]; then
        echo "NVIDIA_UVM_DISABLE_HMM=$(cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm)"
    else
        echo "NVIDIA_UVM_DISABLE_HMM=<parameter-not-exposed>"
    fi
else
    echo "NVIDIA_UVM_MODULE=not-loaded"
fi

echo
echo "=== NVIDIA DEVICE NODES ==="
for node in /dev/nvidiactl /dev/nvidia0 /dev/nvidia1 /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    if [ -e "$node" ]; then
        stat -c '%A %U %G %t:%T %n' "$node"
    else
        echo "MISSING=$node"
    fi
done

echo
echo "=== LIBCUDA RESOLUTION ==="
ldconfig -p 2>/dev/null | grep -E 'libcuda\.so(\.1)? ' || true
"$runtime/bin/python" - <<'PY'
import ctypes
import ctypes.util
import json

out = {"find_library_cuda": ctypes.util.find_library("cuda")}
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    out["load_error"] = str(exc)
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(2)

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
lib.cuGetErrorName.restype = ctypes.c_int
lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
lib.cuGetErrorString.restype = ctypes.c_int
lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.cuDriverGetVersion.restype = ctypes.c_int
lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.cuDeviceGetCount.restype = ctypes.c_int

def decode_error(code: int):
    name = ctypes.c_char_p()
    text = ctypes.c_char_p()
    lib.cuGetErrorName(code, ctypes.byref(name))
    lib.cuGetErrorString(code, ctypes.byref(text))
    return {
        "code": code,
        "name": name.value.decode() if name.value else None,
        "message": text.value.decode() if text.value else None,
    }

version = ctypes.c_int()
version_rc = lib.cuDriverGetVersion(ctypes.byref(version))
out["cuDriverGetVersion"] = {
    **decode_error(version_rc),
    "version": version.value,
}

init_rc = lib.cuInit(0)
out["cuInit"] = decode_error(init_rc)

if init_rc == 0:
    count = ctypes.c_int()
    count_rc = lib.cuDeviceGetCount(ctypes.byref(count))
    out["cuDeviceGetCount"] = {
        **decode_error(count_rc),
        "count": count.value,
    }

print(json.dumps(out, sort_keys=True))
PY

echo
echo "=== CUDA RUNTIME RESOLUTION ==="
"$runtime/bin/python" - <<'PY'
import ctypes
import glob
import json
import os
import site

candidates = []
for base in site.getsitepackages():
    candidates.extend(glob.glob(os.path.join(base, "nvidia", "cuda_runtime", "lib", "libcudart.so*")))
candidates = sorted(set(candidates))
out = {"candidates": candidates}
if not candidates:
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(0)

lib = ctypes.CDLL(candidates[-1])
lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.cudaGetDeviceCount.restype = ctypes.c_int
lib.cudaGetErrorName.argtypes = [ctypes.c_int]
lib.cudaGetErrorName.restype = ctypes.c_char_p
lib.cudaGetErrorString.argtypes = [ctypes.c_int]
lib.cudaGetErrorString.restype = ctypes.c_char_p

count = ctypes.c_int()
rc = lib.cudaGetDeviceCount(ctypes.byref(count))
out["cudaGetDeviceCount"] = {
    "code": rc,
    "name": lib.cudaGetErrorName(rc).decode() if lib.cudaGetErrorName(rc) else None,
    "message": lib.cudaGetErrorString(rc).decode() if lib.cudaGetErrorString(rc) else None,
    "count": count.value,
}
print(json.dumps(out, sort_keys=True))
PY

echo
echo "=== PYTORCH CUDA CONTROL ==="
env -u CUDA_VISIBLE_DEVICES "$runtime/bin/python" - <<'PY'
import json
import os
import torch

out = {
    "torch": torch.__version__,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
try:
    out["device_names"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
except Exception as exc:
    out["device_name_error"] = repr(exc)
print(json.dumps(out, sort_keys=True))
PY

echo
echo "=== RECENT NVIDIA/UVM KERNEL MESSAGES ==="
kernel_log="$(
    journalctl -k -b --no-pager 2>/dev/null         | grep -Ei 'nvidia|nvrm|uvm|xid|request_free_mem_region|hmm|iommu'         | tail -250         || true
)"
if [ -z "$kernel_log" ] && sudo -n true 2>/dev/null; then
    kernel_log="$(
        sudo -n journalctl -k -b --no-pager 2>/dev/null             | grep -Ei 'nvidia|nvrm|uvm|xid|request_free_mem_region|hmm|iommu'             | tail -250             || true
    )"
fi
if [ -n "$kernel_log" ]; then
    printf '%s\n' "$kernel_log"
else
    echo "KERNEL_LOG=unavailable-or-no-matching-lines"
fi

echo
echo "=== DIAGNOSTIC INTERPRETATION HINTS ==="
echo "If cuInit fails, the fault is below PyTorch/Docling."
echo "If /dev/nvidia-uvm is missing or inaccessible, inspect the nvidia_uvm module/device-node path."
echo "If kernel logs contain request_free_mem_region err -34 or HMM/UVM failures, compare against NVIDIA's documented KASLR/HMM initialization issue."
echo "If cuInit succeeds but cudaGetDeviceCount fails, inspect CUDA runtime/library compatibility."
echo "If both driver/runtime APIs succeed but PyTorch fails, return to the Python environment."

require_stopped
echo
echo "DAEMON=inactive/dead PID=0"
echo "CUDA_HOST_DIAGNOSTIC=PASS"
