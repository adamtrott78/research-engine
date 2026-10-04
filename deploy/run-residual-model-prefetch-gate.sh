#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/residual-model-prefetch-$stamp"
model_root="$state/residual-models"
tooling_venv="$state/residual-prefetch-tooling/huggingface-hub-2.1.1"
wheelhouse="$state/residual-runtime-wheelhouse"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
contract="$root/benchmarks/document_intelligence/residual-model-prefetch-contract.v1.json"
source_receipt="$root/benchmarks/document_intelligence/residual-source-audit-pass.v1.json"
helper="$root/benchmarks/document_intelligence/prefetch_residual_models.py"
report="$output_root/model-prefetch-report.json"
hardware_report="$output_root/hardware-compatibility.json"
wheel_report="$output_root/runtime-wheel-prefetch.json"

fail() {
    echo "ERROR=$1" >&2
    exit "${2:-1}"
}

require_stopped() {
    local active sub pid
    active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
    pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
    [ "$active" = "inactive" ] && [ "$sub" = "dead" ] && [ "${pid:-0}" = "0" ]         || fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
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
[ -f "$contract" ] || fail "PREFETCH_CONTRACT_NOT_FOUND" 32
[ -f "$source_receipt" ] || fail "SOURCE_GATE_RECEIPT_NOT_FOUND" 33
[ -f "$helper" ] || fail "PREFETCH_HELPER_NOT_FOUND" 34
[ -x "$standard_runtime/bin/python" ] || fail "STANDARD_RUNTIME_NOT_FOUND" 35

mkdir -p "$output_root" "$model_root" "$wheelhouse" "$(dirname "$tooling_venv")"

echo "=== RESIDUAL MODEL PREFETCH / LICENSE / CHECKSUM GATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "MODEL_ROOT=$model_root"
echo "CANDIDATE_MODEL_INFERENCE_CALLS=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"

python3 - "$source_receipt" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["decision"] == "PASS"
assert r["model_prefetch_ready"] is True
assert r["candidate_model_downloads"] == 0
assert r["candidate_model_inference_calls"] == 0
assert r["scientific_state_materialized"] is False
print("SOURCE_GATE_RECEIPT=PASS")
PY
require_stopped

echo
echo "=== HARDWARE / HOST COMPATIBILITY FACTS (NO CANDIDATE LOAD) ==="
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,compute_mode --format=csv,noheader,nounits     > "$output_root/nvidia-smi.csv"
"$standard_runtime/bin/python" - "$hardware_report" <<'PY'
import json, os, platform, shutil, sys
import torch

devices=[]
for idx in range(torch.cuda.device_count()):
    props=torch.cuda.get_device_properties(idx)
    devices.append({
        "index": idx,
        "name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(idx)),
        "total_memory_bytes": props.total_memory,
    })
mem_total=None
try:
    with open("/proc/meminfo", encoding="utf-8") as f:
        first=f.readline().split()
        mem_total=int(first[1])*1024
except Exception:
    pass
payload={
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda_version": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "devices": devices,
    "host_ram_bytes": mem_total,
    "platform": platform.platform(),
}
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["cuda_available"] or payload["device_count"] < 1:
    raise SystemExit(2)
PY
require_stopped

echo
echo "=== PREFETCH TOOLING VENV ==="
if [ ! -x "$tooling_venv/bin/python" ]; then
    python3 -m venv "$tooling_venv"
fi
"$tooling_venv/bin/python" -m pip install --upgrade pip
"$tooling_venv/bin/python" -m pip install "huggingface-hub==2.1.1"
"$tooling_venv/bin/python" - <<'PY'
import huggingface_hub
assert huggingface_hub.__version__ == "2.1.1", huggingface_hub.__version__
print("HUGGINGFACE_HUB_VERSION=2.1.1")
PY
"$tooling_venv/bin/python" -m pip freeze > "$output_root/prefetch-tooling-freeze.txt"
require_stopped

echo
echo "=== PIN PADDLEOCR 3.7.0 WHEEL (NO INSTALL) ==="
tmp_wheel_dir="$output_root/paddleocr-wheel"
mkdir -p "$tmp_wheel_dir"
"$tooling_venv/bin/python" -m pip download     --no-deps --only-binary=:all:     --dest "$tmp_wheel_dir"     "paddleocr==3.7.0"
wheel="$(find "$tmp_wheel_dir" -maxdepth 1 -type f -name 'paddleocr-3.7.0-*.whl' -print)"
[ "$(printf '%s\n' "$wheel" | sed '/^$/d' | wc -l)" -eq 1 ] || fail "EXPECTED_ONE_PADDLEOCR_WHEEL" 40
wheel_sha="$(sha256sum "$wheel" | awk '{print $1}')"
[ "$wheel_sha" = "c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338" ]     || fail "PADDLEOCR_WHEEL_SHA256_MISMATCH" 41
cp -f "$wheel" "$wheelhouse/"
python3 - "$wheel" "$wheel_report" <<'PY'
import hashlib, json, pathlib, sys
p=pathlib.Path(sys.argv[1])
payload={
    "package":"paddleocr",
    "version":"3.7.0",
    "filename":p.name,
    "bytes":p.stat().st_size,
    "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
    "installed":False,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== PREFETCH SELECTED MODEL ARTIFACTS ==="
"$tooling_venv/bin/python" "$helper"     --contract "$contract"     --model-root "$model_root"     --output "$report"
require_stopped

echo
echo "=== FINAL PREFETCH SAFETY CHECK ==="
python3 - "$report" "$hardware_report" "$wheel_report" <<'PY'
import json, sys
model=json.load(open(sys.argv[1],encoding="utf-8"))
hardware=json.load(open(sys.argv[2],encoding="utf-8"))
wheel=json.load(open(sys.argv[3],encoding="utf-8"))
checks={
    "model_prefetch_pass": model["decision"] == "PASS",
    "four_model_artifacts": len(model["artifacts"]) == 4,
    "three_candidate_families": len(set(a["family"] for a in model["artifacts"])) == 3,
    "all_offline_reopen": all(a["offline_reopen"] for a in model["artifacts"]),
    "all_apache_2": all(a["license"] == "apache-2.0" for a in model["artifacts"]),
    "candidate_inference_zero": model["candidate_model_inference_calls"] == 0,
    "scientific_state_false": model["scientific_state_materialized"] is False,
    "cuda_visible": hardware["cuda_available"] is True and hardware["device_count"] >= 1,
    "paddleocr_wheel_hash": wheel["sha256"] == "c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338",
    "paddleocr_not_installed_by_gate": wheel["installed"] is False,
}
payload={"decision":"PASS" if all(checks.values()) else "REJECT","checks":checks}
print(json.dumps(payload,indent=2,sort_keys=True))
if not all(checks.values()):
    raise SystemExit(2)
PY
require_stopped

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "CANDIDATE_MODEL_INFERENCE_CALLS=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "OUTPUT_ROOT=$output_root"
echo "MODEL_PREFETCH_REPORT=$report"
echo "RESIDUAL_MODEL_PREFETCH_GATE=PASS"
