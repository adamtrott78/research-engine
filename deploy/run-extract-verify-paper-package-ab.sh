#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
native_runtime="$adam/.document-runtimes/docling-native"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
standard_models="$state/document-models/docling-standard-no-ocr-2.131.0"
native_batch="$root/benchmarks/document_intelligence/runtimes/docling_native_batch.py"
standard_batch="$root/benchmarks/document_intelligence/runtimes/docling_standard_no_ocr_batch.py"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/extract-verify-paper-package-ab-$stamp"
plan="$output_root/ab-plan.json"
compile_plan="$output_root/compile-plan.json"
source_links="$output_root/source-links"
raw_native="$output_root/raw/native"
raw_standard="$output_root/raw/standard"
packages_native="$output_root/packages/native"
packages_standard="$output_root/packages/standard"
preflight="$output_root/zero-model-preflight.json"
replay="$output_root/representative-semantic-replay.json"

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

probe_standard_gpu() {
    local index="$1"
    CUDA_VISIBLE_DEVICES="$index" "$standard_runtime/bin/python" - <<'PY'
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
}

echo "=== EXTRACT / VERIFY PAPER-PACKAGE A/B ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$native_runtime/bin/python" ] || fail "DOCLING_NATIVE_RUNTIME_NOT_FOUND" 33
[ -x "$standard_runtime/bin/python" ] || fail "DOCLING_STANDARD_RUNTIME_NOT_FOUND" 34
[ -d "$standard_models" ] || fail "DOCLING_STANDARD_MODEL_CACHE_NOT_FOUND" 35
[ -f "$native_batch" ] || fail "NATIVE_BATCH_SHIM_NOT_FOUND" 36
[ -f "$standard_batch" ] || fail "STANDARD_BATCH_SHIM_NOT_FOUND" 37

mkdir -p     "$visible_tmp_root"     "$output_root"     "$source_links"     "$raw_native"     "$raw_standard"     "$packages_native"     "$packages_standard"

export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"
export RESEARCH_ENGINE_ARTIFACT_ROOT="${RESEARCH_ENGINE_ARTIFACT_ROOT:-$state/artifacts}"
export RESEARCH_ENGINE_SEMANTIC_BACKEND=codex

echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"
echo "NATIVE_RUNTIME=$native_runtime"
echo "STANDARD_RUNTIME=$standard_runtime"
echo "STANDARD_MODELS=$standard_models"

echo
echo "=== REFRESH CORE PACKAGE ONLY ==="
"$core_venv/bin/python" -m pip install -e "$root"
require_stopped

echo
echo "=== DISCOVER PRESERVED NINE-PAPER RUN ==="
"$core_venv/bin/research-engine" paper-package-ab-plan     --expected-papers 9     > "$plan"

run_id="$("$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["source_run_id"])
PY
)"
paper_count="$("$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(len(d["source_artifacts"]))
PY
)"
[ "$paper_count" -eq 9 ] || fail "EXPECTED_EXACTLY_NINE_SOURCE_ARTIFACTS" 40
echo "SOURCE_RUN_ID=$run_id"
echo "SOURCE_PAPERS=$paper_count"

echo
echo "=== VERIFY SOURCE BYTES + CREATE VISIBLE PDF SYMLINK PLAN ==="
"$core_venv/bin/python" - "$plan" "$compile_plan" "$source_links" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

plan_path, output_path, link_root = map(Path, sys.argv[1:])
data = json.loads(plan_path.read_text(encoding="utf-8"))
link_root.mkdir(parents=True, exist_ok=True)

for row in data["source_artifacts"]:
    source = Path(row["source_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    h = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    if h.hexdigest() != row["sha256"]:
        raise ValueError(f"SHA mismatch: {source}")
    if size != int(row["byte_size"]):
        raise ValueError(f"byte-size mismatch: {source}")
    link = link_root / f'{row["sha256"]}.pdf'
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(source, link)
    row["compile_path"] = str(link)

output_path.write_text(
    json.dumps(data, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"COMPILE_PLAN_SOURCES={len(data['source_artifacts'])}")
PY

echo
echo "=== DOCLING NATIVE BATCH: ONE CONVERTER / NINE PDFS ==="
"$native_runtime/bin/python" "$native_batch"     "$compile_plan"     --output-root "$raw_native"
require_stopped

echo
echo "=== SELECT HEALTHY GPU FOR STANDARD STRUCTURE BATCH ==="
command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA_SMI_NOT_FOUND" 45
nvidia-smi     --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode     --format=csv,noheader,nounits

mapfile -t gpu_indices < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits     | awk -F',' '{
        gsub(/^ +| +$/,"",$1);
        gsub(/^ +| +$/,"",$2);
        print $2 "|" $1
    }' | sort -t'|' -k1,1nr | cut -d'|' -f2
)
selected_gpu=""
for index in "${gpu_indices[@]}"; do
    echo "STANDARD_GPU_PROBE=$index"
    if probe_standard_gpu "$index"; then
        selected_gpu="$index"
        break
    fi
done
[ -n "$selected_gpu" ] || fail "NO_USABLE_STANDARD_DOCLING_GPU" 46
echo "STANDARD_PHYSICAL_GPU_INDEX=$selected_gpu"

export CUDA_VISIBLE_DEVICES="$selected_gpu"
export RESEARCH_ENGINE_DOCLING_DEVICE=cuda
export RESEARCH_ENGINE_DOCLING_ARTIFACTS="$standard_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo
echo "=== DOCLING STANDARD/NO-OCR BATCH: ONE CONVERTER / NINE PDFS ==="
"$standard_runtime/bin/python" "$standard_batch"     "$compile_plan"     --output-root "$raw_standard"
require_stopped

echo
echo "=== NORMALIZE BOTH ARMS TO PAPER-PACKAGE.V1 ==="
"$core_venv/bin/python" -     "$compile_plan"     "$raw_native"     "$raw_standard"     "$packages_native"     "$packages_standard" <<'PY'
import json
import sys
from pathlib import Path

from research_engine.document_docling import normalize_docling_document

plan_path, raw_native, raw_standard, pkg_native, pkg_standard = map(
    Path, sys.argv[1:]
)
plan = json.loads(plan_path.read_text(encoding="utf-8"))
for row in plan["source_artifacts"]:
    sha = str(row["sha256"])
    source = str(row["source_path"])
    artifact_id = str(row["id"])
    normalize_docling_document(
        docling_json_path=raw_native / sha / "document.json",
        source_pdf_path=source,
        source_artifact_id=artifact_id,
        source_pdf_sha256=sha,
        output_directory=pkg_native / sha / "paper-package-v1",
        docling_version="2.131.0",
        docling_mode="native",
    )
    normalize_docling_document(
        docling_json_path=raw_standard / sha / "document.json",
        source_pdf_path=source,
        source_artifact_id=artifact_id,
        source_pdf_sha256=sha,
        output_directory=pkg_standard / sha / "paper-package-v1",
        docling_version="2.131.0",
        docling_mode="standard_no_ocr",
    )
print(f"NORMALIZED_PACKAGE_PAIRS={len(plan['source_artifacts'])}")
PY
require_stopped

echo
echo "=== ZERO-MODEL PROVENANCE / QUOTE RECOVERY PREFLIGHT ==="
"$core_venv/bin/research-engine" paper-package-ab-preflight     "$run_id"     --native-package-root "$packages_native"     --standard-package-root "$packages_standard"     --expected-papers 9     > "$preflight"
require_stopped

echo
echo "=== COMPACT ZERO-MODEL PREFLIGHT ==="
"$core_venv/bin/python" - "$preflight" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "decision": d["decision"],
    "package_coverage": d["package_coverage"],
    "historical_evidence_quotes": d["historical_evidence_quotes"],
    "quotes_passing": d["quotes_passing"],
    "quote_recovery_fraction": d["quote_recovery_fraction"],
    "pdf_context_chars_total": d["pdf_context_chars_total"],
    "package_context_chars_total": d["package_context_chars_total"],
    "context_char_reduction_fraction": d["context_char_reduction_fraction"],
}, indent=2, sort_keys=True))
PY

preflight_decision="$("$core_venv/bin/python" - "$preflight" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"
[ "$preflight_decision" = "PASS" ] || fail "PAPER_PACKAGE_ZERO_MODEL_PREFLIGHT_FAILED" 60

echo
echo "=== REPRESENTATIVE SEMANTIC A/B: LOW / MEDIAN / HIGH EXTRACT + VERIFY ==="
echo "Scientific state is not materialized; only operational calibration events are recorded."
"$core_venv/bin/research-engine" paper-package-ab-replay     "$run_id"     --native-package-root "$packages_native"     --standard-package-root "$packages_standard"     > "$replay"
require_stopped

echo
echo "=== COMPACT REPRESENTATIVE SEMANTIC A/B ==="
"$core_venv/bin/python" - "$replay" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
rows=[]
for row in d["rows"]:
    rows.append({
        "source_job_id": row["source_job_id"],
        "job_type": row["job_type"],
        "classification": row["classification"],
        "agreement_pass": row["paper_package"]["agreement_pass"],
        "historical_input_tokens": row["historical"]["input_tokens"],
        "package_input_tokens": row["paper_package"]["usage"]["input_tokens"],
        "input_reduction_fraction": row["input_reduction_fraction"],
        "pdf_control_ran": row["pdf_control"] is not None,
        "pdf_control_agreement_pass": (
            row["pdf_control"]["agreement_pass"]
            if row["pdf_control"] is not None
            else None
        ),
    })
print(json.dumps({
    "source_run_id": d["source_run_id"],
    "jobs": d["jobs"],
    "package_agreement_passes": d["package_agreement_passes"],
    "package_agreement_rate": d["package_agreement_rate"],
    "package_context_regression_signals": d[
        "package_context_regression_signals"
    ],
    "historical_input_tokens": d["historical_input_tokens"],
    "paper_package_input_tokens": d["paper_package_input_tokens"],
    "input_reduction_fraction": d["input_reduction_fraction"],
    "rows": rows,
}, indent=2, sort_keys=True))
PY

regressions="$("$core_venv/bin/python" - "$replay" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["package_context_regression_signals"])
PY
)"
agreement_rate="$("$core_venv/bin/python" - "$replay" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["package_agreement_rate"])
PY
)"

decision="$("$core_venv/bin/python" - "$replay" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
regressions=int(d["package_context_regression_signals"])
rate=float(d["package_agreement_rate"])
if regressions:
    print("FAIL_CONTEXT_REGRESSION")
elif rate >= (5/6):
    print("PASS_CANDIDATE")
else:
    print("INCONCLUSIVE_MODEL_VARIANCE")
PY
)"

echo "REPRESENTATIVE_AB_DECISION=$decision"
echo "PACKAGE_CONTEXT_REGRESSION_SIGNALS=$regressions"
echo "PACKAGE_AGREEMENT_RATE=$agreement_rate"

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "SOURCE_RUN_ID=$run_id"
echo "STANDARD_PHYSICAL_GPU_INDEX=$selected_gpu"
echo "OUTPUT_ROOT=$output_root"
echo "PLAN=$plan"
echo "ZERO_MODEL_PREFLIGHT=$preflight"
echo "SEMANTIC_REPLAY=$replay"
echo "PAPER_PACKAGE_AB_GATE=MEASURED"
