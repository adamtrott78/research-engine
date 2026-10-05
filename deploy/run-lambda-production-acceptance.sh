#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
envf="$state/research-engine.env"
core_venv="$state/venv"
native_runtime="$adam/.document-runtimes/docling-native"
standard_runtime="$state/document-runtimes/docling-standard-no-ocr"
standard_cache="$state/document-models/docling-standard-no-ocr-2.131.0"
formula_runtime="$state/document-runtimes/docling-formula-enrichment"
formula_cache="$state/document-models/docling-formula-enrichment-2.131.0"
table_runtime="$state/document-runtimes/paddle-table-recognition-v2"
table_cache="$state/document-models/paddle-table-recognition-v2-3.7.0"
qwen_runtime="$state/residual-runtimes/qwen3.5-4b-vllm-0.30.0"
qwen_ready="$qwen_runtime/.workspace-control-ready.json"
qwen_model="$state/residual-models/qwen3.5-4b/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/lambda-production-acceptance-$stamp"
report="$output_root/lambda-production-acceptance-report.json"

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

cleanup() {
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== LAMBDA DOCUMENT PIPELINE PRODUCTION ACCEPTANCE ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$core_venv/bin/research-engine" ] || fail "CORE_CLI_NOT_FOUND" 33
[ -f "$envf" ] || fail "PRODUCTION_ENV_NOT_FOUND" 34
[ -d "$state/artifacts" ] || fail "ARTIFACT_ROOT_NOT_FOUND" 35
mkdir -p "$visible_tmp_root" "$output_root"

echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$output_root"

echo
echo "=== SYSTEMD PRODUCTION UNIT ==="
fragment="$(systemctl --user show "$unit" -p FragmentPath --value 2>/dev/null || true)"
[ -n "$fragment" ] || fail "SYSTEMD_UNIT_FRAGMENT_NOT_FOUND" 40
[ -f "$fragment" ] || fail "SYSTEMD_UNIT_FILE_NOT_FOUND" 41
cmp -s "$fragment" "$root/deploy/systemd/research-engine-daemon.service" \
    || fail "SYSTEMD_UNIT_DRIFT_FROM_REPOSITORY" 42
echo "SYSTEMD_UNIT=MATCH"
echo "SYSTEMD_FRAGMENT=$fragment"
require_stopped

echo
echo "=== PRODUCTION ENV CONTRACT ==="
if grep -Eq '^[[:space:]]*OPENAI_API_KEY=' "$envf"; then
    fail "OPENAI_API_KEY_FORBIDDEN_IN_PRODUCTION_ENV" 43
fi
if grep -Eq 'sk-[A-Za-z0-9_-]{8,}' "$envf"; then
    fail "POSSIBLE_API_KEY_FOUND_IN_PRODUCTION_ENV" 44
fi
set -a
. "$envf"
set +a

[ "${RESEARCH_ENGINE_SEMANTIC_BACKEND:-}" = "codex" ] || fail "SEMANTIC_BACKEND_NOT_CODEX" 45
[ "${RESEARCH_ENGINE_VERIFY_PAGE_RADIUS:-}" = "0" ] || fail "VERIFY_PAGE_RADIUS_NOT_ZERO" 46
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED:-}" = "1" ] || fail "LOCAL_INFERENCE_NOT_ENABLED" 47
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING:-}" = "8" ] || fail "LOCAL_INFERENCE_QUEUE_NOT_8" 48
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS:-}" = "900" ] || fail "LOCAL_INFERENCE_TIMEOUT_NOT_900" 49
[ "${RESEARCH_ENGINE_FORMULA_GPU_INDEX:-}" = "0" ] || fail "FORMULA_GPU_NOT_0" 50
[ "${RESEARCH_ENGINE_TABLE_GPU_INDEX:-}" = "1" ] || fail "TABLE_GPU_NOT_1" 51
[ "${RESEARCH_ENGINE_QWEN_VERIFY_ENABLED:-}" = "1" ] || fail "QWEN_VERIFY_NOT_ENABLED" 52
[ "${RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX:-}" = "1" ] || fail "QWEN_VERIFY_GPU_NOT_1" 53
[ "${RESEARCH_ENGINE_QWEN_VERIFY_MAX_CONTEXT_CHARS:-}" = "20000" ] || fail "QWEN_VERIFY_CONTEXT_NOT_20000" 54
[ "${RESEARCH_ENGINE_QWEN_VERIFY_TIMEOUT_SECONDS:-}" = "900" ] || fail "QWEN_VERIFY_TIMEOUT_NOT_900" 55
[ "${RESEARCH_ENGINE_DETERMINISTIC_WORKERS:-}" = "1" ] || fail "DETERMINISTIC_WORKERS_NOT_1" 56
[ "${RESEARCH_ENGINE_CHEAP_WORKERS:-}" = "1" ] || fail "CHEAP_WORKERS_NOT_1" 57
[ "${RESEARCH_ENGINE_STANDARD_WORKERS:-}" = "1" ] || fail "STANDARD_WORKERS_NOT_1" 58
[ "${RESEARCH_ENGINE_STRONG_WORKERS:-}" = "1" ] || fail "STRONG_WORKERS_NOT_1" 59
[ -n "${RESEARCH_ENGINE_CODEX_BIN:-}" ] || fail "CODEX_BIN_NOT_CONFIGURED" 60
[ -x "$RESEARCH_ENGINE_CODEX_BIN" ] || fail "CODEX_BIN_NOT_EXECUTABLE" 61

echo "SEMANTIC_BACKEND=$RESEARCH_ENGINE_SEMANTIC_BACKEND"
echo "VERIFY_PAGE_RADIUS=$RESEARCH_ENGINE_VERIFY_PAGE_RADIUS"
echo "LOCAL_INFERENCE_ENABLED=$RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED"
echo "FORMULA_GPU_INDEX=$RESEARCH_ENGINE_FORMULA_GPU_INDEX"
echo "TABLE_GPU_INDEX=$RESEARCH_ENGINE_TABLE_GPU_INDEX"
echo "QWEN_VERIFY_ENABLED=$RESEARCH_ENGINE_QWEN_VERIFY_ENABLED"
echo "QWEN_VERIFY_GPU_INDEX=$RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX"
"$RESEARCH_ENGINE_CODEX_BIN" --version

export RESEARCH_ENGINE_DATABASE_URL="postgresql://research_engine:research_engine@localhost:5432/research_engine"
export RESEARCH_ENGINE_ARTIFACT_ROOT="$state/artifacts"

echo
echo "=== POSTGRESQL DURABLE STATE ==="
sudo docker compose -f docker-compose.yml up -d db
sudo docker compose -f docker-compose.yml exec -T db \
    pg_isready -U research_engine -d research_engine
require_stopped

"$core_venv/bin/python" - "$output_root/db-safety.json" <<'PY'
import json
import os
import sys
import psycopg

dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
out = sys.argv[1]
with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status,count(*)::int
            FROM research_runs
            GROUP BY status
            ORDER BY status
            """
        )
        run_counts = dict(cur.fetchall())
        cur.execute(
            """
            SELECT executor_tier,status,count(*)::int
            FROM research_jobs
            WHERE status IN ('QUEUED','RUNNING')
            GROUP BY executor_tier,status
            ORDER BY executor_tier,status
            """
        )
        pending = [
            {"executor_tier": r[0], "status": r[1], "count": r[2]}
            for r in cur.fetchall()
        ]
        cur.execute(
            "SELECT count(*)::int FROM research_runs WHERE status='RUNNING'"
        )
        active_runs = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT count(*)::int
            FROM research_jobs
            WHERE status IN ('QUEUED','RUNNING')
              AND executor_tier <> 'HUMAN'
            """
        )
        pending_nonhuman = int(cur.fetchone()[0])

payload = {
    "run_counts": run_counts,
    "pending_jobs": pending,
    "active_running_runs": active_runs,
    "queued_or_running_nonhuman_jobs": pending_nonhuman,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
if active_runs:
    raise SystemExit("ACTIVE_RUNNING_RESEARCH_RUN_PRESENT")
if pending_nonhuman:
    raise SystemExit("QUEUED_OR_RUNNING_NONHUMAN_WORK_PRESENT")
PY

echo
echo "=== INSTALLED DOCUMENT RUNTIMES / CACHES ==="
for path in \
    "$native_runtime/bin/python" \
    "$standard_runtime/bin/python" \
    "$standard_cache" \
    "$formula_runtime/bin/python" \
    "$formula_cache" \
    "$table_runtime/bin/python" \
    "$table_cache" \
    "$qwen_runtime/bin/python" \
    "$qwen_ready" \
    "$qwen_model/model.safetensors-00001-of-00002.safetensors" \
    "$qwen_model/model.safetensors-00002-of-00002.safetensors"
do
    [ -e "$path" ] || fail "MISSING_DOCUMENT_RUNTIME_OR_CACHE:$path" 60
done
echo "NATIVE_RUNTIME=$native_runtime"
echo "STANDARD_RUNTIME=$standard_runtime"
echo "FORMULA_RUNTIME=$formula_runtime"
echo "TABLE_RUNTIME=$table_runtime"
echo "QWEN_VERIFY_RUNTIME=$qwen_runtime"
echo "FORMULA_CACHE_SIZE=$(du -sh "$formula_cache" | awk '{print $1}')"
echo "TABLE_CACHE_SIZE=$(du -sh "$table_cache" | awk '{print $1}')"
echo "QWEN_MODEL_SIZE=$(du -sh "$qwen_model" | awk '{print $1}')"

echo
echo "=== GPU INVENTORY / BOUNDED ALLOCATION PROBES ==="
command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA_SMI_NOT_FOUND" 61
nvidia-smi \
    --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
    --format=csv,noheader,nounits
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
[ "$gpu_count" -ge 2 ] || fail "PRODUCTION_REQUIRES_TWO_VISIBLE_GPUS" 62

CUDA_VISIBLE_DEVICES="$RESEARCH_ENGINE_FORMULA_GPU_INDEX" \
"$formula_runtime/bin/python" - "$output_root/formula-gpu-probe.json" <<'PY'
import json
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("FORMULA_CUDA_UNAVAILABLE")
device = torch.device("cuda:0")
probe = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
probe.fill_(1)
torch.cuda.synchronize(device)
payload = {
    "cuda_available": True,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(device),
    "allocation_probe_mib": 128,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

CUDA_VISIBLE_DEVICES="$RESEARCH_ENGINE_TABLE_GPU_INDEX" \
"$table_runtime/bin/python" - "$output_root/table-gpu-probe.json" <<'PY'
import json
import sys
import paddle

if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
    raise SystemExit("TABLE_CUDA_UNAVAILABLE")
paddle.set_device("gpu:0")
probe = paddle.ones([32, 1024, 1024], dtype="float32")
payload = {
    "compiled_with_cuda": True,
    "device_count": paddle.device.cuda.device_count(),
    "allocation_probe_mib": int(probe.numel() * 4 / 1024 / 1024),
    "mean": float(probe.mean()),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
require_stopped

echo
echo "=== POLICY / LICENSE / BROKER BOOTSTRAP ==="
"$core_venv/bin/python" - \
    "$root/benchmarks/document_intelligence/localization-matrix.v2.json" \
    "$root/benchmarks/document_intelligence/codeformulav2-license-review.v1.json" \
    "$output_root/policy-broker.json" <<'PY'
import json
import os
import sys

from research_engine.local_inference import local_inference_service_from_env
from research_engine.orchestration import JOB_TIERS, ExecutorTier, JobType

matrix_path, license_path, output_path = sys.argv[1:]
matrix = json.load(open(matrix_path, encoding="utf-8"))
license_review = json.load(open(license_path, encoding="utf-8"))
rows = {
    JobType(row["job_type"]): row
    for row in matrix["job_type_decisions"]
}
if set(rows) != set(JobType):
    raise SystemExit("LOCALIZATION_MATRIX_JOB_COVERAGE_MISMATCH")
for job_type, tier in JOB_TIERS.items():
    if rows[job_type]["executor_tier"] != str(tier):
        raise SystemExit(f"LOCALIZATION_MATRIX_TIER_MISMATCH:{job_type}")
if JOB_TIERS[JobType.EXTRACT] != ExecutorTier.STANDARD:
    raise SystemExit("EXTRACT_NOT_STANDARD")
if JOB_TIERS[JobType.VERIFY] != ExecutorTier.STANDARD:
    raise SystemExit("VERIFY_NOT_STANDARD")
if "AUTHORITATIVE_PDF" not in rows[JobType.EXTRACT]["production_placement"]:
    raise SystemExit("EXTRACT_NOT_AUTHORITATIVE_PDF")
if rows[JobType.VERIFY]["production_placement"] != "BOUNDED_LOCAL_QWEN_PRIMARY_WITH_REMOTE_CODEX_FALLTHROUGH":
    raise SystemExit("VERIFY_BOUNDED_LOCAL_ROUTE_NOT_PROMOTED")
if rows[JobType.VERIFY]["localization_decision"] != "PROMOTE_BOUNDED_LOCAL_VERIFY":
    raise SystemExit("VERIFY_LOCALIZATION_DECISION_NOT_PROMOTED")
if "authoritative" not in rows[JobType.VERIFY]["authority"].lower():
    raise SystemExit("VERIFY_AUTHORITY_NOT_PRESERVED")
if license_review["decision"] != "PASS_INTERNAL_DEPLOYMENT_WITH_SHARING_OBLIGATION":
    raise SystemExit("CODEFORMULA_LICENSE_REVIEW_NOT_ACCEPTED")

service = local_inference_service_from_env(require_enabled=True)
health = service.health()
try:
    if health["formula"]["spawn_count"] != 0:
        raise SystemExit("FORMULA_OWNER_SPAWNED_DURING_BOOTSTRAP")
    if health["table"]["spawn_count"] != 0:
        raise SystemExit("TABLE_OWNER_SPAWNED_DURING_BOOTSTRAP")
    if health["qwen_verify"]["spawn_count"] != 0:
        raise SystemExit("QWEN_VERIFY_OWNER_SPAWNED_DURING_BOOTSTRAP")
finally:
    service.close()

payload = {
    "localization_matrix_status": matrix["status"],
    "extract_tier": str(JOB_TIERS[JobType.EXTRACT]),
    "verify_tier": str(JOB_TIERS[JobType.VERIFY]),
    "license_decision": license_review["decision"],
    "broker_constructed_without_model_process_spawn": True,
    "bounded_verify_route": rows[JobType.VERIFY]["production_placement"],
    "qwen_verify_owner_zero_spawn": health["qwen_verify"]["spawn_count"] == 0,
    "broker_initial_health": health,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
require_stopped

echo
echo "=== AUTHORITATIVE PDF HYDRATION SMOKE ==="
"$core_venv/bin/python" - "$output_root/pdf-hydration-smoke.json" <<'PY'
import json
import os
import sys

from research_engine.postgres import PostgresStore
from research_engine.semantic_pipeline import ResearchContextHydrator

store = PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"])
with store.connect() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM source_artifacts
            WHERE lower(coalesce(media_type,''))='application/pdf'
               OR lower(coalesce(format,''))='pdf'
            ORDER BY id
            LIMIT 1
            """
        )
        row = cur.fetchone()
if row is None:
    raise SystemExit("NO_PDF_SOURCE_ARTIFACT_FOR_HYDRATION_SMOKE")

artifact_id = str(row[0])
hydrator = ResearchContextHydrator(
    store=store,
    artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"],
    verify_page_radius=0,
)
context = hydrator._artifact_context(artifact_id)
pages = list(context.get("pages") or [])
if not pages:
    raise SystemExit("AUTHORITATIVE_PDF_HYDRATION_RETURNED_NO_PAGES")
payload = {
    "artifact_id": artifact_id,
    "pages": len(pages),
    "characters": sum(len(str(page.get("text") or "")) for page in pages),
    "verify_page_radius": hydrator.verify_page_radius,
    "source_sha256": context["source_artifact"]["sha256"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
require_stopped

echo
echo "=== COMPOSE / READ-ONLY HEALTH ==="
sudo docker compose -f docker-compose.yml config --quiet
"$core_venv/bin/research-engine" overnight-report --hours 0.01 \
    > "$output_root/read-only-overnight-report.json"
require_stopped

echo
echo "=== BUILD ACCEPTANCE REPORT ==="
"$core_venv/bin/python" - "$output_root" "$report" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_path = Path(sys.argv[2])
head = sys.argv[3]

def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

payload = {
    "schema_version": 1,
    "decision": "PASS",
    "research_engine_head": head,
    "daemon_final_required_state": "inactive/dead PID 0",
    "db_safety": load("db-safety.json"),
    "formula_gpu_probe": load("formula-gpu-probe.json"),
    "table_gpu_probe": load("table-gpu-probe.json"),
    "policy_broker": load("policy-broker.json"),
    "pdf_hydration_smoke": load("pdf-hydration-smoke.json"),
    "semantic_calls_performed": 0,
    "model_server_processes_spawned": 0,
    "scientific_state_materialized": False,
}
report_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "OUTPUT_ROOT=$output_root"
echo "REPORT=$report"
echo "LAMBDA_PRODUCTION_ACCEPTANCE=PASS"
