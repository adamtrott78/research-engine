#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
envf="$state/research-engine.env"
core="$state/venv/bin/python"
runtime="$state/residual-runtimes/qwen3.5-4b-vllm-0.30.0"
ready="$runtime/.workspace-control-ready.json"
model="$state/residual-models/qwen3.5-4b/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
run_id="${RESEARCH_HUD_RUN_ID:-rf-open-world-corpus-expansion-v1}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$visible/qwen35-bounded-verify-production-gate-$stamp"
report="$out/bounded-verify-production-gate.json"

fail(){ echo "ERROR=$1" >&2; exit "${2:-1}"; }
require_stopped(){
  local active sub pid
  active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
  pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
  [ "$active" = inactive ] && [ "$sub" = dead ] && [ "${pid:-0}" = 0 ]     || fail "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" 20
}

mkdir -p "$out"
exec > >(tee "$out/operator-console.log") 2>&1

echo "=== QWEN3.5 BOUNDED VERIFY PRODUCTION GATE ==="
systemctl --user stop "$unit" >/dev/null 2>&1 || true
require_stopped
cd "$root"

[ "$(git branch --show-current)" = main ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -x "$runtime/bin/python" ] || fail "QWEN_RUNTIME_NOT_FOUND" 33
[ -f "$ready" ] || fail "QWEN_RUNTIME_CERTIFICATION_NOT_FOUND" 34
[ -f "$envf" ] || fail "PRODUCTION_ENV_NOT_FOUND" 35

echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$out"
echo "SOURCE_RUN_ID=$run_id"

set -a
. "$envf"
set +a

[ "${RESEARCH_ENGINE_SEMANTIC_BACKEND:-}" = codex ]   || fail "PRODUCTION_SEMANTIC_BACKEND_NOT_CODEX" 36
[ "${RESEARCH_ENGINE_VERIFY_PAGE_RADIUS:-}" = 0 ]   || fail "PRODUCTION_VERIFY_RADIUS_NOT_ZERO" 37
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED:-}" = 1 ]   || fail "LOCAL_INFERENCE_NOT_ENABLED" 38

# The gate opts Qwen VERIFY in ephemerally. It does not mutate the production env.
export RESEARCH_ENGINE_QWEN_VERIFY_ENABLED=1
export RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX="${RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX:-1}"
export RESEARCH_ENGINE_QWEN_VERIFY_MAX_CONTEXT_CHARS=20000
export RESEARCH_ENGINE_QWEN_VERIFY_TIMEOUT_SECONDS=900
export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"
export RESEARCH_ENGINE_ARTIFACT_ROOT="${RESEARCH_ENGINE_ARTIFACT_ROOT:-$state/artifacts}"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  fail "OPENAI_API_KEY_PRESENT_IN_CALLING_SHELL" 39
fi

echo
echo "=== EXACT QWEN RUNTIME / WEIGHTS ==="
"$runtime/bin/python" - "$ready" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected={
  "runtime_contract_version":"qwen35-vllm-cu129-image-native-sampler-v3",
  "sampling_backend":"vllm_native",
  "vllm_distribution":"0.30.0+cu129",
  "torch":"2.13.0+cu129",
  "torchcodec":"0.16.0+cu129",
  "torch_cuda_runtime":"12.9",
  "cuda_allocation_probe":"PASS",
  "vllm_multimodal_import_probe":"PASS",
}
bad={k:(v,p.get(k)) for k,v in expected.items() if p.get(k)!=v}
if bad:
    raise SystemExit("QWEN_RUNTIME_IDENTITY_MISMATCH:"+json.dumps(bad,sort_keys=True))
print("QWEN_RUNTIME_IDENTITY=PASS")
PY

for spec in \
 "model.safetensors-00001-of-00002.safetensors:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61" \
 "model.safetensors-00002-of-00002.safetensors:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
do
  name="${spec%%:*}"; sha="${spec##*:}"
  [ -f "$model/$name" ] || fail "QWEN_WEIGHT_NOT_FOUND:$name" 40
  [ "$(sha256sum "$model/$name" | awk '{print $1}')" = "$sha" ]     || fail "QWEN_WEIGHT_SHA_MISMATCH:$name" 41
done
echo "QWEN_WEIGHT_IDENTITY=PASS"
require_stopped

echo
echo "=== POSTGRESQL / QUIESCENCE ==="
sudo docker compose -f docker-compose.yml up -d db
for _ in $(seq 1 30); do
  if sudo docker compose -f docker-compose.yml exec -T db \
      pg_isready -U research_engine -d research_engine >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
sudo docker compose -f docker-compose.yml exec -T db \
  pg_isready -U research_engine -d research_engine
"$core" - <<'PY'
import os,psycopg
dsn=os.environ["RESEARCH_ENGINE_DATABASE_URL"]
with psycopg.connect(dsn) as conn:
  with conn.cursor() as cur:
    cur.execute("SELECT count(*)::int FROM research_runs WHERE status='RUNNING'")
    active=int(cur.fetchone()[0])
    cur.execute("""
      SELECT count(*)::int FROM research_jobs
      WHERE status IN ('QUEUED','RUNNING') AND executor_tier <> 'HUMAN'
    """)
    pending=int(cur.fetchone()[0])
print(f"ACTIVE_RUNNING_RUNS={active}")
print(f"QUEUED_OR_RUNNING_NONHUMAN={pending}")
if active or pending:
  raise SystemExit("RESEARCH_ENGINE_NOT_QUIESCENT")
PY
require_stopped

echo
echo "=== GPU 1 EXCLUSIVE START STATE ==="
gpu="${RESEARCH_ENGINE_QWEN_VERIFY_GPU_INDEX}"
line="$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used --format=csv,noheader,nounits | awk -F',' -v wanted="$gpu" '$1+0==wanted {print; exit}')"
[ -n "$line" ] || fail "QWEN_VERIFY_GPU_NOT_FOUND:$gpu" 42
gpu_uuid="$(printf '%s' "$line" | cut -d',' -f2 | xargs)"
gpu_free="$(printf '%s' "$line" | cut -d',' -f3 | xargs)"
[ "$gpu_free" -ge 16384 ] || fail "QWEN_VERIFY_GPU_FREE_MEMORY_LT_16384" 43
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$gpu_uuid"; then
  fail "QWEN_VERIFY_GPU_HAS_ACTIVE_COMPUTE_PROCESS" 44
fi
echo "QWEN_VERIFY_GPU_UUID=$gpu_uuid"
echo "QWEN_VERIFY_GPU_FREE_MIB=$gpu_free"

echo
echo "=== REAL MULTI-PAGE VERIFY / PAGE-SCOPED QWEN / NO-CODEX GATE ==="
PYTHONPATH="$root/src" "$core" - "$run_id" "$report" <<'PY'
from __future__ import annotations
import json,os,sys
from pathlib import Path

from research_engine.bounded_verify import (
    BoundedVerifySemanticDelegate,
    prepare_bounded_verify_pages,
)
from research_engine.local_inference import local_inference_service_from_env
from research_engine.orchestration import ExecutorTier, JobType
from research_engine.paper_package_ab import PaperPackageABPlanner, verify_agreement
from research_engine.postgres import PostgresStore
from research_engine.semantic_pipeline import ResearchContextHydrator

run_id,out=sys.argv[1:]
store=PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"])
artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
planner=PaperPackageABPlanner(store=store,artifact_root=artifact_root)
hydrator=ResearchContextHydrator(
    store=store,
    artifact_root=artifact_root,
    verify_page_radius=0,
)

selected=None
rejections=[]
for source in planner.completed_jobs(run_id,JobType.VERIFY):
    try:
        hydrated=hydrator.hydrate(source)
    except Exception as exc:
        rejections.append({
            "job_id":source.id,
            "reason":f"HYDRATE:{type(exc).__name__}",
        })
        continue
    page_scoped_units,reason=prepare_bounded_verify_pages(
        hydrated,
        max_context_chars=20000,
    )
    if page_scoped_units is None:
        rejections.append({"job_id":source.id,"reason":reason})
        continue
    if len(page_scoped_units) < 2:
        rejections.append({
            "job_id":source.id,
            "reason":"GATE_REQUIRES_REAL_MULTI_PAGE_VERIFY",
        })
        continue
    selected=(source,hydrated,page_scoped_units)
    break

if selected is None:
    raise SystemExit(
        "NO_COMPLETED_MULTI_PAGE_VERIFY_FITS_PAGE_SCOPED_CLASS:"
        + json.dumps(rejections,sort_keys=True)
    )

source,hydrated,page_scoped_units=selected
page_numbers=[page for page,_job,_ids in page_scoped_units]
page_evidence_ids={
    str(page):ids for page,_job,ids in page_scoped_units
}

class NoCodexFallback:
    def __init__(self):
        self.calls=0
    def execute(self,_job):
        self.calls+=1
        return {
            "verified":False,
            "item_id":"__NO_CODEX_FALLBACK__",
            "accepted_evidence_ids":[],
            "issues":["LOCAL_ROUTE_FELL_THROUGH"],
            "_execution":{
                "provider":"forbidden-fallback-sentinel",
                "model":None,
                "reasoning_effort":None,
                "usage":{
                    "input_tokens":0,
                    "cached_input_tokens":0,
                    "output_tokens":0,
                    "reasoning_output_tokens":0,
                },
            },
        }

fallback=NoCodexFallback()
service=local_inference_service_from_env(require_enabled=True)
try:
    pre=service.health()
    if pre["formula"]["spawn_count"] != 0 or pre["table"]["spawn_count"] != 0:
        raise SystemExit("DOCUMENT_MODEL_OWNER_SPAWNED_BEFORE_QWEN")
    if pre["qwen_verify"]["spawn_count"] != 0:
        raise SystemExit("QWEN_OWNER_SPAWNED_BEFORE_REQUEST")

    delegate=BoundedVerifySemanticDelegate(
        tier=ExecutorTier.STANDARD,
        fallback=fallback,
        local_inference=service,
        enabled=True,
        max_context_chars=20000,
    )
    result=delegate.execute(hydrated)
    post=service.health()
finally:
    service.close()

execution=dict(result.get("_execution") or {})
routing=dict(execution.get("routing") or {})
if fallback.calls:
    raise SystemExit(
        "LOCAL_PAGE_SCOPED_VERIFY_FELL_THROUGH_WITHOUT_CODEX_CALL:"
        + json.dumps(routing,sort_keys=True)
    )
if execution.get("provider") != "local-qwen-vllm":
    raise SystemExit(
        "VERIFY_NOT_SERVED_BY_LOCAL_QWEN:"
        + json.dumps(execution,sort_keys=True)
    )
if routing.get("decision") != "LOCAL_QWEN_ACCEPTED":
    raise SystemExit(
        "LOCAL_QWEN_RESULT_NOT_GUARD_ACCEPTED:"
        + json.dumps(routing,sort_keys=True)
    )
expected_calls=len(page_scoped_units)
if int(routing.get("page_count") or 0) != expected_calls:
    raise SystemExit("ROUTING_PAGE_COUNT_MISMATCH")
if int(routing.get("qwen_calls") or 0) != expected_calls:
    raise SystemExit("ROUTING_QWEN_CALL_COUNT_MISMATCH")
if post["qwen_verify"]["spawn_count"] != 1:
    raise SystemExit("QWEN_OWNER_SPAWN_COUNT_NOT_ONE")
if post["qwen_verify"]["requests_completed"] != expected_calls:
    raise SystemExit(
        f"QWEN_REQUEST_COUNT_MISMATCH:"
        f"expected={expected_calls}:"
        f"actual={post['qwen_verify']['requests_completed']}"
    )
if post["formula"]["spawn_count"] != 0 or post["table"]["spawn_count"] != 0:
    raise SystemExit("DOCUMENT_MODEL_OWNER_SPAWNED_DURING_VERIFY")

agreement=verify_agreement(source.result,result)
strict_agreement=bool(
    agreement["verified_equal"]
    and agreement["accepted_evidence_ids_equal"]
    and agreement["issues_equal"]
)
if not strict_agreement:
    raise SystemExit(
        "PAGE_SCOPED_QWEN_VERIFY_DISAGREES_WITH_EXISTING_CODEX_RESULT:"
        + json.dumps(
            {
                "agreement":agreement,
                "qwen_result":{
                    k:v for k,v in result.items() if k != "_execution"
                },
                "historical_result":{
                    k:v for k,v in (source.result or {}).items()
                    if k != "_execution"
                },
            },
            sort_keys=True,
        )
    )

context=dict(hydrated.payload.get("_research_context") or {})
payload={
  "schema_version":2,
  "decision":"PASS",
  "route":"QWEN35_PAGE_SCOPED_VERIFY_PRIMARY",
  "source_run_id":run_id,
  "source_job_id":source.id,
  "source_context_page_numbers":[
      int(x["page_number"]) for x in context.get("pages") or []
  ],
  "page_scoped_unit_count":len(page_scoped_units),
  "page_scoped_unit_numbers":page_numbers,
  "page_scoped_evidence_ids":page_evidence_ids,
  "authoritative_evidence_count":len(
      context.get("authoritative_evidence") or []
  ),
  "qwen_calls":expected_calls,
  "qwen_model_boots":post["qwen_verify"]["spawn_count"],
  "qwen_result":{k:v for k,v in result.items() if k != "_execution"},
  "qwen_execution":execution,
  "historical_codex_agreement":agreement,
  "strict_agreement_pass":strict_agreement,
  "codex_calls_performed":0,
  "scientific_state_materialized":False,
  "production_routing_changed":False,
  "broker_health_before":pre,
  "broker_health_after":post,
}
Path(out).write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n",
    encoding="utf-8",
)
print(json.dumps(payload,indent=2,sort_keys=True))
PY
require_stopped

echo
echo "=== FINAL PROCESS / SAFETY STATE ==="
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$gpu_uuid"; then
  fail "QWEN_VERIFY_PROCESS_LEAK_ON_GPU" 50
fi
require_stopped
echo "QWEN_BOUNDED_VERIFY_PRODUCTION_GATE=PASS"
echo "CODEX_CALLS=0"
echo "SCIENTIFIC_STATE_MATERIALIZED=false"
echo "PRODUCTION_ENV_MUTATED=false"
echo "DAEMON=inactive/dead PID=0"
echo "REPORT=$report"
