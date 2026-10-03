#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
unit="research-engine-daemon.service"
core_venv="$state/venv"
resume_root="${1:-}"

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

[ -n "$resume_root" ] || fail "RESUME_ROOT_REQUIRED" 30
resume_root="$(readlink -f "$resume_root")"
case "$resume_root" in
    "$adam/workspace-control-tmp/"*) ;;
    *) fail "RESUME_ROOT_MUST_BE_UNDER_VISIBLE_WORKSPACE_CONTROL_TMP" 31 ;;
esac

plan="$resume_root/ab-plan.json"
packages_native="$resume_root/packages/native"
packages_standard="$resume_root/packages/standard"
preflight="$resume_root/zero-model-preflight.json"
replay="$resume_root/representative-semantic-replay.json"

echo "=== RESUME EXTRACT / VERIFY PAPER-PACKAGE A/B ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 32
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 33
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 34
[ -f "$plan" ] || fail "AB_PLAN_NOT_FOUND" 35
[ -d "$packages_native" ] || fail "NATIVE_PACKAGE_ROOT_NOT_FOUND" 36
[ -d "$packages_standard" ] || fail "STANDARD_PACKAGE_ROOT_NOT_FOUND" 37

native_count="$(find "$packages_native" -type f -path '*/paper-package-v1/package.json' | wc -l)"
standard_count="$(find "$packages_standard" -type f -path '*/paper-package-v1/package.json' | wc -l)"
[ "$native_count" -eq 9 ] || fail "EXPECTED_NINE_NATIVE_PACKAGES" 38
[ "$standard_count" -eq 9 ] || fail "EXPECTED_NINE_STANDARD_PACKAGES" 39

export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"
export RESEARCH_ENGINE_ARTIFACT_ROOT="${RESEARCH_ENGINE_ARTIFACT_ROOT:-$state/artifacts}"
export RESEARCH_ENGINE_SEMANTIC_BACKEND=codex

echo "HEAD=$(git rev-parse HEAD)"
echo "RESUME_ROOT=$resume_root"
echo "NATIVE_PACKAGES=$native_count"
echo "STANDARD_PACKAGES=$standard_count"

echo
echo "=== VERIFY POSTGRESQL ONLY ==="
sudo docker compose -f docker-compose.yml exec -T db     pg_isready -U research_engine -d research_engine
require_stopped

run_id="$("$core_venv/bin/python" - "$plan" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["source_run_id"])
PY
)"
[ -n "$run_id" ] || fail "SOURCE_RUN_ID_MISSING" 40
echo "SOURCE_RUN_ID=$run_id"

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

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "SOURCE_RUN_ID=$run_id"
echo "RESUME_ROOT=$resume_root"
echo "ZERO_MODEL_PREFLIGHT=$preflight"
echo "SEMANTIC_REPLAY=$replay"
echo "PAPER_PACKAGE_AB_RESUME_GATE=MEASURED"
