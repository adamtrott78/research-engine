#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="$HOME/adamArchives/Adam/.research-engine"
scratch="$HOME/adamArchives/Adam/.workspace-control/tmp"
venv="$state/venv"
envf="$state/research-engine.env"
run_id="${RESEARCH_HUD_RUN_ID:-rf-open-world-corpus-expansion-v1}"
unit="research-engine-daemon.service"

mkdir -p "$scratch"

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
        fail "DAEMON_MUST_REMAIN_STOPPED_FOR_VERIFY_CALIBRATION" 20
    fi
}

observe() {
    local window="$1"
    local label="$2"
    local note="$3"
    local left reset_in
    read -r -p "$label LEFT percent [blank=skip]: " left
    left="${left%\%}"
    [ -n "$left" ] || return 0
    read -r -p "$label resets in [e.g. 2h 30m or 5d 7h; blank=unknown]: " reset_in

    local args=(
        allowance-observe
        --window "$window"
        --left-percent "$left"
        --note "$note"
    )
    if [ -n "$reset_in" ]; then
        args+=(--resets-in "$reset_in")
    fi
    "$venv/bin/research-engine" "${args[@]}"
}

echo "=== VERIFY CONTEXT LOW/HIGH CALIBRATION ==="
echo "SOURCE_RUN=$run_id"

require_stopped

[ -f "$envf" ] || fail "ENV_FILE_NOT_FOUND" 21
[ -x "$venv/bin/research-engine" ] || fail "RESEARCH_ENGINE_COMMAND_NOT_FOUND" 22

if [ -n "${OPENAI_API_KEY:-}" ]; then
    fail "OPENAI_API_KEY_PRESENT_IN_CALLING_SHELL" 23
fi
if grep -Eq '^[[:space:]]*OPENAI_API_KEY[[:space:]]*=' "$envf"; then
    fail "OPENAI_API_KEY_PRESENT_IN_PRODUCTION_ENV" 24
fi

set -a
. "$envf"
set +a

export RESEARCH_ENGINE_DATABASE_URL="postgresql://research_engine:research_engine@localhost:5432/research_engine"
export RESEARCH_ENGINE_ARTIFACT_ROOT="$state/artifacts"

[ "${RESEARCH_ENGINE_SEMANTIC_BACKEND:-codex}" = "codex" ] ||
    fail "CALIBRATION_REQUIRES_CODEX_BACKEND" 25
[ -z "${OPENAI_API_KEY:-}" ] ||
    fail "OPENAI_API_KEY_VISIBLE_AFTER_ENV_LOAD" 26

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31

echo "CODE_SHA=$(git rev-parse HEAD)"
echo "VERIFY_PAGE_RADIUS=${RESEARCH_ENGINE_VERIFY_PAGE_RADIUS:-0}"

echo
echo "--- APPLY ADDITIVE CALIBRATION SCHEMA ---"
"$venv/bin/python" - "$root/sql/schema.sql" <<'PY'
import os
import sys
from research_engine.postgres import PostgresStore

PostgresStore(
    os.environ["RESEARCH_ENGINE_DATABASE_URL"]
).initialize(sys.argv[1])
print("CALIBRATION_SCHEMA=PASS")
PY

echo
echo "--- ZERO-MODEL EXTRACT PROVENANCE AUDIT ---"
audit_file="$scratch/extract-provenance-audit.json"
"$venv/bin/research-engine" audit-extract-provenance "$run_id" > "$audit_file"
"$venv/bin/python" - "$audit_file" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print("AUDIT_PAPERS=" + str(data["paper_count"]))
print("AUDIT_TOTAL_PDF_PAGES=" + str(data["total_pdf_pages"]))
print("AUDIT_EVIDENCE_OBJECTS=" + str(data["total_evidence_objects"]))
print("AUDIT_EVIDENCE_BEARING_PAGES=" + str(data["total_evidence_bearing_pages"]))
print("AUDIT_EXTRACTION_CONTEXT_CHARS=" + str(data["total_extraction_context_chars"]))
print("EXTRACT_AUDIT_JSON=" + json.dumps(data, sort_keys=True))
PY

require_stopped

echo
echo "--- PRE-CALIBRATION PLAN READING ---"
echo "Enter one fresh usage-UI reading pair immediately before the two Terra replays."
observe "5h" "5-hour" "pre low/high VERIFY replay batch"
observe "weekly" "WEEKLY" "pre low/high VERIFY replay batch"

require_stopped

print_replay() {
    local result_file="$1"
    "$venv/bin/python" - "$result_file" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
historical = data["historical"]
replay = data["replay"]
agreement = data["agreement"]

print("SELECTOR=" + data["selector"])
print("CALIBRATION_EVENT_ID=" + str(data["calibration_event_id"]))
print("SOURCE_JOB_ID=" + data["source_job_id"])
print("CONTEXT_PAGES=" + str(data["context_page_count"]))
print("CONTEXT_PAGE_NUMBERS=" + ",".join(map(str, data["context_page_numbers"])))
print("CONTEXT_CHARS=" + str(data["context_char_count"]))
print("EVIDENCE_COUNT=" + str(data["evidence_count"]))
print("HISTORICAL_INPUT=" + str(historical["input_tokens"]))
print("HISTORICAL_CACHED=" + str(historical["cached_input_tokens"]))
print("HISTORICAL_OUTPUT=" + str(historical["output_tokens"]))
print("HISTORICAL_REASONING_OUTPUT=" + str(historical["reasoning_output_tokens"]))
print("REPLAY_INPUT=" + str(replay["input_tokens"]))
print("REPLAY_CACHED=" + str(replay["cached_input_tokens"]))
print("REPLAY_OUTPUT=" + str(replay["output_tokens"]))
print("REPLAY_REASONING_OUTPUT=" + str(replay["reasoning_output_tokens"]))
print("REPLAY_WALL_TIME_MS=" + str(replay["wall_time_ms"]))
reduction = data.get("input_reduction_fraction")
print(
    "INPUT_REDUCTION_PERCENT="
    + ("NA" if reduction is None else f"{100*reduction:.2f}")
)
print("VERIFIED_EQUAL=" + str(bool(agreement["verified_equal"])).lower())
print(
    "ACCEPTED_EVIDENCE_IDS_EQUAL="
    + str(bool(agreement["accepted_evidence_ids_equal"])).lower()
)
print("CALIBRATION_JSON=" + json.dumps(data, sort_keys=True))
PY
}

echo
echo "--- LOWEST-COST TERRA VERIFY REPLAY ---"
low_file="$scratch/verify-replay-lowest.json"
"$venv/bin/research-engine" calibrate-verify "$run_id" --selector lowest > "$low_file"
print_replay "$low_file"

require_stopped

echo
echo "--- HIGHEST-COST TERRA VERIFY REPLAY ---"
high_file="$scratch/verify-replay-highest.json"
"$venv/bin/research-engine" calibrate-verify "$run_id" --selector highest > "$high_file"
print_replay "$high_file"

require_stopped

echo
echo "--- POST-CALIBRATION PLAN READING ---"
echo "Refresh the usage UI once after both calls, then enter the displayed values."
observe "5h" "5-hour" "post low/high VERIFY replay batch"
observe "weekly" "WEEKLY" "post low/high VERIFY replay batch"

echo
echo "--- UPDATED HUD ---"
if command -v HUD >/dev/null 2>&1; then
    HUD --once --no-color
else
    "$venv/bin/research-engine" hud --run-id "$run_id" --once --no-color
fi

require_stopped

echo
echo "DAEMON_FINAL=inactive/dead pid=0"
echo "VERIFY_LOW_HIGH_REPLAY_CALIBRATION=PASS"
