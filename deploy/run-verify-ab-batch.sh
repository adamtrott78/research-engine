#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state="$HOME/adamArchives/Adam/.research-engine"
scratch="$HOME/adamArchives/Adam/.workspace-control/tmp"
venv="$state/venv"
envf="$state/research-engine.env"
run_id="${RESEARCH_HUD_RUN_ID:-rf-open-world-corpus-expansion-v1}"
unit="research-engine-daemon.service"
source_job_id="${1:-}"
repeats="${2:-${RESEARCH_ENGINE_VERIFY_AB_REPEATS:-3}}"

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
        fail "DAEMON_MUST_REMAIN_STOPPED_FOR_VERIFY_AB_BATCH" 20
    fi
}

observe_paste() {
    local note="$1"
    echo "Refresh usage, then paste the complete 5-hour + weekly usage-limit block once."
    "$venv/bin/research-engine" allowance-observe-paste --note "$note"
}

summarize_phase() {
    local radius="$1"
    local jsonl="$2"
    "$venv/bin/python" - "$radius" "$jsonl" <<'PY'
import json
import statistics
import sys

radius = int(sys.argv[1])
path = sys.argv[2]
rows = [
    json.loads(line)
    for line in open(path, encoding="utf-8")
    if line.strip()
]
if not rows:
    raise SystemExit("no calibration rows")

inputs = [int(r["replay"]["input_tokens"]) for r in rows]
cached = [int(r["replay"]["cached_input_tokens"]) for r in rows]
outputs = [int(r["replay"]["output_tokens"]) for r in rows]
reasoning = [int(r["replay"]["reasoning_output_tokens"]) for r in rows]
wall = [int(r["replay"]["wall_time_ms"]) for r in rows]
verified = [bool(r["agreement"]["verified_equal"]) for r in rows]
accepted = [bool(r["agreement"]["accepted_evidence_ids_equal"]) for r in rows]

summary = {
    "verify_page_radius": radius,
    "calls": len(rows),
    "source_job_id": rows[0]["source_job_id"],
    "context_page_count": rows[0]["context_page_count"],
    "context_page_numbers": rows[0]["context_page_numbers"],
    "context_char_count": rows[0]["context_char_count"],
    "evidence_count": rows[0]["evidence_count"],
    "historical_input_tokens": rows[0]["historical"]["input_tokens"],
    "historical_cached_input_tokens": rows[0]["historical"]["cached_input_tokens"],
    "verified_equal_calls": sum(verified),
    "accepted_evidence_ids_equal_calls": sum(accepted),
    "all_verified_equal": all(verified),
    "all_accepted_evidence_ids_equal": all(accepted),
    "replay_input_tokens": {
        "total": sum(inputs),
        "mean": statistics.mean(inputs),
        "median": statistics.median(inputs),
        "min": min(inputs),
        "max": max(inputs),
    },
    "replay_cached_input_tokens": {
        "total": sum(cached),
        "mean": statistics.mean(cached),
    },
    "replay_output_tokens": {
        "total": sum(outputs),
        "mean": statistics.mean(outputs),
    },
    "replay_reasoning_output_tokens": {
        "total": sum(reasoning),
        "mean": statistics.mean(reasoning),
    },
    "wall_time_ms": {
        "mean": statistics.mean(wall),
        "median": statistics.median(wall),
    },
}
print("VERIFY_AB_PHASE_SUMMARY=" + json.dumps(summary, sort_keys=True))
PY
}

run_phase() {
    local radius="$1"
    local jsonl="$2"
    : > "$jsonl"

    echo
    echo "--- RADIUS $radius PHASE: $repeats REPLAYS ---"
    local i result
    for ((i=1; i<=repeats; i++)); do
        require_stopped
        echo "RADIUS_${radius}_CALL=$i/$repeats"
        result="$(
            RESEARCH_ENGINE_VERIFY_PAGE_RADIUS="$radius"             "$venv/bin/research-engine" calibrate-verify "$run_id"                 --source-job-id "$source_job_id"
        )"
        printf '%s\n' "$result" >> "$jsonl"
        "$venv/bin/python" - "$result" <<'PY'
import json
import sys

d = json.loads(sys.argv[1])
a = d["agreement"]
r = d["replay"]
print(
    "CALL_RESULT "
    f"radius={d['verify_page_radius']} "
    f"input={r['input_tokens']} "
    f"cached={r['cached_input_tokens']} "
    f"output={r['output_tokens']} "
    f"reasoning={r['reasoning_output_tokens']} "
    f"verified_equal={str(bool(a['verified_equal'])).lower()} "
    f"accepted_ids_equal={str(bool(a['accepted_evidence_ids_equal'])).lower()}"
)
PY
    done
    summarize_phase "$radius" "$jsonl"
}

echo "=== VERIFY RADIUS 0/1 REPEATED A/B CALIBRATION ==="
echo "SOURCE_RUN=$run_id"

[ -n "$source_job_id" ] || fail "SOURCE_JOB_ID_REQUIRED" 10
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS_MUST_BE_POSITIVE_INTEGER" 11
[ "$repeats" -le 12 ] || fail "REPEATS_TOO_LARGE_MAX_12" 12

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
echo "REPEATS_PER_RADIUS=$repeats"
echo "TOTAL_SEMANTIC_CALLS=$((2 * repeats))"
echo "SOURCE_JOB_ID=$source_job_id"

radius0_file="$scratch/verify-ab-radius0.jsonl"
radius1_file="$scratch/verify-ab-radius1.jsonl"

echo
echo "--- PRE-BATCH ALLOWANCE READING ---"
observe_paste "pre repeated VERIFY radius0 phase"

run_phase 0 "$radius0_file"

echo
echo "--- MID-BATCH ALLOWANCE READING ---"
observe_paste "post repeated VERIFY radius0 / pre radius1 phase"

run_phase 1 "$radius1_file"

echo
echo "--- POST-BATCH ALLOWANCE READING ---"
observe_paste "post repeated VERIFY radius1 phase"

require_stopped

echo
echo "--- FINAL SUMMARY ---"
summarize_phase 0 "$radius0_file"
summarize_phase 1 "$radius1_file"
echo "RADIUS0_JSONL=$radius0_file"
echo "RADIUS1_JSONL=$radius1_file"
echo "DAEMON_FINAL=inactive/dead pid=0"
echo "VERIFY_AB_BATCH=PASS"
