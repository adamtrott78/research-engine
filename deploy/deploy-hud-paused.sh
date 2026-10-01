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

echo "=== PAUSED HUD DEPLOYMENT ==="
echo "RUN=$run_id"

[ -x "$venv/bin/python" ] || fail "VENV_NOT_FOUND" 20
[ -f "$envf" ] || fail "ENV_FILE_NOT_FOUND" 21

active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
if [ "$active" != "inactive" ] || [ "$sub" != "dead" ] || [ "${pid:-0}" != "0" ]; then
    fail "DAEMON_MUST_BE_STOPPED_BEFORE_HUD_DEPLOY" 22
fi
echo "DAEMON_PRE=inactive/dead pid=0"

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
git fetch --no-tags origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)" ] || fail "CHECKOUT_NOT_AT_FETCHED_MAIN" 32
echo "DEPLOYED_HEAD=$(git rev-parse HEAD)"

if [ -n "${OPENAI_API_KEY:-}" ]; then
    fail "OPENAI_API_KEY_PRESENT_IN_CALLING_SHELL" 40
fi
if grep -Eq '^[[:space:]]*OPENAI_API_KEY[[:space:]]*=' "$envf"; then
    fail "OPENAI_API_KEY_PRESENT_IN_PRODUCTION_ENV" 41
fi

set -a
. "$envf"
set +a
export RESEARCH_ENGINE_DATABASE_URL="postgresql://research_engine:research_engine@localhost:5432/research_engine"
export RESEARCH_ENGINE_ARTIFACT_ROOT="$state/artifacts"
[ -z "${OPENAI_API_KEY:-}" ] || fail "OPENAI_API_KEY_VISIBLE_AFTER_ENV_LOAD" 42

module_path="$("$venv/bin/python" - <<'PY'
import pathlib
import research_engine
print(pathlib.Path(research_engine.__file__).resolve())
PY
)"
echo "MODULE_PATH_BEFORE=$module_path"

case "$module_path" in
    "$root"/*)
        echo "EDITABLE_SOURCE=HEALTHY"
        ;;
    *)
        echo "EDITABLE_SOURCE=REINSTALL_REQUIRED"
        "$venv/bin/pip" install --no-deps -e "$root"
        ;;
esac

module_path="$("$venv/bin/python" - <<'PY'
import pathlib
import research_engine
print(pathlib.Path(research_engine.__file__).resolve())
PY
)"
case "$module_path" in
    "$root"/*) ;;
    *) fail "VENV_NOT_USING_REPOSITORY_SOURCE" 43 ;;
esac
echo "MODULE_PATH_FINAL=$module_path"

pre="$scratch/pre-observability-state.json"
"$venv/bin/python" - "$run_id" "$pre" <<'PY'
import json
import os
import sys
from pathlib import Path
import psycopg

run_id, out = sys.argv[1], Path(sys.argv[2])
with psycopg.connect(os.environ['RESEARCH_ENGINE_DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT status,current_phase,human_gate_state,knowledge_checkpoint_id
               FROM research_runs WHERE id=%s""",
            (run_id,),
        )
        run = cur.fetchone()
        if not run:
            raise SystemExit('run not found')
        cur.execute(
            """SELECT job_type,status,count(*)::int
               FROM research_jobs WHERE run_id=%s
               GROUP BY job_type,status ORDER BY job_type,status""",
            (run_id,),
        )
        jobs = [list(x) for x in cur.fetchall()]
        cur.execute(
            """SELECT count(*)::int,COALESCE(sum(input_tokens),0)::bigint,
                      COALESCE(sum(output_tokens),0)::bigint
               FROM research_usage_events WHERE run_id=%s""",
            (run_id,),
        )
        usage = list(cur.fetchone())
        counts = {}
        for table in ('evidence_objects','knowledge_checkpoints','gap_candidates','novelty_challenges'):
            cur.execute(f'SELECT count(*)::int FROM {table}')
            counts[table] = cur.fetchone()[0]
state = {'run': list(run), 'jobs': jobs, 'usage_core': usage, 'scientific_counts': counts}
out.write_text(json.dumps(state, indent=2, default=str) + '\n')
print('PRE_STATE=' + json.dumps(state, sort_keys=True, default=str))
PY

echo "--- APPLY OBSERVABILITY SCHEMA ---"
"$venv/bin/python" - "$root/sql/schema.sql" <<'PY'
import os
import sys
from research_engine.postgres import PostgresStore
PostgresStore(os.environ['RESEARCH_ENGINE_DATABASE_URL']).initialize(sys.argv[1])
print('SCHEMA_INITIALIZATION=PASS')
PY

"$venv/bin/python" - "$run_id" "$pre" <<'PY'
import json
import os
import sys
from pathlib import Path
import psycopg

run_id = sys.argv[1]
before = json.loads(Path(sys.argv[2]).read_text())
with psycopg.connect(os.environ['RESEARCH_ENGINE_DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT status,current_phase,human_gate_state,knowledge_checkpoint_id
               FROM research_runs WHERE id=%s""",
            (run_id,),
        )
        run = cur.fetchone()
        cur.execute(
            """SELECT job_type,status,count(*)::int
               FROM research_jobs WHERE run_id=%s
               GROUP BY job_type,status ORDER BY job_type,status""",
            (run_id,),
        )
        jobs = [list(x) for x in cur.fetchall()]
        cur.execute(
            """SELECT count(*)::int,COALESCE(sum(input_tokens),0)::bigint,
                      COALESCE(sum(output_tokens),0)::bigint
               FROM research_usage_events WHERE run_id=%s""",
            (run_id,),
        )
        usage = list(cur.fetchone())
        counts = {}
        for table in ('evidence_objects','knowledge_checkpoints','gap_candidates','novelty_challenges'):
            cur.execute(f'SELECT count(*)::int FROM {table}')
            counts[table] = cur.fetchone()[0]
        cur.execute("SELECT to_regclass('public.research_allowance_observations')")
        allowance = cur.fetchone()[0]
        cur.execute("SELECT count(*)::int FROM research_usage_events WHERE logical_agent_id IS NOT NULL")
        attributed_usage = cur.fetchone()[0]
        cur.execute("SELECT count(*)::int FROM research_workers WHERE logical_agent_id IS NOT NULL")
        attributed_workers = cur.fetchone()[0]
after = {'run': list(run), 'jobs': jobs, 'usage_core': usage, 'scientific_counts': counts}
if before != json.loads(json.dumps(after, default=str)):
    raise SystemExit('protected research state changed during observability migration')
if allowance != 'research_allowance_observations':
    raise SystemExit('allowance observation table missing')
print('MIGRATION_AUDIT=' + json.dumps({
    'protected_state_preserved': True,
    'historically_attributed_usage_rows': attributed_usage,
    'historically_attributed_worker_rows': attributed_workers,
}, sort_keys=True))
PY

bash "$root/deploy/install-operator-shortcuts.sh"

observe_window() {
    local window="$1"
    local label="$2"
    local db_window="$3"
    local existing
    existing="$("$venv/bin/python" - "$db_window" <<'PY'
import os
import sys
import psycopg
with psycopg.connect(os.environ['RESEARCH_ENGINE_DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        cur.execute(
            'SELECT count(*)::int FROM research_allowance_observations WHERE window_type=%s',
            (sys.argv[1],),
        )
        print(cur.fetchone()[0])
PY
)"
    if [ "$existing" -gt 0 ]; then
        echo "$label observation already exists; skipping initial prompt."
        return
    fi
    local used reset
    read -r -p "Current $label USED percent [blank=skip]: " used
    used="${used%\%}"
    [ -n "$used" ] || return
    read -r -p "$label reset ISO-8601 with offset [blank=unknown]: " reset
    local args=(allowance-observe --window "$window" --used-percent "$used" --note "Initial live HUD observation")
    if [ -n "$reset" ]; then
        args+=(--resets-at "$reset")
    fi
    "$venv/bin/research-engine" "${args[@]}"
}

echo "--- CURRENT CODEX ALLOWANCE ---"
observe_window "5h" "5-hour" "FIVE_HOUR"
observe_window "weekly" "WEEKLY" "WEEKLY"

snapshot="$scratch/research-ops-snapshot.json"
"$venv/bin/research-engine" ops-snapshot --run-id "$run_id" > "$snapshot"
echo "OPS_SNAPSHOT_FILE=$snapshot"

echo "--- ONE-SHOT HUD ---"
"$HOME/.local/bin/HUD" --once --no-color

active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
if [ "$active" != "inactive" ] || [ "$sub" != "dead" ] || [ "${pid:-0}" != "0" ]; then
    fail "OBSERVABILITY_DEPLOY_STARTED_DAEMON" 80
fi

echo "DAEMON_FINAL=inactive/dead pid=0"
echo "SHORTCUTS=START STOP HUD STATUS"
echo "PAUSED_HUD_DEPLOYMENT=PASS"
