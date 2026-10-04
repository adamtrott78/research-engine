#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
envf="$state/research-engine.env"
core="$state/venv/bin/research-engine"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$visible_tmp_root/bounded-production-integration-$stamp"

cleanup() {
  systemctl --user stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

fail() {
  echo "ERROR=$1" >&2
  exit "${2:-1}"
}

require_stopped() {
  [ "$(systemctl --user is-active "$unit" 2>/dev/null || true)" = "inactive" ] || fail "DAEMON_NOT_INACTIVE"
  [ "$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)" = "dead" ] || fail "DAEMON_NOT_DEAD"
  [ "$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)" = "0" ] || fail "DAEMON_PID_NOT_ZERO"
}

no_model_servers() {
  ! pgrep -f '[d]ocling_formula_server.py' >/dev/null 2>&1 || fail "FORMULA_MODEL_SERVER_LEAK"
  ! pgrep -f '[p]addle_table_server.py' >/dev/null 2>&1 || fail "TABLE_MODEL_SERVER_LEAK"
}

echo "=== BOUNDED PRODUCTION INTEGRATION ==="
systemctl --user stop "$unit" || true
require_stopped
no_model_servers

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH"
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY"
mkdir -p "$out"

set -a
. "$envf"
set +a
export RESEARCH_ENGINE_DATABASE_URL="postgresql://research_engine:research_engine@localhost:5432/research_engine"
export RESEARCH_ENGINE_ARTIFACT_ROOT="$state/artifacts"

[ "${RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED:-}" = "1" ] || fail "DOCUMENT_SIDECAR_NOT_ENABLED"
[ "${RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED:-}" = "1" ] || fail "DOCUMENT_SIDECAR_SPECIALIZED_NOT_ENABLED"
[ "${RESEARCH_ENGINE_VERIFY_PAGE_RADIUS:-}" = "0" ] || fail "VERIFY_PAGE_RADIUS_NOT_ZERO"

echo "HEAD=$(git rev-parse HEAD)"
echo "OUTPUT_ROOT=$out"

echo "=== DATABASE + SCHEMA ==="
sudo docker compose -f docker-compose.yml up -d db
sudo docker compose -f docker-compose.yml exec -T db pg_isready -U research_engine -d research_engine
sudo docker compose -f docker-compose.yml exec -T db psql -v ON_ERROR_STOP=1 -U research_engine -d research_engine < sql/schema.sql > "$out/schema-apply.log"

echo "=== QUIESCENCE PREFLIGHT ==="
"$core" bounded-integration-preflight > "$out/preflight.json"
cat "$out/preflight.json"

echo "=== SCIENTIFIC FINGERPRINT BEFORE ==="
"$core" bounded-integration-fingerprint > "$out/scientific-fingerprint-before.json"

echo "=== DERIVED SIDECAR MISS -> HIT ==="
"$core" document-sidecar-smoke > "$out/document-sidecar-smoke.json"
"$core" bounded-integration-validate-sidecar "$out/document-sidecar-smoke.json" > "$out/sidecar-validation.json"
cat "$out/sidecar-validation.json"
no_model_servers
require_stopped

echo "=== SIDECAR TELEMETRY ==="
"$core" bounded-integration-sidecar-telemetry --smoke-json "$out/document-sidecar-smoke.json" > "$out/sidecar-telemetry.json"
cat "$out/sidecar-telemetry.json"

echo "=== IDLE DAEMON LIFECYCLE ==="
systemctl --user start "$unit"
for _ in $(seq 1 20); do
  [ "$(systemctl --user is-active "$unit" 2>/dev/null || true)" = "active" ] && break
  sleep 1
done
[ "$(systemctl --user is-active "$unit" 2>/dev/null || true)" = "active" ] || fail "DAEMON_DID_NOT_START"
pid="$(systemctl --user show "$unit" -p MainPID --value)"
[ "${pid:-0}" != "0" ] || fail "DAEMON_MAINPID_MISSING"
echo "DAEMON_MAINPID=$pid"
sleep 6
no_model_servers
"$core" bounded-integration-workers "$pid" > "$out/daemon-workers.json"
cat "$out/daemon-workers.json"

systemctl --user stop "$unit"
for _ in $(seq 1 20); do
  active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
  sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
  mpid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
  [ "$active" = "inactive" ] && [ "$sub" = "dead" ] && [ "${mpid:-0}" = "0" ] && break
  sleep 1
done
require_stopped
no_model_servers

echo "=== SCIENTIFIC FINGERPRINT AFTER ==="
"$core" bounded-integration-fingerprint > "$out/scientific-fingerprint-after.json"
"$core" bounded-integration-compare-fingerprints "$out/scientific-fingerprint-before.json" "$out/scientific-fingerprint-after.json" > "$out/fingerprint-comparison.json"
cat "$out/fingerprint-comparison.json"

echo "=== BUILD BOUNDED INTEGRATION REPORT ==="
"$core" bounded-integration-build-report   "$out/preflight.json"   "$out/sidecar-validation.json"   "$out/daemon-workers.json"   "$out/fingerprint-comparison.json"   "$out/sidecar-telemetry.json"   --head "$(git rev-parse HEAD)"   > "$out/bounded-production-integration-report.json"
cat "$out/bounded-production-integration-report.json"

echo "=== FINAL SAFETY STATE ==="
require_stopped
no_model_servers
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "SCIENTIFIC_FINGERPRINT=UNCHANGED"
echo "OUTPUT_ROOT=$out"
echo "REPORT=$out/bounded-production-integration-report.json"
echo "BOUNDED_PRODUCTION_INTEGRATION=PASS"
