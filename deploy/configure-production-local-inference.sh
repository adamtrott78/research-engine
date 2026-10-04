#!/usr/bin/env bash
set -euo pipefail

adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
envf="$state/research-engine.env"
unit="research-engine-daemon.service"
backup_root="$state/backups"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup="$backup_root/research-engine.env.$stamp"

fail() {
    echo "ERROR=$1" >&2
    exit "${2:-1}"
}

active="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
sub="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null || true)"
pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)"
if [ "$active" != "inactive" ] || [ "$sub" != "dead" ] || [ "${pid:-0}" != "0" ]; then
    fail "RESEARCH_ENGINE_DAEMON_MUST_BE_STOPPED_BEFORE_ENV_UPDATE" 20
fi

[ -f "$envf" ] || fail "PRODUCTION_ENV_NOT_FOUND" 21
mkdir -p "$backup_root"
cp -a "$envf" "$backup"

python3 - "$envf" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
begin = "# BEGIN MANAGED LOCAL DOCUMENT INFERENCE"
end = "# END MANAGED LOCAL DOCUMENT INFERENCE"
block = """# BEGIN MANAGED LOCAL DOCUMENT INFERENCE
RESEARCH_ENGINE_VERIFY_PAGE_RADIUS=0
RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1
RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING=8
RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS=900
RESEARCH_ENGINE_FORMULA_GPU_INDEX=0
RESEARCH_ENGINE_TABLE_GPU_INDEX=1
RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED=1
RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED=1
RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX=0
RESEARCH_ENGINE_DOCUMENT_SIDECAR_TIMEOUT_SECONDS=900
# END MANAGED LOCAL DOCUMENT INFERENCE
"""

if begin in text or end in text:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit("managed local inference markers are malformed")
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    text = text[:start].rstrip() + "\n\n" + text[stop:].lstrip("\n")

keys = {
    "RESEARCH_ENGINE_VERIFY_PAGE_RADIUS",
    "RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED",
    "RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING",
    "RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS",
    "RESEARCH_ENGINE_FORMULA_GPU_INDEX",
    "RESEARCH_ENGINE_TABLE_GPU_INDEX",
    "RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED",
    "RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED",
    "RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX",
    "RESEARCH_ENGINE_DOCUMENT_SIDECAR_TIMEOUT_SECONDS",
}
lines = []
for line in text.splitlines():
    stripped = line.strip()
    if any(stripped.startswith(key + "=") for key in keys):
        continue
    lines.append(line)

text = "\n".join(lines).rstrip() + "\n\n" + block
path.write_text(text, encoding="utf-8")
PY

set -a
. "$envf"
set +a

[ "${RESEARCH_ENGINE_VERIFY_PAGE_RADIUS:-}" = "0" ] || fail "VERIFY_PAGE_RADIUS_WRITE_FAILED" 29
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED:-}" = "1" ] || fail "LOCAL_INFERENCE_ENABLE_WRITE_FAILED" 30
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING:-}" = "8" ] || fail "LOCAL_INFERENCE_QUEUE_WRITE_FAILED" 31
[ "${RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS:-}" = "900" ] || fail "LOCAL_INFERENCE_TIMEOUT_WRITE_FAILED" 32
[ "${RESEARCH_ENGINE_FORMULA_GPU_INDEX:-}" = "0" ] || fail "FORMULA_GPU_WRITE_FAILED" 33
[ "${RESEARCH_ENGINE_TABLE_GPU_INDEX:-}" = "1" ] || fail "TABLE_GPU_WRITE_FAILED" 34
[ "${RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED:-}" = "1" ] || fail "DOCUMENT_SIDECAR_ENABLE_WRITE_FAILED" 35
[ "${RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED:-}" = "1" ] || fail "DOCUMENT_SIDECAR_SPECIALIZED_WRITE_FAILED" 36
[ "${RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX:-}" = "0" ] || fail "DOCUMENT_STRUCTURE_GPU_WRITE_FAILED" 37
[ "${RESEARCH_ENGINE_DOCUMENT_SIDECAR_TIMEOUT_SECONDS:-}" = "900" ] || fail "DOCUMENT_SIDECAR_TIMEOUT_WRITE_FAILED" 38

echo "PRODUCTION_LOCAL_INFERENCE_ENV=CONFIGURED"
echo "ENV_FILE=$envf"
echo "BACKUP=$backup"
echo "ROLLBACK=cp '$backup' '$envf'"
