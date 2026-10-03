#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"
unit="research-engine-daemon.service"
core_venv="$state/venv"
run_id="${1:-rf-open-world-corpus-expansion-v1}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="$visible_tmp_root/paper-package-ab-posthoc-$stamp"
report="$output_root/posthoc-contemporaneous-report.json"

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

echo "=== PAPER-PACKAGE A/B POSTHOC CONTEMPORANEOUS COMPARISON ==="
systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/research-engine" ] || fail "CORE_VENV_NOT_FOUND" 32

mkdir -p "$visible_tmp_root" "$output_root"

export RESEARCH_ENGINE_DATABASE_URL="${RESEARCH_ENGINE_DATABASE_URL:-postgresql://research_engine:research_engine@localhost:5432/research_engine}"

echo "HEAD=$(git rev-parse HEAD)"
echo "SOURCE_RUN_ID=$run_id"
echo "OUTPUT_ROOT=$output_root"

echo
echo "=== VERIFY POSTGRESQL ONLY ==="
sudo docker compose -f docker-compose.yml exec -T db \
    pg_isready -U research_engine -d research_engine
require_stopped

echo
echo "=== COMPARE ALREADY-RECORDED PACKAGE VS PDF CONTROL OUTPUTS ==="
"$core_venv/bin/research-engine" paper-package-ab-posthoc "$run_id" > "$report"
require_stopped

echo
echo "=== COMPACT CONTEMPORANEOUS POSTHOC SUMMARY ==="
"$core_venv/bin/python" - "$report" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
rows=[]
for row in d["rows"]:
    rows.append({
        "source_job_id": row["source_job_id"],
        "job_type": row["job_type"],
        "controlled_pair": row["controlled_pair"],
        "package_vs_pdf_agreement_pass": row[
            "package_vs_pdf_agreement_pass"
        ],
        "package_input_tokens": row["package_input_tokens"],
        "pdf_control_input_tokens": row["pdf_control_input_tokens"],
        "controlled_input_reduction_fraction": row[
            "controlled_input_reduction_fraction"
        ],
    })
print(json.dumps({
    "decision": d["decision"],
    "package_events": d["package_events"],
    "controlled_pairs": d["controlled_pairs"],
    "controlled_pair_passes": d["controlled_pair_passes"],
    "controlled_pair_agreement_rate": d[
        "controlled_pair_agreement_rate"
    ],
    "controlled_pdf_input_tokens": d["controlled_pdf_input_tokens"],
    "controlled_package_input_tokens": d[
        "controlled_package_input_tokens"
    ],
    "controlled_input_reduction_fraction": d[
        "controlled_input_reduction_fraction"
    ],
    "controlled_pdf_cached_input_tokens": d[
        "controlled_pdf_cached_input_tokens"
    ],
    "controlled_package_cached_input_tokens": d[
        "controlled_package_cached_input_tokens"
    ],
    "controlled_cached_input_reduction_fraction": d[
        "controlled_cached_input_reduction_fraction"
    ],
    "rows": rows,
}, indent=2, sort_keys=True))
PY

decision="$("$core_venv/bin/python" - "$report" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"

echo "POSTHOC_DECISION=$decision"

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "SOURCE_RUN_ID=$run_id"
echo "REPORT=$report"
echo "PAPER_PACKAGE_AB_POSTHOC=MEASURED"
