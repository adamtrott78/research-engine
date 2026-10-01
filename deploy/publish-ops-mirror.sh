#!/usr/bin/env bash
set -eo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
state="$HOME/adamArchives/Adam/.research-engine"
scratch="$HOME/adamArchives/Adam/.workspace-control/tmp"
venv="$state/venv"
envf="$state/research-engine.env"

run_id="$RESEARCH_HUD_RUN_ID"
repo="$RESEARCH_ENGINE_GITHUB_REPO"
title="$RESEARCH_ENGINE_OPS_ISSUE_TITLE"
mode="$1"

[ -n "$run_id" ] || run_id="rf-open-world-corpus-expansion-v1"
[ -n "$repo" ] || repo="adamtrott78/research-engine"
[ -n "$title" ] || title="[ops] research-engine live state"
[ -n "$mode" ] || mode="current"

fail() {
    echo "ERROR=$1" >&2
    exit "$2"
}

case "$mode" in
    current|rollup) ;;
    *) fail "MODE_MUST_BE_CURRENT_OR_ROLLUP" 10 ;;
esac

[ -f "$envf" ] || fail "ENV_FILE_NOT_FOUND" 20
[ -x "$venv/bin/research-engine" ] || fail "RESEARCH_ENGINE_COMMAND_NOT_FOUND" 21
command -v gh >/dev/null 2>&1 || fail "GH_CLI_NOT_FOUND" 22

gh auth status --hostname github.com >/dev/null 2>&1 ||
    fail "GH_AUTH_NOT_READY" 23

set -a
. "$envf"
set +a

if [ -z "$RESEARCH_ENGINE_DATABASE_URL" ]; then
    export RESEARCH_ENGINE_DATABASE_URL="postgresql://research_engine:research_engine@localhost:5432/research_engine"
fi

mkdir -p "$scratch"
body="$scratch/research-engine-ops-mirror.md"

cd "$root"
"$venv/bin/research-engine" ops-mirror --run-id "$run_id" --format markdown > "$body"

size="$(wc -c < "$body" | tr -d ' ')"
[ "$size" -le 60000 ] || fail "OPS_MIRROR_BODY_TOO_LARGE" 24

issues_json="$(
    gh issue list \
        --repo "$repo" \
        --state open \
        --limit 100 \
        --json number,title
)"

number="$(
    "$venv/bin/python" -c '
import json, sys
title = sys.argv[1]
rows = json.loads(sys.argv[2])
matches = [str(row["number"]) for row in rows if row.get("title") == title]
print(matches[0] if matches else "")
' "$title" "$issues_json"
)"

created=0
if [ -z "$number" ]; then
    url="$(
        gh issue create \
            --repo "$repo" \
            --title "$title" \
            --body-file "$body"
    )"
    number="$(basename "$url")"
    created=1
    echo "OPS_MIRROR_CREATED=$url"
fi

if [ "$mode" = "current" ]; then
    if [ "$created" -eq 0 ]; then
        gh issue edit "$number" \
            --repo "$repo" \
            --body-file "$body" >/dev/null
    fi
    echo "OPS_MIRROR_CURRENT_ISSUE=$number"
else
    if [ "$created" -eq 0 ]; then
        gh issue comment "$number" \
            --repo "$repo" \
            --body-file "$body" >/dev/null
        echo "OPS_MIRROR_ROLLUP_COMMENTED=$number"
    else
        echo "OPS_MIRROR_ROLLUP_CREATED_CURRENT_ISSUE=$number"
    fi
fi

echo "OPS_MIRROR_MODE=$mode"
echo "OPS_MIRROR_RUN=$run_id"
echo "OPS_MIRROR_BODY=$body"
