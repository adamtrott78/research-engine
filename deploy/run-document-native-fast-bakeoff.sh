#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
scratch="$adam/.workspace-control/tmp"
runtimes="$adam/.document-runtimes"
core_venv="$state/venv"
unit="research-engine-daemon.service"

manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_root="$scratch/document-native-fast-$stamp"
docling_venv="$runtimes/docling-native"
mineru_venv="$runtimes/mineru-base"

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

find_padataset() {
    local candidate
    for candidate in \
        "$adam/varMax/PADataset" \
        "$adam/PADataset"
    do
        if [ -d "$candidate/papers/milcom2026/reference_notes/papers" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    candidate="$(
        find "$adam" -maxdepth 5 -type d \
            -path '*/PADataset/papers/milcom2026/reference_notes/papers' \
            -print -quit 2>/dev/null \
        | sed 's#/papers/milcom2026/reference_notes/papers$##'
    )"
    [ -n "$candidate" ] || return 1
    printf '%s\n' "$candidate"
}

report_size() {
    local label="$1"
    local path="$2"
    if [ -e "$path" ]; then
        echo "$label=$(du -sh "$path" | awk '{print $1}')"
    else
        echo "$label=ABSENT"
    fi
}

echo "=== DOCUMENT NATIVE FAST-PATH BAKEOFF ==="

systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 32
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 33

code_sha="$(git rev-parse HEAD)"
echo "CODE_SHA=$code_sha"

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

echo "PADATASET=$padataset"
echo "BUNDLE_ROOT=$bundle_root"
echo "OUTPUT_ROOT=$out_root"

mkdir -p "$scratch" "$runtimes" "$out_root"

echo
echo "=== CORE VENV REFRESH ==="
"$core_venv/bin/python" -m pip install -e "$root"

echo
echo "=== BEFORE RUNTIME INSTALL ==="
report_size "DOCLING_VENV_BEFORE" "$docling_venv"
report_size "MINERU_VENV_BEFORE" "$mineru_venv"
report_size "HF_CACHE_BEFORE" "$HOME/.cache/huggingface"
report_size "DOCLING_CACHE_BEFORE" "$HOME/.cache/docling"
report_size "MINERU_CACHE_BEFORE" "$HOME/.cache/mineru"

if [ ! -x "$docling_venv/bin/python" ]; then
    echo
    echo "=== CREATE DOCLING VENV ==="
    python3 -m venv "$docling_venv"
fi

if [ ! -x "$mineru_venv/bin/python" ]; then
    echo
    echo "=== CREATE MINERU VENV ==="
    python3 -m venv "$mineru_venv"
fi

echo
echo "=== INSTALL PINNED LIGHTWEIGHT RUNTIMES ==="
"$docling_venv/bin/python" -m pip install \
    'docling-slim[cli,convert-core,format-pdf]==2.131.0'

"$mineru_venv/bin/python" -m pip install \
    'mineru>=4.0,<5'

echo
echo "=== DOCLING NATIVE SDK PREFLIGHT ==="
"$docling_venv/bin/python" \
    "$root/benchmarks/document_intelligence/runtimes/docling_native.py" \
    --self-test

export RESEARCH_ENGINE_DOCLING_BIN="$docling_venv/bin/docling"
export RESEARCH_ENGINE_MINERU_BIN="$mineru_venv/bin/mineru-kit"

[ -x "$RESEARCH_ENGINE_DOCLING_BIN" ] || fail "DOCLING_EXECUTABLE_MISSING" 50
[ -x "$RESEARCH_ENGINE_MINERU_BIN" ] || fail "MINERU_EXECUTABLE_MISSING" 51

echo
echo "=== ADAPTER STATUS ==="
"$core_venv/bin/research-engine" document-adapter-status \
    | tee "$out_root/adapter-status.json"

require_stopped

echo
echo "=== DOCLING NATIVE: 7-PAPER / 73-PAGE SEED ==="
"$core_venv/bin/research-engine" document-benchmark-external \
    "$manifest" \
    docling \
    native \
    --bundle-root "$bundle_root" \
    --output-root "$out_root/docling-native" \
    | tee "$out_root/docling-native-summary.json"

require_stopped

echo
echo "=== MINERU FLASH NATIVE: 7-PAPER / 73-PAGE SEED ==="
"$core_venv/bin/research-engine" document-benchmark-external \
    "$manifest" \
    mineru \
    flash_native \
    --bundle-root "$bundle_root" \
    --output-root "$out_root/mineru-flash-native" \
    | tee "$out_root/mineru-flash-native-summary.json"

require_stopped

echo
echo "=== AFTER RUNTIME INSTALL / RUN ==="
report_size "DOCLING_VENV_AFTER" "$docling_venv"
report_size "MINERU_VENV_AFTER" "$mineru_venv"
report_size "HF_CACHE_AFTER" "$HOME/.cache/huggingface"
report_size "DOCLING_CACHE_AFTER" "$HOME/.cache/docling"
report_size "MINERU_CACHE_AFTER" "$HOME/.cache/mineru"
report_size "BAKEOFF_OUTPUT_SIZE" "$out_root"

echo
echo "=== FINAL SAFETY STATE ==="
require_stopped
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "OUTPUT_ROOT=$out_root"
echo "DOCUMENT_NATIVE_FAST_BAKEOFF=PASS"
