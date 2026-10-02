#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adam="$HOME/adamArchives/Adam"
state="$adam/.research-engine"
core_venv="$state/venv"
unit="research-engine-daemon.service"

manifest="$root/benchmarks/document_intelligence/cases/milcom-mathpix-seed-v1.json"

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

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /absolute/path/to/document-native-fast-OUTPUT_ROOT" >&2
    exit 2
fi

raw_root="$(readlink -f "$1")"
raw_docling_root="$raw_root/docling-native"
normalized_root="$raw_root/docling-paper-package-v1"

echo "=== DOCLING STRUCTURED NORMALIZATION REPLAY ==="

systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 33
[ -d "$raw_docling_root" ] || fail "RAW_DOCLING_ROOT_NOT_FOUND:$raw_docling_root" 34

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

echo "HEAD=$(git rev-parse HEAD)"
echo "RAW_ROOT=$raw_root"
echo "RAW_DOCLING_ROOT=$raw_docling_root"
echo "NORMALIZED_ROOT=$normalized_root"
echo "BUNDLE_ROOT=$bundle_root"

if [ -e "$normalized_root" ]; then
    fail "NORMALIZED_OUTPUT_ALREADY_EXISTS:$normalized_root" 50
fi

echo
echo "=== CORE VENV REFRESH ==="
"$core_venv/bin/python" -m pip install -e "$root"

require_stopped

echo
echo "=== NORMALIZE EXISTING DOCLING STRUCTURED OUTPUT ==="
"$core_venv/bin/research-engine" document-normalize-docling-benchmark \
    "$manifest" \
    --bundle-root "$bundle_root" \
    --raw-docling-root "$raw_docling_root" \
    --output-root "$normalized_root" \
    | tee "$raw_root/docling-normalization-terminal-summary.json"

require_stopped

echo
echo "=== NORMALIZATION ARTIFACTS ==="
echo "NORMALIZED_SIZE=$(du -sh "$normalized_root" | awk '{print $1}')"
echo "SUMMARY=$normalized_root/docling-normalization-summary.json"

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "NORMALIZED_ROOT=$normalized_root"
echo "DOCLING_STRUCTURED_NORMALIZATION=PASS"
