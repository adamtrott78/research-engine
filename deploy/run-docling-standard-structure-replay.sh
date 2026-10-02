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
    for candidate in         "$adam/varMax/PADataset"         "$adam/PADataset"
    do
        if [ -d "$candidate/papers/milcom2026/reference_notes/papers" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    candidate="$(
        find "$adam" -maxdepth 5 -type d             -path '*/PADataset/papers/milcom2026/reference_notes/papers'             -print -quit 2>/dev/null         | sed 's#/papers/milcom2026/reference_notes/papers$##'
    )"
    [ -n "$candidate" ] || return 1
    printf '%s\n' "$candidate"
}

if [ "$#" -ne 2 ]; then
    echo "usage: $0 /absolute/path/to/document-docling-standard-no-ocr-OUTPUT_ROOT /absolute/path/to/docling-native-normalized-root" >&2
    exit 2
fi

raw_root="$(readlink -f "$1")"
native_root="$(readlink -f "$2")"
normalized_root="$raw_root/docling-standard-no-ocr-paper-package-v1"

echo "=== DOCLING STANDARD/NO-OCR NORMALIZATION + CONTENT EVALUATION REPLAY ==="

systemctl --user stop "$unit" || true
require_stopped

cd "$root"
[ "$(git branch --show-current)" = "main" ] || fail "EXPECTED_MAIN_BRANCH" 30
[ -z "$(git status --porcelain)" ] || fail "CHECKOUT_DIRTY" 31
[ -x "$core_venv/bin/python" ] || fail "CORE_VENV_NOT_FOUND" 32
[ -f "$manifest" ] || fail "BENCHMARK_MANIFEST_NOT_FOUND" 33
[ -d "$raw_root" ] || fail "PRESERVED_STANDARD_OUTPUT_NOT_FOUND:$raw_root" 34
[ -d "$native_root" ] || fail "NORMALIZED_NATIVE_ROOT_NOT_FOUND:$native_root" 35
[ ! -e "$normalized_root" ] || fail "NORMALIZED_OUTPUT_ALREADY_EXISTS:$normalized_root" 36

padataset="$(find_padataset)" || fail "PADATASET_NOT_FOUND" 40
bundle_root="$padataset/papers/milcom2026/reference_notes"
[ -d "$bundle_root/papers" ] || fail "PADATASET_REFERENCE_NOTES_INVALID" 41

echo "HEAD=$(git rev-parse HEAD)"
echo "PRESERVED_STANDARD_OUTPUT_ROOT=$raw_root"
echo "NATIVE_PACKAGE_ROOT=$native_root"
echo "NORMALIZED_OUTPUT_ROOT=$normalized_root"
echo "BUNDLE_ROOT=$bundle_root"

echo
echo "=== REFRESH CORE CLI ONLY ==="
"$core_venv/bin/python" -m pip install -e "$root"

require_stopped

echo
echo "=== REPLAY PRESERVED STRUCTURED JSON; NO PARSER INFERENCE ==="
"$core_venv/bin/research-engine" document-normalize-docling-benchmark     "$manifest"     --bundle-root "$bundle_root"     --raw-docling-root "$raw_root"     --output-root "$normalized_root"     --docling-version "2.131.0"     --docling-mode "standard_no_ocr"     --native-package-root "$native_root"     | tee "$raw_root/docling-standard-normalization-evaluation-terminal-summary.json"

require_stopped

echo
echo "=== EVALUATION ARTIFACTS ==="
echo "NORMALIZED_SIZE=$(du -sh "$normalized_root" | awk '{print $1}')"
echo "SUMMARY=$normalized_root/docling-normalization-summary.json"

echo
echo "=== FINAL SAFETY STATE ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "DAEMON=inactive/dead PID=0"
echo "NORMALIZED_OUTPUT_ROOT=$normalized_root"
echo "DOCLING_STANDARD_STRUCTURE_REPLAY=PASS"
