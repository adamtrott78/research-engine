#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin_dir="$HOME/.local/bin"
mkdir -p "$bin_dir"

for cmd in START STOP HUD STATUS; do
    install -m 0755 "$root/deploy/operator/$cmd" "$bin_dir/$cmd"
done

if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
    bashrc="$HOME/.bashrc"
    marker="# research-engine operator shortcuts"
    if ! grep -Fq "$marker" "$bashrc" 2>/dev/null; then
        printf "\n%s\nexport PATH=\"$HOME/.local/bin:$PATH\"\n" "$marker" >> "$bashrc"
    fi
fi

echo "Installed: START STOP HUD STATUS"
echo "Shortcut directory: $bin_dir"
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
    echo "Open a new terminal or run: source ~/.bashrc"
fi
