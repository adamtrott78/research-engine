#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research_engine.qwen_runtime_certification import (
    RUNTIME_CONTRACT_V3,
    operation_identity,
    probe_completed_operation,
)

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SUCCESS_SHA = "a3d80e9329e231889b11430f98c0de342349fe1f"
CROP_SHAS = [
    "59a1c0e4cd1413ff329931a4d9bf9082d46fd2a316bdfccfc59858a99b23b04a",
    "fc9b9b633ead850c313a2b17181eed52146f29b4db753baa9cc90afbcb134579",
]


def expected_identity() -> str:
    return operation_identity(
        {
            "operation": "qwen3.5-4b-targeted-visual-fallback-benchmark",
            "git_source_identity": SUCCESS_SHA,
            "model": {"revision": MODEL_REVISION},
            "runtime": {
                "contract_version": RUNTIME_CONTRACT_V3,
                "sampling_backend": "vllm_native",
            },
            "renders": {
                "targets": [
                    {"crop_sha256": value}
                    for value in CROP_SHAS
                ]
            },
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dry-run recovery probe for the exact completed Qwen benchmark."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "adamArchives/Adam/workspace-control-tmp",
    )
    p.add_argument("--receipt", type=Path)
    args = p.parse_args()

    if args.receipt is not None:
        candidates = [args.receipt]
    else:
        candidates = sorted(
            args.root.glob("qwen35-visual-benchmark-*/qwen35-benchmark-receipt.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    result = probe_completed_operation(
        candidates,
        expected_operation_identity=expected_identity(),
    )
    result["probe_mode"] = "NON_DESTRUCTIVE_DRY_RUN"
    result["benchmark_launch_attempted"] = False
    result["package_install_or_download_attempted"] = False
    result["model_initialization_attempted"] = False
    result["inference_attempted"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "COMPLETED_REUSE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
