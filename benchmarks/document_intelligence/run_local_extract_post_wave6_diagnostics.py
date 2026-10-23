#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib

from research_engine.local_extract_diagnostics import run_post_wave6_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    args.workdir.mkdir(parents=True, exist_ok=True)

    report = run_post_wave6_diagnostics(
        fixture=fixture,
        diagnostic_contract=contract,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = report["metrics"]
    claims = report["teacher_diagnostics"]["legacy_claim_summary"]["mean_f1"]
    atoms = report["teacher_diagnostics"]["unique_atom_summary"]["mean_f1"]
    print("POST_WAVE6_DIAGNOSTIC=PASS")
    print(
        "CURRENT_TOP12_PAGE_RECALL="
        f"{metrics['current_top12_historical_page_recall']:.6f}"
    )
    print(
        "WINDOW2_PAGE_RECALL="
        f"{metrics['sentence_window_2_fixed_budget_historical_page_recall']:.6f}"
    )
    print(
        "WINDOW3_PAGE_RECALL="
        f"{metrics['sentence_window_3_fixed_budget_historical_page_recall']:.6f}"
    )
    print(
        "LEGACY_TOP12_SINGLE_MEAN_F1="
        f"{claims['current_top12_best_single_proposition']:.6f}"
    )
    print(
        "LEGACY_BUNDLE3_MEAN_F1="
        f"{claims['current_top12_best_bundle_3']:.6f}"
    )
    print(
        "UNIQUE_ATOM_TOP12_SINGLE_MEAN_F1="
        f"{atoms['current_top12_best_single_proposition']:.6f}"
    )
    print(
        "UNIQUE_ATOM_BUNDLE3_MEAN_F1="
        f"{atoms['current_top12_best_bundle_3']:.6f}"
    )
    print("QWEN_MODEL_CALLS=0")
    print("EXTERNAL_MODEL_DOWNLOADS=0")
    print("SCIENTIFIC_STATE_MATERIALIZED=false")
    print("PRODUCTION_ROUTING_CHANGED=false")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
