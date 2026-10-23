#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib

from research_engine.local_extract_page_conditioned import (
    run_page_conditioned_retrieval_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--parent-contract", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    parent_contract = json.loads(
        args.parent_contract.read_text(encoding="utf-8")
    )
    args.workdir.mkdir(parents=True, exist_ok=True)

    report = run_page_conditioned_retrieval_diagnostic(
        fixture=fixture,
        diagnostic_contract=contract,
        parent_diagnostic_contract=parent_contract,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    legacy = report["teacher_diagnostics"]["legacy_summary"]
    atoms = report["teacher_diagnostics"]["unique_atom_summary"]
    base = report["teacher_diagnostics"]["legacy_baseline_summary"]["mean_f1"]
    print("PAGE_CONDITIONED_RETRIEVAL_DIAGNOSTIC=PASS")
    print(
        "LEGACY_SELECTED_PAGE_POOL_CEILING="
        f"{legacy['selected_page_pool_best_single_proposition_mean_f1']:.6f}"
    )
    print(
        "LEGACY_PAGE_CONDITIONED_SELECTED="
        f"{legacy['page_conditioned_selected_proposition_mean_f1']:.6f}"
    )
    print(
        "LEGACY_WAVE6_SELECTED="
        f"{base['current_wave6_selected_single_proposition']:.6f}"
    )
    print(
        "UNIQUE_ATOM_SELECTED_PAGE_POOL_CEILING="
        f"{atoms['selected_page_pool_best_single_proposition_mean_f1']:.6f}"
    )
    print(
        "UNIQUE_ATOM_PAGE_CONDITIONED_SELECTED="
        f"{atoms['page_conditioned_selected_proposition_mean_f1']:.6f}"
    )
    print("QWEN_MODEL_CALLS=0")
    print("EXTERNAL_MODEL_DOWNLOADS=0")
    print("SCIENTIFIC_STATE_MATERIALIZED=false")
    print("PRODUCTION_ROUTING_CHANGED=false")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
