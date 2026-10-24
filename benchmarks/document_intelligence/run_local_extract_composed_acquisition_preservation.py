from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_composed_acquisition_preservation import (
    evaluate_composed_acquisition_preservation,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--facet-report", type=pathlib.Path, required=True)
    parser.add_argument("--alignment-result", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    facet_payload = args.facet_report.read_bytes()
    facet_sha = hashlib.sha256(facet_payload).hexdigest()

    report = evaluate_composed_acquisition_preservation(
        fixture=_load(args.fixture),
        support_fixture=_load(args.support_fixture),
        design=_load(args.design),
        facet_report=json.loads(facet_payload),
        facet_report_sha256=facet_sha,
        alignment_result=_load(args.alignment_result),
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(
        {
            "composition_decision": report["composition_decision"],
            "supported_atom_count": report["primary_teacher"][
                "supported_atom_count"
            ],
            "supported_atoms": report["primary_teacher"]["supported_atoms"],
            "sensitivity_supported_atom_count": report[
                "posthoc_flagged_origin_sensitivity_teacher"
            ]["supported_atom_count"],
            "sensitivity_supported_atoms": report[
                "posthoc_flagged_origin_sensitivity_teacher"
            ]["supported_atoms"],
            "fixed_budget": report["fixed_budget"],
            "incremental_raw_source_chars_vs_top24": report[
                "cost_provenance"
            ]["incremental_raw_source_chars_vs_top24"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
