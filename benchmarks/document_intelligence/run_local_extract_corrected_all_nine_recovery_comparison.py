from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_corrected_all_nine_recovery_comparison import (
    evaluate_corrected_all_nine_recovery_comparison,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--wave7a3-contract", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    generated_payload = args.generated_intents.read_bytes()
    baseline_payload = args.baseline_report.read_bytes()

    report = evaluate_corrected_all_nine_recovery_comparison(
        fixture=_load(args.fixture),
        wave7a3_contract=_load(args.wave7a3_contract),
        generated=json.loads(generated_payload),
        generated_intents_sha256=hashlib.sha256(
            generated_payload
        ).hexdigest(),
        baseline_report=json.loads(baseline_payload),
        baseline_report_sha256=hashlib.sha256(
            baseline_payload
        ).hexdigest(),
        design=_load(args.design),
        workdir=args.workdir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(
        {
            "cross_paper_recovery_decision": report[
                "cross_paper_recovery_decision"
            ],
            "valid_paper_count": report["valid_paper_count"],
            "valid_stress_case_count": report[
                "valid_stress_case_count"
            ],
            "local_exact_recovery_count": report[
                "local_exact_removed_proposition_recovery_count"
            ],
            "allocation_exact_recovery_count": report[
                "allocation_exact_removed_proposition_recovery_count"
            ],
            "both_exact_recovery_count": report[
                "both_exact_recovery_count"
            ],
            "local_only_exact_recovery_count": report[
                "local_only_exact_recovery_count"
            ],
            "allocation_only_exact_recovery_count": report[
                "allocation_only_exact_recovery_count"
            ],
            "neither_exact_recovery_count": report[
                "neither_exact_recovery_count"
            ],
            "allocation_shallow_exact_recovery_count": report[
                "allocation_exact_recoveries_from_shallow_frontier_count"
            ],
            "allocation_depth_exact_recovery_count": report[
                "allocation_exact_recoveries_from_depth_extension_count"
            ],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
