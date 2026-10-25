from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_local_first_allocation_fallback import (
    evaluate_local_first_allocation_fallback,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--source-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source_payload = args.source_report.read_bytes()
    report = evaluate_local_first_allocation_fallback(
        source_report=json.loads(source_payload),
        source_report_sha256=hashlib.sha256(source_payload).hexdigest(),
        design=_load(args.design),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "hybrid_realization_decision": report[
                    "hybrid_realization_decision"
                ],
                "paper_count": report["paper_count"],
                "stress_case_count": report["stress_case_count"],
                "hybrid_exact_recovery_count": report[
                    "hybrid_exact_recovery_count"
                ],
                "known_local_exact_recoveries_preserved_count": report[
                    "known_local_exact_recoveries_preserved_count"
                ],
                "known_allocation_only_recoveries_captured_count": report[
                    "known_allocation_only_recoveries_captured_count"
                ],
                "hybrid_local_selected_case_count": report[
                    "hybrid_local_selected_case_count"
                ],
                "hybrid_allocation_fallback_case_count": report[
                    "hybrid_allocation_fallback_case_count"
                ],
                "hybrid_candidate_count_examined": report["cost"][
                    "hybrid_candidate_count_examined"
                ],
                "hybrid_candidate_raw_source_chars_examined": report["cost"][
                    "hybrid_candidate_raw_source_chars_examined"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
