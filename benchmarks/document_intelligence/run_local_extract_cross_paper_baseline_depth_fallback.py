from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_cross_paper_baseline_depth_fallback import (
    evaluate_cross_paper_baseline_depth_fallback,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--wave7a3-contract", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--source-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    generated_payload = args.generated_intents.read_bytes()
    source_payload = args.source_report.read_bytes()

    report = evaluate_cross_paper_baseline_depth_fallback(
        fixture=_load(args.fixture),
        wave7a3_contract=_load(args.wave7a3_contract),
        generated=json.loads(generated_payload),
        generated_intents_sha256=hashlib.sha256(generated_payload).hexdigest(),
        source_report=json.loads(source_payload),
        source_report_sha256=hashlib.sha256(source_payload).hexdigest(),
        design=_load(args.design),
        workdir=args.workdir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    failing = report["failing_paper"]
    fallback = failing.get("fallback_event") or {}
    print(json.dumps({
        "baseline_depth_fallback_decision": report[
            "baseline_depth_fallback_decision"
        ],
        "original_complete_paper_count": report[
            "original_complete_paper_count"
        ],
        "final_complete_paper_count": report["final_complete_paper_count"],
        "failing_original_selected_count": failing[
            "original_selected_count"
        ],
        "failing_original_event_count": failing["original_event_count"],
        "failing_final_selected_count": failing["final_selected_count"],
        "failing_final_event_count": failing["final_event_count"],
        "fallback_event_added": failing["fallback_event_added"],
        "fallback_event_proposition_id": fallback.get("proposition_id"),
        "fallback_event_bm25_rank": fallback.get("bm25_rank"),
        "fallback_candidate_count_examined": report["cost"][
            "fallback_candidate_count_examined"
        ],
        "fallback_raw_source_chars_examined": report["cost"][
            "fallback_raw_source_chars_examined"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
