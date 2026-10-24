from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_neighborhood_rescue import (
    evaluate_neighborhood_rescue,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--composed-report", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    composed_payload = args.composed_report.read_bytes()
    composed_sha = hashlib.sha256(composed_payload).hexdigest()
    generated_payload = args.generated_intents.read_bytes()
    generated_sha = hashlib.sha256(generated_payload).hexdigest()

    report = evaluate_neighborhood_rescue(
        fixture=_load(args.fixture),
        support_fixture=_load(args.support_fixture),
        design=_load(args.design),
        composed_report=json.loads(composed_payload),
        composed_report_sha256=composed_sha,
        generated=json.loads(generated_payload),
        generated_intents_sha256=generated_sha,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rescued = report.get("rescued_teacher") or {}
    chosen = report.get("chosen_rescue")
    print(json.dumps(
        {
            "neighborhood_rescue_decision": report[
                "neighborhood_rescue_decision"
            ],
            "chosen_rescue_proposition_id": (
                chosen["proposition_id"] if chosen is not None else None
            ),
            "eligible_neighbor_count": len(
                report["eligible_neighbor_candidates"]
            ),
            "supported_atom_count": rescued.get("supported_atom_count"),
            "supported_atoms": rescued.get("supported_atoms"),
            "fixed_selected_count": report["cost_provenance"][
                "selected_proposition_count"
            ],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
