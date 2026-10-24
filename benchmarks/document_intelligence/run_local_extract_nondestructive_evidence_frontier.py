from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_nondestructive_evidence_frontier import (
    evaluate_nondestructive_evidence_frontier,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--wave7a3-contract", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--wave7a3-result", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    payload = args.generated_intents.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    result_checkpoint = _load(args.wave7a3_result)
    expected_sha = str(result_checkpoint["generated_intents_sha256"])
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"generated-intents identity mismatch: {actual_sha} != {expected_sha}"
        )

    report = evaluate_nondestructive_evidence_frontier(
        fixture=_load(args.fixture),
        wave7a3_contract=_load(args.wave7a3_contract),
        support_fixture=_load(args.support_fixture),
        generated=json.loads(payload),
        design=_load(args.design),
        generated_intents_sha256=actual_sha,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(
        {
            "frontier_decision": report["frontier_decision"],
            "qualifying_budget": report["qualifying_budget"],
            "teacher_support_by_budget": {
                budget: row["teacher"]["supported_atom_count"]
                for budget, row in report["frontier_metrics"].items()
            },
            "residual_recovery_by_budget": {
                budget: row["teacher"]["observed_residual_atoms_recovered"]
                for budget, row in report["frontier_metrics"].items()
            },
            "immutable_core_prefix_preserved": report["hard_safety"][
                "immutable_core_prefix_preserved_all_papers"
            ],
            "residual_prefix24_matches_frozen": report["hard_safety"][
                "residual_prefix24_matches_frozen_all_papers"
            ],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
