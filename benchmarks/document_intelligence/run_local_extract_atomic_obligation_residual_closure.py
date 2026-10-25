from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_atomic_obligation_residual_closure import (
    evaluate_atomic_obligation_residual_closure,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--automatic-report", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-rescue-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    report = evaluate_atomic_obligation_residual_closure(
        fixture=_load(args.fixture),
        support_fixture=_load(args.support_fixture),
        generated=_load(args.generated_intents),
        generated_sha256=_sha256(args.generated_intents),
        automatic_report_sha256=_sha256(args.automatic_report),
        rescue_report=_load(args.baseline_rescue_report),
        rescue_report_sha256=_sha256(args.baseline_rescue_report),
        design=_load(args.design),
        workdir=args.workdir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "mechanism_outcome": report["mechanism_outcome"],
                "completeness_safety_outcome": report[
                    "completeness_safety_outcome"
                ],
                "acceptance_boundary_evidence": report[
                    "acceptance_boundary_evidence"
                ],
                "secondary_evaluator_outcome": report[
                    "secondary_evaluator_outcome"
                ],
                "primary_source_fidelity_support_count": report[
                    "primary_budget28"
                ]["source_fidelity_residual_support"]["supported_count"],
                "primary_legacy_anchor_support_count": report[
                    "primary_budget28"
                ]["legacy_exact_anchor_residual_support_count"],
                "prior_supported_atom_retention_count": report[
                    "primary_budget28"
                ]["prior_supported_atom_retention_count"],
                "frozen_control_retention_count": report[
                    "primary_budget28"
                ]["frozen_control_retention_count"],
                "core_obligations_satisfied": report[
                    "primary_budget28"
                ]["core_obligations_satisfied"],
                "fallback_required": report["primary_budget28"][
                    "fallback_required"
                ],
                "budget32_source_fidelity_support_count": report[
                    "append_only_budget32_sensitivity"
                ]["source_fidelity_residual_support"]["supported_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
