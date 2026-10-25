from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from research_engine.local_extract_atomic_obligation_residual_closure import (
    evaluate_atomic_obligation_residual_closure,
    git_blob_sha,
)


def _read_json(path: pathlib.Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    return json.loads(payload), payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-rescue-report", type=pathlib.Path, required=True)
    parser.add_argument("--generated-intents", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fixture, fixture_bytes = _read_json(args.fixture)
    support, support_bytes = _read_json(args.support_fixture)
    design, design_bytes = _read_json(args.design)
    baseline, baseline_bytes = _read_json(args.baseline_rescue_report)
    generated, generated_bytes = _read_json(args.generated_intents)

    report = evaluate_atomic_obligation_residual_closure(
        fixture=fixture,
        fixture_git_blob=git_blob_sha(fixture_bytes),
        support_fixture=support,
        support_fixture_git_blob=git_blob_sha(support_bytes),
        design=design,
        design_git_blob=git_blob_sha(design_bytes),
        baseline_report=baseline,
        baseline_report_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        generated=generated,
        generated_sha256=hashlib.sha256(generated_bytes).hexdigest(),
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
                "runtime_core_obligations_satisfied": report[
                    "runtime_core_obligations_satisfied"
                ],
                "primary_source_fidelity_residual_support": report[
                    "primary_source_fidelity_residual_support"
                ]["supported_count"],
                "sensitivity_source_fidelity_residual_support": report[
                    "sensitivity_source_fidelity_residual_support"
                ]["supported_count"],
                "prior_supported_atom_retention": report[
                    "prior_supported_atom_retention"
                ]["retained_count"],
                "frozen_control_retention": report[
                    "frozen_control_retention"
                ]["retained_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
