from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

from research_engine.local_extract_obligation_validity_microbenchmark import (
    run_obligation_validity_microbenchmark,
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument(
        "--baseline-rescue-report",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    expected_gpu_uuids = [
        value
        for value in os.environ.get(
            "P5_GPU_TARGET_UUIDS",
            "",
        ).split(",")
        if value
    ]
    if len(expected_gpu_uuids) != 2:
        raise RuntimeError(
            "obligation-validity microbenchmark requires the frozen "
            "two-GPU target binding"
        )

    report = run_obligation_validity_microbenchmark(
        fixture=_load(args.fixture),
        design=_load(args.design),
        baseline_rescue_report=_load(
            args.baseline_rescue_report
        ),
        baseline_rescue_report_sha256=_sha256(
            args.baseline_rescue_report
        ),
        workdir=args.workdir,
        model_path=os.environ["P5_MODEL_PATH"],
        expected_gpu_uuids=expected_gpu_uuids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "mechanism_outcome": report[
                    "mechanism_outcome"
                ],
                "arm_a_pass": report["arms"][
                    "PROOF_CARRYING_EXPLICIT_SLOT_GROUNDING_V1"
                ]["scientific_gate"]["precision_first_pass"],
                "arm_b_pass": report["arms"][
                    "HOLISTIC_TRI_STATE_SOURCE_GROUNDED_VERIFIER_V1"
                ]["scientific_gate"]["precision_first_pass"],
                "model_calls": report["runtime_metrics"][
                    "model_calls"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
