#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import traceback

from research_engine.local_extract_wave4 import run_wave4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    expected_uuids = [
        value.strip()
        for value in os.environ["P5_GPU_TARGET_UUIDS"].split(",")
        if value.strip()
    ]
    if len(expected_uuids) != 2:
        raise RuntimeError(
            f"Wave 4 requires exactly two admitted GPUs: {expected_uuids}"
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    try:
        report = run_wave4(
            fixture=fixture,
            workdir=args.workdir,
            model_path=os.environ["P5_MODEL_PATH"],
            expected_gpu_uuids=expected_uuids,
        )
        report["candidate_sha"] = os.environ.get("EXPECTED_SHA")
        report["model_id"] = os.environ.get("P5_MODEL_ID")
        report["model_revision"] = os.environ.get("P5_MODEL_REVISION")
        report["model_object_id"] = os.environ.get("P5_MODEL_OBJECT_ID")
        report["runtime_tree_sha256"] = os.environ.get(
            "P5_EXPANDED_RUNTIME_TREE_SHA256"
        )
        report["gpu_uuids"] = expected_uuids
        report["gpu_selected_devices"] = os.environ.get(
            "P5_GPU_SELECTED_DEVICES", ""
        ).split(",")
    except BaseException as exc:
        report = {
            "schema_version": 1,
            "operation": (
                "LOCAL_EXTRACT_WAVE4_EXACT_PROPOSAL_THEN_NORMALIZATION"
            ),
            "wave4_decision": "REJECT",
            "failure": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "candidate_sha": os.environ.get("EXPECTED_SHA"),
            "model_id": os.environ.get("P5_MODEL_ID"),
            "model_revision": os.environ.get("P5_MODEL_REVISION"),
            "model_object_id": os.environ.get("P5_MODEL_OBJECT_ID"),
            "runtime_tree_sha256": os.environ.get(
                "P5_EXPANDED_RUNTIME_TREE_SHA256"
            ),
            "gpu_uuids": expected_uuids,
            "scientific_state_materialized": False,
            "production_routing_changed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = report["metrics"]
    control = report["teacher_subset_agreement"]
    available = report["teacher_available_proposition_alignment"]
    selected = report["teacher_selected_proposition_alignment"]
    print(f"WAVE4_DECISION={report['wave4_decision']}")
    print(
        "HISTORICAL_EVIDENCE_PAGE_RECALL_AT_12="
        f"{metrics['historical_evidence_page_recall_at_12']:.6f}"
    )
    print(
        "JOB_LOCAL_COMPLETION_FRACTION="
        f"{metrics['job_local_completion_fraction']:.6f}"
    )
    print(
        "PROPOSAL_MODEL_CALLS="
        f"{metrics['proposal_model_calls']}"
    )
    print(
        "NORMALIZATION_MODEL_CALLS="
        f"{metrics['normalization_model_calls']}"
    )
    print(
        "PROJECTED_REMOTE_INPUT_TOKEN_REDUCTION="
        f"{metrics['projected_remote_input_token_reduction']:.6f}"
    )
    print(
        "TEACHER_SUBSET_PRECISION="
        f"{control['qwen_precision_at_f1_0_5']:.6f}"
    )
    print(
        "TEACHER_SUBSET_COVERAGE="
        f"{control['teacher_claim_coverage_at_f1_0_5']:.6f}"
    )
    print(
        "TEACHER_AVAILABLE_PROPOSITION_MEAN_F1="
        f"{available['mean_best_normalized_f1']:.6f}"
    )
    print(
        "TEACHER_SELECTED_PROPOSITION_MEAN_F1="
        f"{selected['mean_best_normalized_f1']:.6f}"
    )
    print("SCIENTIFIC_STATE_MATERIALIZED=false")
    print("PRODUCTION_ROUTING_CHANGED=false")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
