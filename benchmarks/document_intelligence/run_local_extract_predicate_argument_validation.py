from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

from research_engine.local_extract_nli_certificate_validation import (
    binary_metrics,
    build_spark_pairs,
    select_threshold,
)
from research_engine.local_extract_predicate_argument_validation import (
    evaluate_fobie,
    score_spark_pairs,
    validation_verdict,
)


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--spark-json", type=pathlib.Path, required=True)
    parser.add_argument("--fobie-dev-json", type=pathlib.Path, required=True)
    parser.add_argument("--fobie-test-json", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    design = _load(args.design)
    selected = design["selected_validation"]
    if selected["id"] != "DEPENDENCY_PREDICATE_ARGUMENT_CERTIFICATE_VALIDATION_V1":
        raise RuntimeError("unexpected validation design")

    import spacy

    nlp = spacy.load("en_core_web_sm")
    extraction = selected["evidence_unit_extraction"]
    max_units = int(extraction["max_units_per_sentence"])

    spark = _load(args.spark_json)
    pairs = build_spark_pairs(spark, negative_per_positive=5)

    cal_labels, cal_scores, cal_diag = score_spark_pairs(
        list(pairs["calibration"]),
        nlp=nlp,
        max_units_per_sentence=max_units,
    )
    held_labels, held_scores, held_diag = score_spark_pairs(
        list(pairs["heldout"]),
        nlp=nlp,
        max_units_per_sentence=max_units,
    )

    threshold_grid = [
        float(value)
        for value in selected["redundancy_score"]["threshold_grid"]
    ]
    selected_threshold = select_threshold(
        cal_labels,
        cal_scores,
        threshold_grid,
        precision_min=0.95,
        recall_min=0.05,
    )

    if selected_threshold is None:
        heldout_metrics = {
            "threshold": None,
            "tp": 0,
            "fp": 0,
            "tn": sum(1 for value in held_labels if value == 0),
            "fn": sum(1 for value in held_labels if value == 1),
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
        }
    else:
        heldout_metrics = binary_metrics(
            held_labels,
            held_scores,
            float(selected_threshold["threshold"]),
        )

    fobie_dev = evaluate_fobie(
        _load(args.fobie_dev_json),
        nlp=nlp,
        max_units_per_sentence=max_units,
        relation_match_threshold=0.50,
    )
    fobie_test = evaluate_fobie(
        _load(args.fobie_test_json),
        nlp=nlp,
        max_units_per_sentence=max_units,
        relation_match_threshold=0.50,
    )

    decision = validation_verdict(
        threshold_selected=selected_threshold is not None,
        spark_heldout=heldout_metrics,
        fobie_test=fobie_test,
        gates=selected["gates"],
    )

    report = {
        "schema_version": 1,
        "operation": "DEPENDENCY_PREDICATE_ARGUMENT_CERTIFICATE_VALIDATION",
        "design_id": design["design_id"],
        **decision,
        "parser": {
            "package": selected["parser"]["package"],
            "package_version": selected["parser"]["package_version"],
            "model": selected["parser"]["model"],
            "model_version": selected["parser"]["model_version"],
            "model_wheel_sha256": selected["parser"][
                "model_wheel_sha256"
            ],
            "spacy_runtime_version": spacy.__version__,
        },
        "external_data": {
            "spark": {
                "repository_commit": selected[
                    "external_validation_sets"
                ][0]["repository_commit"],
                "git_blob_sha": selected[
                    "external_validation_sets"
                ][0]["git_blob_sha"],
                "file_sha256": _sha256(args.spark_json),
            },
            "fobie": {
                "repository_commit": selected[
                    "external_validation_sets"
                ][1]["repository_commit"],
                "dev_git_blob_sha": selected[
                    "external_validation_sets"
                ][1]["dev_git_blob_sha"],
                "test_git_blob_sha": selected[
                    "external_validation_sets"
                ][1]["test_git_blob_sha"],
                "dev_file_sha256": _sha256(args.fobie_dev_json),
                "test_file_sha256": _sha256(args.fobie_test_json),
            },
        },
        "spark": {
            "calibration_topic_count": len(
                pairs["calibration_topics"]
            ),
            "heldout_topic_count": len(pairs["heldout_topics"]),
            "calibration_positive_count": pairs[
                "calibration_positive_count"
            ],
            "calibration_negative_count": pairs[
                "calibration_negative_count"
            ],
            "heldout_positive_count": pairs[
                "heldout_positive_count"
            ],
            "heldout_negative_count": pairs[
                "heldout_negative_count"
            ],
            "selected_threshold": (
                float(selected_threshold["threshold"])
                if selected_threshold is not None
                else None
            ),
            "calibration_metrics": selected_threshold,
            "heldout_metrics": heldout_metrics,
            "calibration_extraction_diagnostics": cal_diag,
            "heldout_extraction_diagnostics": held_diag,
        },
        "fobie": {
            "dev": fobie_dev,
            "test": fobie_test,
        },
        "hard_safety": {
            "extract_teacher_accessed": False,
            "extract_scientific_state_materialized": False,
            "predicate_argument_certificate_used_in_extract": False,
            "production_routing_changed": False,
            "model_or_parser_training_executed": False,
            "wave7b_started": False,
            "semantic_control_or_ontology_in_extract": False,
            "rdf_or_ontology_output_emitted": False,
        },
        "extract_teacher_accessed": False,
        "extract_scientific_state_materialized": False,
        "predicate_argument_certificate_used_in_extract": False,
        "production_routing_changed": False,
        "model_or_parser_training_executed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_in_extract": False,
        "rdf_or_ontology_output_emitted": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "predicate_argument_certificate_decision": report[
                "predicate_argument_certificate_decision"
            ],
            "selected_threshold": report["spark"][
                "selected_threshold"
            ],
            "spark_heldout": report["spark"][
                "heldout_metrics"
            ],
            "fobie_test": report["fobie"]["test"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
