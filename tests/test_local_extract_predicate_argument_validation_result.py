from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-predicate-argument-validation-result.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_predicate_argument_result_is_rejected_without_tuning() -> None:
    row = _load()
    assert row["predicate_argument_certificate_decision"] == (
        "PREDICATE_ARGUMENT_CERTIFICATE_REJECTED"
    )
    assert row["spark"]["selected_threshold"] is None
    assert row["scientific_disposition"][
        "predicate_argument_rerun_authorized"
    ] is False
    assert row["scientific_disposition"][
        "predicate_argument_threshold_tuning_authorized"
    ] is False


def test_fobie_failure_is_not_empty_representation() -> None:
    test = _load()["fobie"]["test"]
    assert test["tradeoff_relation_count"] == 153
    assert test["recovered_tradeoff_relation_count"] == 4
    assert test["full_relation_recall"] < 0.50
    assert test["sentence_with_at_least_one_unit_fraction"] == 1.0
    assert test["mean_units_per_sentence"] <= 8.0


def test_hard_safety_and_coordination_boundaries_hold() -> None:
    row = _load()
    safety = row["hard_safety"]
    assert safety["extract_teacher_accessed"] is False
    assert safety["predicate_argument_certificate_used_in_extract"] is False
    assert safety["production_routing_changed"] is False
    assert safety["wave7b_started"] is False
    assert safety["semantic_control_or_ontology_in_extract"] is False
    assert safety["rdf_or_ontology_output_emitted"] is False
    assert row["independent_review_boundary"]["authority_over_extract"] is False
