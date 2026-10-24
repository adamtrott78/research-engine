from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-predicate-argument-evidence-unit-design.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_selected_validation_is_external_and_teacher_blind() -> None:
    row = _load()
    selected = row["selected_validation"]
    assert selected["id"] == "DEPENDENCY_PREDICATE_ARGUMENT_CERTIFICATE_VALIDATION_V1"
    assert selected["extract_scientific_state_materialized"] is False
    assert selected["production_routing_changed"] is False
    assert row["boundaries"]["extract_teacher_used_for_validation"] is False
    assert selected["evidence_unit_extraction"]["teacher_blind"] is True


def test_parser_and_external_data_are_pinned() -> None:
    selected = _load()["selected_validation"]
    parser = selected["parser"]
    assert parser["package_version"] == "3.8.2"
    assert parser["model_version"] == "3.8.0"
    assert parser["model_wheel_sha256"] == (
        "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
    )
    ids = {row["id"]: row for row in selected["external_validation_sets"]}
    assert ids["SPARK_PROPOSITION_CLUSTERING_MANUAL"]["repository_commit"] == (
        "71465f36c83cfef589facadcb014e70c0bad1b3b"
    )
    assert ids["FOBIE_SCIENTIFIC_RELATION_FIDELITY"]["repository_commit"] == (
        "b49683fa2dced187d305abadcc36c55e2fb16c44"
    )


def test_gate_and_unit_bounds_are_frozen() -> None:
    selected = _load()["selected_validation"]
    assert selected["evidence_unit_extraction"]["max_units_per_sentence"] == 12
    assert selected["redundancy_score"]["threshold_grid"] == [
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ]
    assert selected["gates"]["spark_heldout"] == {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }
    assert selected["gates"]["fobie_test"] == {
        "full_relation_recall_min": 0.50,
        "sentence_with_at_least_one_unit_fraction_min": 0.95,
        "mean_units_per_sentence_max": 8.0,
    }


def test_review_and_semantic_control_cannot_own_extract() -> None:
    row = _load()
    assert row["independent_review_boundary"]["authority_over_extract"] is False
    assert row["coordination_boundary"]["authority_over_extract"] is False
    assert row["boundaries"]["semantic_control_or_ontology_in_extract"] is False
    assert row["boundaries"]["rdf_or_ontology_output"] is False
