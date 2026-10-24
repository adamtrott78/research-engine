from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-amr-graph-certificate-design.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_amr_validation_is_external_and_teacher_blind() -> None:
    row = _load()
    selected = row["selected_validation"]
    assert selected["id"] == "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1"
    assert selected["graph_similarity"]["teacher_blind"] is True
    assert row["hard_boundaries"]["extract_teacher_accessed"] is False
    assert row["hard_boundaries"]["amr_certificate_used_in_extract"] is False


def test_parser_identity_and_bio_gate_are_frozen() -> None:
    selected = _load()["selected_validation"]
    model = selected["parser_model"]
    assert model["release_asset_id"] == 57062880
    assert model["release_asset_size_bytes"] == 515968261
    assert model["published_md5"] == "f8eb889468f4d0c3c0677fc910fb2240"
    assert model["inference"]["num_beams"] == 1
    bio = selected["external_validation_sets"][0]
    assert bio["expected_amr_count"] == 500
    assert bio["predeclared_gate"] == {
        "corpus_smatch_f1_min": 0.60,
        "valid_prediction_fraction_min": 0.98,
    }


def test_spark_gate_and_threshold_grid_are_frozen() -> None:
    selected = _load()["selected_validation"]
    assert selected["graph_similarity"]["threshold_grid"] == [
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ]
    spark = selected["external_validation_sets"][1]
    assert spark["heldout_gate"] == {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }


def test_next_phase_does_not_authorize_extract_or_semantic_control() -> None:
    row = _load()
    assert row["hard_boundaries"]["production_routing_changed"] is False
    assert row["hard_boundaries"]["wave7b"] is False
    assert row["hard_boundaries"]["parser_training_authorized"] is False
    assert row["independent_review_boundary"]["authority_over_extract"] is False
    assert row["coordination_boundary"]["authority_over_extract"] is False
    assert row["hard_boundaries"]["semantic_control_or_ontology_in_extract"] is False
    assert row["hard_boundaries"]["rdf_or_ontology_output"] is False
