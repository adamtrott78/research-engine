from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-semantic-certificate-generalization-design.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_semantic_generalization_starts_with_external_validation() -> None:
    design = _load()
    selected = design["selected_validation"]
    assert selected["id"] == "SCIENTIFIC_NLI_REDUNDANCY_CERTIFICATE_VALIDATION_V1"
    assert selected["extract_scientific_state_materialized"] is False
    assert selected["production_routing_changed"] is False
    assert selected["calibration"]["no_extract_teacher_access"] is True
    assert selected["model"]["revision"] == "9d545db675cfad1e7d9c5c56351e19b4868dce65"
    assert selected["model"]["expected_model_safetensors_sha256"] == (
        "9af30c7ad7235a2054300bc2df1d98149ad6008dd1ef06212be8b32b5d1b3458"
    )


def test_mutual_entailment_gate_is_frozen_before_extract_use() -> None:
    selected = _load()["selected_validation"]
    assert selected["certificate_score"]["definition"].startswith("min(")
    assert selected["calibration"]["threshold_grid"] == [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 0.97, 0.99,
    ]
    assert selected["gates"]["spark_heldout"] == {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }
    assert selected["gates"]["scinli_heldout"] == {
        "roc_auc_min": 0.65,
        "average_precision_min": 0.40,
    }


def test_semantic_control_is_coordination_only() -> None:
    design = _load()
    assert design["coordination_boundary"]["authority_over_extract"] is False
    boundaries = design["boundaries"]
    assert boundaries["semantic_control_or_ontology_in_extract"] is False
    assert boundaries["codex_extract_changed"] is False
    assert boundaries["accepted_qwen_verify_changed"] is False
    assert boundaries["wave7b"] is False
    assert boundaries["model_training_authorized"] is False
