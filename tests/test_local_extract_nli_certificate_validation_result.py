from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-nli-certificate-validation-result.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_nli_validation_is_rejected_without_threshold_tuning() -> None:
    row = _load()
    assert row["nli_certificate_decision"] == "NLI_CERTIFICATE_REJECTED"
    assert row["spark"]["selected_threshold"] is None
    assert row["spark"]["calibration_metrics"] is None
    disposition = row["scientific_disposition"]
    assert disposition["learned_nli_extract_experiment_authorized"] is False
    assert disposition["nli_threshold_tuning_authorized"] is False
    assert disposition["nli_rerun_authorized"] is False


def test_scinli_pass_does_not_override_redundancy_safety_failure() -> None:
    row = _load()
    assert row["scinli"]["roc_auc"] >= row["scinli"]["gate_roc_auc_min"]
    assert row["scinli"]["average_precision"] >= row["scinli"]["gate_average_precision_min"]
    assert row["scinli"]["gate_pass"] is True
    assert row["nli_certificate_decision"] == "NLI_CERTIFICATE_REJECTED"


def test_exact_result_identity_and_safety_are_preserved() -> None:
    row = _load()
    assert row["scientific_run"]["run_id"] == 40049443488
    assert row["scientific_run"]["artifact_id"] == 12544077187
    assert row["scientific_run"]["artifact_zip_sha256"] == (
        "4282b5434de94e84ce62b71b976ac377dac92e0c7a17f230366de03682ba29cb"
    )
    assert row["scientific_run"]["report_sha256"] == (
        "2e07359b7ca23c563134904f3dd1f8bb04116cfb1b4490f4855b6f28adba9333"
    )
    safety = row["hard_safety"]
    assert safety["extract_teacher_accessed"] is False
    assert safety["learned_certificate_used_in_extract"] is False
    assert safety["semantic_control_or_ontology_in_extract"] is False
    assert row["coordination_boundary"]["extract_integration_signal_ready"] is False
