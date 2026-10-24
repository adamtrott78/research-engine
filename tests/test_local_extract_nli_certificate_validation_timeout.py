from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-nli-certificate-validation-timeout.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_timeout_checkpoint_does_not_invent_scientific_verdict() -> None:
    row = _load()
    assert row["execution_result"] == "WORKFLOW_TIMEOUT_NO_REPORT"
    assert row["scientific_decision"] is None
    assert row["scientific_gate_consumed"] is False
    assert row["workflow"]["conclusion"] == "cancelled"
    assert row["workflow"]["artifact_count"] == 0
    assert row["scientific_state"]["report_sha256"] is None


def test_recovery_is_execution_envelope_only() -> None:
    row = _load()
    disposition = row["disposition"]
    assert disposition["replacement_run_authorized"] is True
    assert disposition["replacement_run_limit"] == 1
    assert disposition["recovery_scope"] == "EXECUTION_ENVELOPE_ONLY"
    assert disposition["allowed_change"] == [
        "Increase workflow timeout-minutes so the exact frozen scorer can finish."
    ]
    assert "batch size" in disposition["forbidden_changes"]
    assert "mutual-entailment score definition" in disposition["forbidden_changes"]


def test_semantic_control_boundary_remains_closed() -> None:
    state = _load()["scientific_state"]
    assert state["semantic_control_or_ontology_in_extract"] is False
    assert state["rdf_or_ontology_output_emitted"] is False
    assert state["extract_teacher_accessed"] is False
