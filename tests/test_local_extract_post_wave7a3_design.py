from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-post-wave7a3-coverage-design.v1.json"
)


def _load() -> dict:
    return json.loads(DESIGN.read_text())


def test_post_wave7a3_design_preserves_valid_reject_and_boundaries() -> None:
    design = _load()
    excluded = design["authoritative_exclusion_evidence"]
    assert excluded["workflow_run_id"] == 40027366528
    assert excluded["decision"] == "WAVE7A3_REJECT"
    assert excluded["scientific_gate_consumed"] is True

    boundaries = design["boundaries"]
    assert boundaries == {
        "wave7b": False,
        "production_standard_codex_extract_changed": False,
        "accepted_qwen_verify_changed": False,
        "qwen35_generation_model_changed": False,
        "retriever_capacity_reopened": False,
        "retrieval_budget_changed": False,
        "new_model_training_authorized": False,
        "new_model_download_authorized": False,
        "ontology_or_semantic_control_in_extract": False,
        "gitrunner_controller_provisioner_change": False,
    }


def test_post_wave7a3_design_reframes_residual_as_coverage_preservation() -> None:
    design = _load()
    diagnostic = design["posthoc_design_diagnostic"]
    assert diagnostic["scientific_status"] == "NON_VERDICT_DESIGN_EVIDENCE_ONLY"
    observations = diagnostic["observations"]
    assert observations["source_aware_bm25_supported_atoms"] == [
        "e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b"
    ]
    assert observations["author_anchored_bm25_supported_atoms"] == [
        "e1.a", "e3.a", "e4.a", "e4.b", "e6.a"
    ]
    assert observations["union_support_count"] == 9
    assert set(observations["union_of_both_standalone_top24_support"]) == {
        "e1.a", "e2.a", "e2.b", "e3.a", "e3.b",
        "e4.a", "e4.b", "e6.a", "e6.b",
    }


def test_post_wave7a3_first_diagnostic_is_zero_generation_and_teacher_blind() -> None:
    design = _load()
    selected = design["selected_first_diagnostic"]
    assert selected["id"] == "MULTIVIEW_SUBMODULAR_COVERAGE_DIAGNOSTIC_V1"
    assert selected["new_model_calls"] == 0
    assert selected["new_model_downloads"] == 0
    assert selected["scientific_state_materialized"] is False
    assert selected["retrieval"]["retriever"] == "BM25_DIRECT_ONLY_FOR_FIRST_DIAGNOSTIC"
    assert selected["retrieval"]["final_global_proposition_budget"] == 24
    assert selected["selector"]["id"] == "BALANCED_FACILITY_LOCATION_RRF_V1"
    assert selected["selector"]["teacher_blind"] is True
    assert selected["selector"]["no_weight_tuning_after_result"] is True


def test_post_wave7a3_design_keeps_ontology_downstream_only() -> None:
    design = _load()
    coordination = design["coordination_boundary"]
    assert coordination["parallel_task"] == "workspace-knowledge-action-ontology-v1"
    assert coordination["authority"] == "coordination_only"
    assert (
        coordination["future_integration"]
        == "accepted EXTRACT output -> versioned provenance-bearing evidence/output adapter -> semantic-control prototype"
    )
