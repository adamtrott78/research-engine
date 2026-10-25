from __future__ import annotations

import json
from pathlib import Path


DESIGN = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-atomic-obligation-residual-closure-design.v1.json"
)


def _design() -> dict:
    return json.loads(DESIGN.read_text())


def test_design_is_development_only_and_teacher_blind() -> None:
    design = _design()
    assert design["status"] == "FROZEN_DESIGN_ONLY"
    assert design["selected_discriminating_experiment"]["id"] == (
        "ATOMIC_OBLIGATION_RESIDUAL_CLOSURE_DIAGNOSTIC_V1"
    )
    boundaries = design["frozen_scientific_boundaries"]
    assert boundaries["development_only"] is True
    assert boundaries["teacher_labels_used_for_runtime_selection"] is False
    assert boundaries["source_fidelity_equivalence_sets_used_for_runtime_selection"] is False
    assert boundaries["sealed_acceptance_benchmark_accessed"] is False
    assert boundaries["acceptance_candidate_frozen"] is False


def test_primary_budget_and_sensitivity_are_distinct() -> None:
    exp = _design()["selected_discriminating_experiment"]
    frozen = exp["frozen_inputs"]
    assert frozen["exact_primary_budget"] == 28
    assert frozen["diagnostic_append_only_budget"] == 32
    assert exp["append_only_budget32_sensitivity"]["purpose"].startswith(
        "Cause discrimination only"
    )
    assert exp["primary_budget28_selection"]["no_teacher_selection"] is True


def test_four_core_runtime_obligations_are_answer_free() -> None:
    exp = _design()["selected_discriminating_experiment"]
    obligations = exp["runtime_obligations"]
    assert len(obligations) == 4
    assert all(row["core"] is True for row in obligations)
    text = json.dumps(obligations).lower()
    for forbidden in (
        "page:p2:s34",
        "page:p2:s35",
        "page:p2:s17",
        "page:p6:s15",
        "page:p6:s09",
        "e1.b",
        "e1.c",
        "e3.c",
        "e6.c",
    ):
        assert forbidden not in text


def test_posthoc_equivalence_sets_are_evaluation_only() -> None:
    exp = _design()["selected_discriminating_experiment"]
    posthoc = exp["posthoc_evaluation_only"]
    assert posthoc["source_fidelity_sets_used_for_runtime_selection"] is False
    assert posthoc["source_fidelity_equivalence_sets"] == {
        "e1.b": ["page:p2:s34", "page:p4:s15"],
        "e1.c": ["page:p2:s35"],
        "e3.c": ["page:p2:s17"],
        "e6.c": ["page:p6:s15", "page:p6:s09"],
    }


def test_mechanism_and_completeness_outcomes_are_independent() -> None:
    exp = _design()["selected_discriminating_experiment"]
    mechanism = exp["mechanism_outcome_partition"]
    completeness = exp["completeness_safety_outcome_partition"]
    boundary = exp["acceptance_boundary_evidence"]
    assert "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED" in mechanism
    assert "BUDGET_SELECTION_COLLISION" in mechanism
    assert "OBLIGATION_EXPOSURE_GAP" in mechanism
    assert "PRESERVATION_REGRESSION" in mechanism
    assert "SAFE_INCOMPLETENESS_CONFIRMED" in completeness
    assert "UNSAFE_FALSE_COMPLETE" in completeness
    assert "RESIDUAL_CLOSURE" in boundary
    assert "SAFE_INCOMPLETENESS" in boundary
    assert exp["completeness_controller"]["incomplete_action"] == (
        "LOCAL_INCOMPLETE_REQUIRE_FALLBACK"
    )
    assert exp["reporting_rule"].startswith(
        "Always report mechanism_outcome and completeness_safety_outcome independently"
    )


def test_no_acceptance_authority() -> None:
    design = _design()
    assert (
        design["interpretation_authority"]["acceptance_freeze_authority"]
        == "NONE"
    )


def test_review_correction_freezes_exact_corpus_and_matcher() -> None:
    design = _design()
    exp = design["selected_discriminating_experiment"]
    frozen = exp["frozen_inputs"]
    assert frozen["corpus_fixture_path"] == (
        "benchmarks/document_intelligence/local-extract-wave1-corpus.v1.json"
    )
    assert frozen["corpus_fixture_blob"] == (
        "c472329d560df3d7dccab2bf49bf94d6805faef2"
    )
    assert frozen["teacher_pdf_sha256"] == (
        "1097bdfabf6cd1b1e26ff9df58b6bb810e4bf74fbd3890ad20e86316d2ee0c91"
    )
    matcher = exp["runtime_candidate_generation"][
        "normalization_and_witness_matching"
    ]
    assert matcher["unicode_normalization"] == "NFKD"
    assert matcher["tokenization_regex"] == "[a-z0-9]+"
    assert "token prefix" in matcher["witness_term_semantics"]


def test_review_correction_freezes_protected_origins_and_precedence() -> None:
    exp = _design()["selected_discriminating_experiment"]
    selection = exp["primary_budget28_selection"]
    assert selection["protected_selection_origins"] == [
        "salience_stratified_reservation",
        "validated_alignment_swap",
        "local_neighborhood_rescue",
    ]
    assert selection["evictable_selection_origins"] == [
        "immutable_source_aware_core"
    ]
    assert selection["expected_starting_selection_origin_counts"] == {
        "immutable_source_aware_core": 23,
        "salience_stratified_reservation": 3,
        "validated_alignment_swap": 1,
        "local_neighborhood_rescue": 1,
    }
    assert exp["mechanism_outcome_precedence"] == [
        "INVALID_EXECUTION",
        "PRESERVATION_REGRESSION",
        "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED",
        "BUDGET_SELECTION_COLLISION",
        "OBLIGATION_EXPOSURE_GAP",
    ]
    assert exp["completeness_safety_outcome_precedence"] == [
        "INVALID_EXECUTION",
        "UNSAFE_FALSE_COMPLETE",
        "COMPLETE_WITH_CLOSURE",
        "SAFE_INCOMPLETENESS_CONFIRMED",
    ]
