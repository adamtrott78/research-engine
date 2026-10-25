from __future__ import annotations

from research_engine.local_extract_atomic_obligation_residual_closure import (
    _acceptance_boundary_evidence,
    _choose_eviction,
    _completeness_outcome,
    _eligible_for_obligation,
    _mechanism_outcome,
)


def _row(pid: str, text: str, origin: str = "immutable_source_aware_core") -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": text,
        "selection_origin": origin,
    }


def test_token_prefix_witness_matching_is_exactly_frozen() -> None:
    obligation = {
        "witness_groups": [
            ["misclass", "failure"],
            ["investigat", "follow"],
        ]
    }
    assert _eligible_for_obligation(
        _row("p1", "A device is consistently misclassified; further investigation is planned."),
        obligation,
    )
    assert not _eligible_for_obligation(
        _row("p2", "A device is consistently misclassified."),
        obligation,
    )


def test_eviction_uses_frozen_dispensability_order() -> None:
    selected = [
        _row("p1", "one"),
        _row("p2", "two"),
        _row("p3", "protected", "salience_stratified_reservation"),
    ]
    dispensability = {
        "p1": {
            "automatic_top12_appearance_count": 1,
            "best_automatic_rank": 12,
            "original_selected_position": 1,
        },
        "p2": {
            "automatic_top12_appearance_count": 1,
            "best_automatic_rank": 12,
            "original_selected_position": 2,
        },
        "p3": {
            "automatic_top12_appearance_count": 0,
            "best_automatic_rank": 13,
            "original_selected_position": 3,
        },
    }
    chosen = _choose_eviction(
        selected=selected,
        dispensability=dispensability,
        evictable_origins={"immutable_source_aware_core"},
    )
    assert chosen is not None
    assert chosen["row"]["proposition_id"] == "p2"


def test_mechanism_precedence_is_frozen() -> None:
    assert _mechanism_outcome(
        safe=False,
        preservation_regression=True,
        primary_source_fidelity_count=4,
        sensitivity_source_fidelity_count=4,
    ) == "INVALID_EXECUTION"
    assert _mechanism_outcome(
        safe=True,
        preservation_regression=True,
        primary_source_fidelity_count=4,
        sensitivity_source_fidelity_count=4,
    ) == "PRESERVATION_REGRESSION"
    assert _mechanism_outcome(
        safe=True,
        preservation_regression=False,
        primary_source_fidelity_count=4,
        sensitivity_source_fidelity_count=4,
    ) == "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED"
    assert _mechanism_outcome(
        safe=True,
        preservation_regression=False,
        primary_source_fidelity_count=3,
        sensitivity_source_fidelity_count=4,
    ) == "BUDGET_SELECTION_COLLISION"
    assert _mechanism_outcome(
        safe=True,
        preservation_regression=False,
        primary_source_fidelity_count=3,
        sensitivity_source_fidelity_count=3,
    ) == "OBLIGATION_EXPOSURE_GAP"


def test_completeness_precedence_is_frozen() -> None:
    assert _completeness_outcome(
        safe=False,
        runtime_complete=True,
        primary_source_fidelity_count=4,
    ) == "INVALID_EXECUTION"
    assert _completeness_outcome(
        safe=True,
        runtime_complete=True,
        primary_source_fidelity_count=3,
    ) == "UNSAFE_FALSE_COMPLETE"
    assert _completeness_outcome(
        safe=True,
        runtime_complete=True,
        primary_source_fidelity_count=4,
    ) == "COMPLETE_WITH_CLOSURE"
    assert _completeness_outcome(
        safe=True,
        runtime_complete=False,
        primary_source_fidelity_count=2,
    ) == "SAFE_INCOMPLETENESS_CONFIRMED"


def test_acceptance_boundary_evidence_does_not_grant_acceptance_freeze() -> None:
    assert _acceptance_boundary_evidence(
        mechanism_outcome="ATOMIC_RESIDUAL_CLOSURE_CONFIRMED",
        completeness_outcome="COMPLETE_WITH_CLOSURE",
        prior_retention_count=9,
        control_retention_count=7,
    ) == "RESIDUAL_CLOSURE"
    assert _acceptance_boundary_evidence(
        mechanism_outcome="BUDGET_SELECTION_COLLISION",
        completeness_outcome="SAFE_INCOMPLETENESS_CONFIRMED",
        prior_retention_count=9,
        control_retention_count=7,
    ) == "SAFE_INCOMPLETENESS"
    assert _acceptance_boundary_evidence(
        mechanism_outcome="PRESERVATION_REGRESSION",
        completeness_outcome="SAFE_INCOMPLETENESS_CONFIRMED",
        prior_retention_count=8,
        control_retention_count=7,
    ) == "NONE"
