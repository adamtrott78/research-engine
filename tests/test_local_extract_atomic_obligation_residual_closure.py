from __future__ import annotations

from research_engine.local_extract_atomic_obligation_residual_closure import (
    _acceptance_boundary_evidence,
    _completeness_outcome,
    _eligible_witness,
    _eviction_candidate,
    _mechanism_outcome,
    _normalized_tokens,
    _runtime_primary_selection,
)


def _prop(pid: str, text: str, origin: str = "immutable_source_aware_core") -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": text,
        "selection_origin": origin,
    }


def test_normalization_and_prefix_witness_matching() -> None:
    assert _normalized_tokens("USRP-B210 @ 45 MSps") == [
        "usrp",
        "b210",
        "45",
        "msps",
    ]
    candidate = _prop(
        "p1",
        "The classifier is trained with source-domain device labels.",
    )
    assert _eligible_witness(
        candidate,
        [["train", "classifier"], ["label"], ["source", "domain"]],
    )
    assert not _eligible_witness(
        candidate,
        [["train"], ["target"], ["label"]],
    )


def test_eviction_order_prefers_fewest_appearances_then_worst_rank() -> None:
    selected = [
        _prop("a", "a"),
        _prop("b", "b"),
        _prop("c", "c", "salience_stratified_reservation"),
    ]
    stats = {
        "a": {"appearance_count": 1, "best_rank": 3},
        "b": {"appearance_count": 1, "best_rank": 11},
        "c": {"appearance_count": 0, "best_rank": None},
    }
    chosen = _eviction_candidate(
        selected=selected,
        structural_stats=stats,
        evictable_origins={"immutable_source_aware_core"},
    )
    assert chosen is not None
    assert chosen["proposition_id"] == "b"


def test_eviction_tie_uses_latest_position_before_pid() -> None:
    selected = [
        _prop("z", "z"),
        _prop("a", "a"),
    ]
    stats = {
        "z": {"appearance_count": 0, "best_rank": None},
        "a": {"appearance_count": 0, "best_rank": None},
    }
    chosen = _eviction_candidate(
        selected=selected,
        structural_stats=stats,
        evictable_origins={"immutable_source_aware_core"},
    )
    assert chosen is not None
    assert chosen["proposition_id"] == "a"


def test_runtime_selection_is_teacher_blind_and_budget_preserving() -> None:
    starting = [
        _prop(f"p{i:02d}", f"baseline {i}")
        for i in range(28)
    ]
    obligations = [
        {
            "id": "training",
            "witness_groups": [["train"], ["label"], ["source"]],
        },
        {
            "id": "testing",
            "witness_groups": [["test"], ["target"]],
        },
    ]
    rankings = {
        "training": [
            {
                "proposition_id": "new-train",
                "page_number": 2,
                "raw_source_form": (
                    "The classifier training uses source device labels."
                ),
                "bm25_rank": 1,
                "bm25_score": 1.0,
                "eligible_witness": True,
            }
        ],
        "testing": [
            {
                "proposition_id": "new-test",
                "page_number": 2,
                "raw_source_form": "The trained model is tested on target data.",
                "bm25_rank": 1,
                "bm25_score": 1.0,
                "eligible_witness": True,
            }
        ],
    }
    stats = {
        f"p{i:02d}": {"appearance_count": 1, "best_rank": i % 12 + 1}
        for i in range(28)
    }
    selection_cfg = {
        "obligation_order": ["training", "testing"],
        "evictable_selection_origins": ["immutable_source_aware_core"],
    }
    result = _runtime_primary_selection(
        starting_selected=starting,
        obligations=obligations,
        obligation_rankings=rankings,
        structural_stats=stats,
        selection_cfg=selection_cfg,
    )
    assert len(result["selected"]) == 28
    assert result["core_obligations_satisfied"] == 2
    assert result["controller_complete"] is True
    assert result["fallback_required"] is False
    assert [row["action"] for row in result["actions"]] == ["SWAP", "SWAP"]


def test_mechanism_outcome_partition() -> None:
    assert _mechanism_outcome(
        safe=False,
        preservation_regression=False,
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


def test_completeness_outcome_independent_of_mechanism_cause() -> None:
    assert _completeness_outcome(
        safe=True,
        controller_complete=False,
        core_obligations_satisfied=3,
        source_fidelity_count=3,
    ) == "SAFE_INCOMPLETENESS_CONFIRMED"
    assert _completeness_outcome(
        safe=True,
        controller_complete=True,
        core_obligations_satisfied=4,
        source_fidelity_count=3,
    ) == "UNSAFE_FALSE_COMPLETE"
    assert _completeness_outcome(
        safe=True,
        controller_complete=True,
        core_obligations_satisfied=4,
        source_fidelity_count=4,
    ) == "COMPLETE_WITH_CLOSURE"


def test_acceptance_boundary_evidence_is_not_acceptance_freeze() -> None:
    assert _acceptance_boundary_evidence(
        mechanism_outcome="ATOMIC_RESIDUAL_CLOSURE_CONFIRMED",
        completeness_outcome="COMPLETE_WITH_CLOSURE",
        preservation_regression=False,
        frozen_controls_retained=7,
    ) == "RESIDUAL_CLOSURE"
    assert _acceptance_boundary_evidence(
        mechanism_outcome="BUDGET_SELECTION_COLLISION",
        completeness_outcome="SAFE_INCOMPLETENESS_CONFIRMED",
        preservation_regression=False,
        frozen_controls_retained=7,
    ) == "SAFE_INCOMPLETENESS"
    assert _acceptance_boundary_evidence(
        mechanism_outcome="OBLIGATION_EXPOSURE_GAP",
        completeness_outcome="UNSAFE_FALSE_COMPLETE",
        preservation_regression=False,
        frozen_controls_retained=7,
    ) == "NONE"
