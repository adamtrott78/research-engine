from __future__ import annotations

import pytest

from research_engine.local_extract_corrected_all_nine_recovery_comparison import (
    _materialize_selected,
    _verdict,
)


def _counts(**overrides: int) -> dict[str, int]:
    rows = {
        "local_exact_removed_proposition_recovery_count": 0,
        "allocation_exact_removed_proposition_recovery_count": 0,
        "both_exact_recovery_count": 0,
        "local_only_exact_recovery_count": 0,
        "allocation_only_exact_recovery_count": 0,
        "neither_exact_recovery_count": 0,
        "allocation_exact_recoveries_from_shallow_frontier_count": 0,
        "allocation_exact_recoveries_from_depth_extension_count": 0,
    }
    rows.update(overrides)
    return rows


def test_verdict_invalid_execution_precedes_science() -> None:
    result = _verdict(
        safe=False,
        counts=_counts(local_only_exact_recovery_count=2),
    )
    assert result["cross_paper_recovery_decision"] == "INVALID_EXECUTION"


def test_verdict_complementary() -> None:
    result = _verdict(
        safe=True,
        counts=_counts(
            local_exact_removed_proposition_recovery_count=2,
            allocation_exact_removed_proposition_recovery_count=2,
            local_only_exact_recovery_count=1,
            allocation_only_exact_recovery_count=1,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "COMPLEMENTARY_STRUCTURAL_RECOVERY"
    )


def test_verdict_local_advantage() -> None:
    result = _verdict(
        safe=True,
        counts=_counts(
            local_exact_removed_proposition_recovery_count=1,
            local_only_exact_recovery_count=1,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "LOCAL_STRUCTURAL_ADVANTAGE"
    )


def test_verdict_allocation_advantage() -> None:
    result = _verdict(
        safe=True,
        counts=_counts(
            allocation_exact_removed_proposition_recovery_count=1,
            allocation_only_exact_recovery_count=1,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "SURVIVING_ALLOCATION_ADVANTAGE"
    )


def test_verdict_both_weak() -> None:
    result = _verdict(
        safe=True,
        counts=_counts(neither_exact_recovery_count=36),
    )
    assert result["cross_paper_recovery_decision"] == "BOTH_STRUCTURALLY_WEAK"


def test_verdict_equivalent() -> None:
    result = _verdict(
        safe=True,
        counts=_counts(
            local_exact_removed_proposition_recovery_count=2,
            allocation_exact_removed_proposition_recovery_count=2,
            both_exact_recovery_count=2,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "EQUIVALENT_STRUCTURAL_RECOVERY"
    )


def _props(count: int = 30) -> list[dict]:
    return [
        {
            "proposition_id": f"page:p1:s{i:02d}",
            "page_number": 1,
            "raw_source_form": f"text {i}",
        }
        for i in range(count)
    ]


def _baseline_row() -> dict:
    ids = [f"page:p1:s{i:02d}" for i in range(28)]
    events = [
        {
            "frontier_rank": 25 + i,
            "tier": 1,
            "visit_group": "source",
            "round": 1,
            "generator_id": "QWEN35_SOURCE_AWARE_INTENTS_V1",
            "intent_index": i,
            "group_kind": "facet",
            "group_value": "method",
            "bm25_rank": 1 + i,
            "proposition_id": ids[24 + i],
        }
        for i in range(4)
    ]
    return {
        "final_complete": True,
        "final_selected_count": 28,
        "final_event_count": 4,
        "final_selected_proposition_ids": ids,
        "final_reservation_events": events,
    }


def test_materialize_selected_preserves_exact_order_and_events() -> None:
    row = _baseline_row()
    selected, events = _materialize_selected(
        propositions=_props(),
        baseline_row=row,
    )
    assert [x["proposition_id"] for x in selected] == (
        row["final_selected_proposition_ids"]
    )
    assert events == row["final_reservation_events"]


def test_materialize_selected_rejects_missing_proposition() -> None:
    row = _baseline_row()
    row["final_selected_proposition_ids"][-1] = "page:p9:s99"
    with pytest.raises(RuntimeError, match="not locatable"):
        _materialize_selected(
            propositions=_props(),
            baseline_row=row,
        )


def test_materialize_selected_rejects_incomplete_baseline() -> None:
    row = _baseline_row()
    row["final_complete"] = False
    with pytest.raises(RuntimeError, match="not final_complete"):
        _materialize_selected(
            propositions=_props(),
            baseline_row=row,
        )
