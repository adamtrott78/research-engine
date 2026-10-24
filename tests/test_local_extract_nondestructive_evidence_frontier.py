from __future__ import annotations

from research_engine.local_extract_nondestructive_evidence_frontier import (
    _append_frontier,
    _frontier_verdict,
)


def _row(pid: str, rank: int, *, page: int = 1) -> dict:
    return {
        "proposition_id": pid,
        "page_number": page,
        "fusion_rank": rank,
        "fusion_score": 1.0 / rank,
        "raw_source_form": f"source text {pid}",
    }


def test_append_frontier_never_reorders_or_replaces_core() -> None:
    core = [_row(f"c{i}", i) for i in range(1, 25)]
    residual = [
        _row("c3", 1),
        _row("r1", 2),
        _row("c10", 3),
        _row("r2", 4),
        _row("r3", 5),
        _row("r4", 6),
    ]
    selected = _append_frontier(
        core=core,
        residual_order=residual,
        budget=28,
    )
    assert [row["proposition_id"] for row in selected[:24]] == [
        row["proposition_id"] for row in core
    ]
    assert [row["proposition_id"] for row in selected[24:]] == [
        "r1", "r2", "r3", "r4"
    ]
    assert all(
        row["selection_origin"] == "immutable_source_aware_core"
        for row in selected[:24]
    )
    assert all(
        row["selection_origin"] == "append_only_author_anchored_frontier"
        for row in selected[24:]
    )


def test_append_frontier_uses_deeper_order_after_duplicates() -> None:
    core = [_row(f"c{i}", i) for i in range(1, 25)]
    residual = [_row(f"c{i}", i) for i in range(1, 25)]
    residual += [_row(f"r{i}", 24 + i) for i in range(1, 9)]
    selected = _append_frontier(
        core=core,
        residual_order=residual,
        budget=32,
    )
    assert len(selected) == 32
    assert [row["proposition_id"] for row in selected[-8:]] == [
        f"r{i}" for i in range(1, 9)
    ]


def _metrics(
    by_budget: dict[int, tuple[int, int, int]],
) -> dict[int, dict]:
    return {
        budget: {
            "teacher": {
                "supported_atom_count": support,
                "frozen_control_retention_count": retained,
                "observed_residual_recovery_count": residual,
            }
        }
        for budget, (support, retained, residual) in by_budget.items()
    }


def test_verdict_confirmed_at_small_frontier() -> None:
    metrics = _metrics({
        24: (7, 7, 0),
        26: (8, 7, 1),
        28: (9, 7, 2),
        32: (9, 7, 2),
    })
    result = _frontier_verdict(metrics, safe=True)
    assert result["frontier_decision"] == "ADDITIVE_FRONTIER_CONFIRMED"
    assert result["qualifying_budget"] == 28


def test_verdict_partial_when_only_one_residual_is_small() -> None:
    metrics = _metrics({
        24: (7, 7, 0),
        26: (8, 7, 1),
        28: (8, 7, 1),
        32: (8, 7, 1),
    })
    result = _frontier_verdict(metrics, safe=True)
    assert result["frontier_decision"] == "ADDITIVE_FRONTIER_PARTIAL"
    assert result["qualifying_budget"] == 26


def test_verdict_partial_when_both_residuals_require_budget32() -> None:
    metrics = _metrics({
        24: (7, 7, 0),
        26: (7, 7, 0),
        28: (7, 7, 0),
        32: (9, 7, 2),
    })
    result = _frontier_verdict(metrics, safe=True)
    assert result["frontier_decision"] == "ADDITIVE_FRONTIER_PARTIAL"
    assert result["qualifying_budget"] == 32


def test_verdict_negative_when_frontier_fails_or_safety_breaks() -> None:
    metrics = _metrics({
        24: (7, 7, 0),
        26: (7, 7, 0),
        28: (7, 7, 0),
        32: (8, 7, 1),
    })
    assert _frontier_verdict(metrics, safe=True)["frontier_decision"] == (
        "ADDITIVE_FRONTIER_NEGATIVE"
    )
    assert _frontier_verdict(metrics, safe=False)["frontier_decision"] == (
        "ADDITIVE_FRONTIER_NEGATIVE"
    )
