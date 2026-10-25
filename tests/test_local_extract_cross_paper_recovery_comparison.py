from __future__ import annotations

from research_engine.local_extract_cross_paper_recovery_comparison import (
    _allocation_candidates,
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


def test_verdict_complementary_precedes_advantages() -> None:
    result = _verdict(
        safe=True,
        baseline_complete=True,
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
        baseline_complete=True,
        counts=_counts(
            local_exact_removed_proposition_recovery_count=1,
            local_only_exact_recovery_count=1,
        ),
    )
    assert result["cross_paper_recovery_decision"] == "LOCAL_STRUCTURAL_ADVANTAGE"


def test_verdict_allocation_advantage() -> None:
    result = _verdict(
        safe=True,
        baseline_complete=True,
        counts=_counts(
            allocation_exact_removed_proposition_recovery_count=1,
            allocation_only_exact_recovery_count=1,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "SURVIVING_ALLOCATION_ADVANTAGE"
    )


def test_verdict_equivalent() -> None:
    result = _verdict(
        safe=True,
        baseline_complete=True,
        counts=_counts(
            local_exact_removed_proposition_recovery_count=2,
            allocation_exact_removed_proposition_recovery_count=2,
            both_exact_recovery_count=2,
        ),
    )
    assert result["cross_paper_recovery_decision"] == (
        "EQUIVALENT_STRUCTURAL_RECOVERY"
    )


def test_verdict_both_weak() -> None:
    result = _verdict(
        safe=True,
        baseline_complete=True,
        counts=_counts(neither_exact_recovery_count=36),
    )
    assert result["cross_paper_recovery_decision"] == "BOTH_STRUCTURALLY_WEAK"


def test_verdict_baseline_gate_precedes_mechanism_preference() -> None:
    result = _verdict(
        safe=True,
        baseline_complete=False,
        counts=_counts(local_only_exact_recovery_count=4),
    )
    assert result["cross_paper_recovery_decision"] == (
        "INSUFFICIENT_BASELINE_GENERALIZATION"
    )


def _candidate(pid: str, rank: int) -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": f"text {pid}",
        "oracle_rank": rank,
    }


def test_allocation_pool_includes_shallow_and_depth_candidates() -> None:
    paths = [
        {
            "generator_id": "g1",
            "intent_index": 0,
            "surviving_path_order": 0,
            "ranking": [
                _candidate("p1", 1),
                _candidate("p2", 2),
                _candidate("p3", 3),
                _candidate("p4", 4),
            ],
        },
        {
            "generator_id": "g2",
            "intent_index": 1,
            "surviving_path_order": 1,
            "ranking": [
                _candidate("p4", 1),
                _candidate("p5", 5),
            ],
        },
    ]
    rows = _allocation_candidates(
        paths=paths,
        stressed_ids={"p1"},
    )
    by_id = {row["proposition_id"]: row for row in rows}
    assert "p2" in by_id
    assert "p5" in by_id
    assert by_id["p2"]["rank_stratum"] == "SHALLOW_FRONTIER"
    assert by_id["p5"]["rank_stratum"] == "DEPTH_EXTENSION"
    assert by_id["p4"]["support_count"] == 2
    assert rows[0]["proposition_id"] == "p4"
