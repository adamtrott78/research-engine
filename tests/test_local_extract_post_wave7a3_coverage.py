from __future__ import annotations

from research_engine.local_extract_post_wave7a3_coverage import (
    _composition_verdict,
    _facility_location_merge,
    _quota_merge,
)


def _row(pid: str) -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": pid,
    }


def _ranking(*pids: str) -> list[dict]:
    return [_row(pid) for pid in pids]


def test_facility_location_balances_branch_weight_not_intent_count() -> None:
    branches = {
        "control": {
            "rankings": [
                {"intent_index": 0, "ranking": _ranking("c1", "shared")},
                {"intent_index": 1, "ranking": _ranking("c2", "shared")},
                {"intent_index": 2, "ranking": _ranking("c3", "shared")},
            ]
        },
        "residual": {
            "rankings": [
                {"intent_index": 0, "ranking": _ranking("r1", "shared")},
            ]
        },
    }
    selected = _facility_location_merge(branches, rrf_k=60, budget=4)
    assert selected[0]["proposition_id"] == "r1"
    assert {row["proposition_id"] for row in selected} == {
        "r1",
        "c1",
        "c2",
        "c3",
    }


def test_facility_location_prefers_collective_coverage_before_redundancy() -> None:
    branches = {
        "control": {
            "rankings": [
                {"intent_index": 0, "ranking": _ranking("a", "x")},
                {"intent_index": 1, "ranking": _ranking("b", "x")},
            ]
        },
        "residual": {
            "rankings": [
                {"intent_index": 0, "ranking": _ranking("c", "x")},
                {"intent_index": 1, "ranking": _ranking("d", "x")},
            ]
        },
    }
    selected = _facility_location_merge(branches, rrf_k=60, budget=4)
    assert {row["proposition_id"] for row in selected} == {"a", "b", "c", "d"}


def test_quota_merge_is_deterministic_and_deduplicated() -> None:
    control = _ranking(*[f"c{i}" for i in range(1, 25)])
    residual = _ranking("c1", "c2", "r1", "r2", "r3", "r4", "r5")
    merged = _quota_merge(
        control,
        residual,
        control_quota=20,
        residual_quota=4,
        budget=24,
    )
    assert len(merged) == 24
    assert len({row["proposition_id"] for row in merged}) == 24
    assert [row["proposition_id"] for row in merged[-4:]] == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]


def _candidate(
    supported: list[str],
    *,
    page_fraction: float = 1.0,
) -> dict:
    residual = sorted(set(supported) & {"e1.a", "e6.a"})
    control = sorted(
        set(supported)
        & {"e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b"}
    )
    return {
        "teacher": {
            "supported_atoms_at_24": len(supported),
            "observed_residual_recovery_count": len(residual),
            "frozen_control_retention_count": len(control),
        },
        "nine_paper": {
            "merged24_page_recall_fraction_of_ceiling": page_fraction,
        },
    }


def test_composition_confirmed_requires_nine_support_both_residuals_all_control() -> None:
    supported = [
        "e1.a",
        "e6.a",
        "e2.a",
        "e2.b",
        "e3.a",
        "e3.b",
        "e4.a",
        "e4.b",
        "e6.b",
    ]
    assert _composition_verdict(_candidate(supported))["composition_decision"] == (
        "COMPOSITION_CONFIRMED"
    )


def test_composition_partial_requires_real_gain_and_six_control_atoms() -> None:
    supported = [
        "e1.a",
        "e2.a",
        "e2.b",
        "e3.a",
        "e3.b",
        "e4.a",
        "e4.b",
        "e1.b",
    ]
    assert _composition_verdict(_candidate(supported))["composition_decision"] == (
        "COMPOSITION_PARTIAL"
    )


def test_composition_negative_when_control_is_displaced() -> None:
    supported = ["e1.a", "e6.a", "e3.a", "e4.a", "e4.b"]
    assert _composition_verdict(_candidate(supported))["composition_decision"] == (
        "COMPOSITION_NEGATIVE"
    )
