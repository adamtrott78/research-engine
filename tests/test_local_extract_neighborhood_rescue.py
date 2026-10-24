from __future__ import annotations

from research_engine.local_extract_neighborhood_rescue import (
    _candidate_neighbors,
    _classify,
    _score_candidates,
)


def _prop(pid: str) -> dict:
    page = int(pid.split(":")[1][1:])
    return {
        "proposition_id": pid,
        "page_number": page,
        "raw_source_form": f"text {pid}",
    }


def _support(
    *,
    controls: int = 7,
    e1a: bool = False,
    e6a: bool = False,
) -> dict:
    supported = [f"c{i}" for i in range(controls)]
    if e1a:
        supported.append("e1.a")
    if e6a:
        supported.append("e6.a")
    return {
        "supported_atoms": supported,
        "frozen_control_retention_count": controls,
    }


def test_immediate_neighbors_do_not_cross_page_or_reselect() -> None:
    propositions = [
        _prop("page:p2:s31"),
        _prop("page:p2:s32"),
        _prop("page:p2:s33"),
        _prop("page:p3:s01"),
    ]
    seeds = [{
        "frontier_rank": 28,
        "proposition_id": "page:p2:s32",
        "page_number": 2,
    }]
    candidates = _candidate_neighbors(
        seeds=seeds,
        propositions=propositions,
        stressed_ids={"page:p2:s31", "page:p2:s32"},
    )
    assert [row["proposition_id"] for row in candidates] == ["page:p2:s33"]
    assert candidates[0]["neighbor_relation"] == "successor"
    assert candidates[0]["seed_frontier_rank"] == 28


def test_candidate_ranking_uses_support_then_rank_then_seed() -> None:
    candidates = [
        {
            "proposition_id": "page:p2:s33",
            "page_number": 2,
            "raw_source_form": "a",
            "origins": [],
            "canonical_seed_proposition_id": "page:p2:s32",
            "seed_frontier_rank": 28,
            "neighbor_relation": "successor",
            "neighbor_distance": 1,
        },
        {
            "proposition_id": "page:p1:s03",
            "page_number": 1,
            "raw_source_form": "b",
            "origins": [],
            "canonical_seed_proposition_id": "page:p1:s04",
            "seed_frontier_rank": 27,
            "neighbor_relation": "predecessor",
            "neighbor_distance": 1,
        },
    ]
    rankings = [
        {
            "generator_id": "g1",
            "intent_index": 0,
            "question": "q1",
            "group": "method",
            "ranking": [
                {"proposition_id": "page:p2:s33", "oracle_rank": 10},
                {"proposition_id": "page:p1:s03", "oracle_rank": 1},
            ],
        },
        {
            "generator_id": "g2",
            "intent_index": 1,
            "question": "q2",
            "group": "contribution",
            "ranking": [
                {"proposition_id": "page:p2:s33", "oracle_rank": 2},
            ],
        },
    ]
    eligible, ineligible = _score_candidates(
        candidates=candidates,
        rankings=rankings,
    )
    assert not ineligible
    assert [row["proposition_id"] for row in eligible] == [
        "page:p2:s33",
        "page:p1:s03",
    ]
    assert eligible[0]["support_count"] == 2
    assert eligible[0]["best_rank"] == 2


def test_classification_confirmed() -> None:
    result = _classify(
        support=_support(e1a=True, e6a=True),
        eligible_neighbor_count=1,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == "LOCAL_RESCUE_CONFIRMED"


def test_classification_partial_is_only_e1a_restored_e6a_lost() -> None:
    result = _classify(
        support=_support(e1a=True, e6a=False),
        eligible_neighbor_count=1,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == "LOCAL_RESCUE_PARTIAL"


def test_classification_negative_covers_e1a_absent_e6a_present() -> None:
    result = _classify(
        support=_support(e1a=False, e6a=True),
        eligible_neighbor_count=1,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == "LOCAL_RESCUE_NEGATIVE"


def test_classification_negative_covers_both_residuals_absent() -> None:
    result = _classify(
        support=_support(e1a=False, e6a=False),
        eligible_neighbor_count=1,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == "LOCAL_RESCUE_NEGATIVE"


def test_no_eligible_neighbor_precedes_scientific_support() -> None:
    result = _classify(
        support=None,
        eligible_neighbor_count=0,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == (
        "LOCAL_RESCUE_NO_ELIGIBLE_NEIGHBOR"
    )


def test_collision_precedes_residual_outcome() -> None:
    result = _classify(
        support=_support(controls=6, e1a=True, e6a=True),
        eligible_neighbor_count=1,
        safe=True,
    )
    assert result["neighborhood_rescue_decision"] == "LOCAL_RESCUE_COLLISION"


def test_invalid_execution_is_not_scientific() -> None:
    result = _classify(
        support=_support(e1a=True, e6a=True),
        eligible_neighbor_count=1,
        safe=False,
    )
    assert result["neighborhood_rescue_decision"] == "INVALID_EXECUTION"
