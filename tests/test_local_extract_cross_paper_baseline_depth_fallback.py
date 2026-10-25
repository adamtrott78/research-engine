from __future__ import annotations

from research_engine.local_extract_cross_paper_baseline_depth_fallback import (
    _apply_depth_fallback,
    _classify,
)


def _event(rank: int, pid: str, intent_index: int = 0) -> dict:
    return {
        "frontier_rank": rank,
        "tier": 1,
        "visit_group": "source",
        "round": 1,
        "generator_id": "QWEN35_SOURCE_AWARE_INTENTS_V1",
        "intent_index": intent_index,
        "group_kind": "facet",
        "group_value": "method",
        "bm25_rank": 1,
        "proposition_id": pid,
    }


def _selected(count: int) -> list[dict]:
    return [
        {
            "proposition_id": f"p{i}",
            "page_number": 1,
            "raw_source_form": f"text p{i}",
        }
        for i in range(count)
    ]


def _intent(ranking: list[dict]) -> dict:
    return {
        "generator_id": "QWEN35_SOURCE_AWARE_INTENTS_V1",
        "intent_index": 0,
        "question": "method question",
        "facet": "method",
        "ranking": ranking,
    }


def _candidate(pid: str, rank: int) -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": f"text {pid}",
        "oracle_rank": rank,
    }


def test_classify_complete() -> None:
    result = _classify(
        safe=True,
        identity_invariants_pass=True,
        final_complete_paper_count=9,
    )
    assert result["baseline_depth_fallback_decision"] == (
        "DEPTH_FALLBACK_COMPLETES_ALL_NINE"
    )


def test_classify_still_incomplete() -> None:
    result = _classify(
        safe=True,
        identity_invariants_pass=True,
        final_complete_paper_count=8,
    )
    assert result["baseline_depth_fallback_decision"] == (
        "DEPTH_FALLBACK_STILL_INCOMPLETE"
    )


def test_classify_identity_regression_precedes_completion() -> None:
    result = _classify(
        safe=True,
        identity_invariants_pass=False,
        final_complete_paper_count=9,
    )
    assert result["baseline_depth_fallback_decision"] == (
        "BASELINE_IDENTITY_REGRESSION"
    )


def test_classify_invalid_execution_precedes_science() -> None:
    result = _classify(
        safe=False,
        identity_invariants_pass=True,
        final_complete_paper_count=9,
    )
    assert result["baseline_depth_fallback_decision"] == "INVALID_EXECUTION"


def test_complete_baseline_is_unchanged() -> None:
    selected = _selected(28)
    events = [_event(25 + i, f"p{24 + i}", i) for i in range(4)]
    baseline = {
        "complete": True,
        "selected": selected,
        "events": events,
        "tier1_source": [],
        "tier1_author": [],
    }
    result = _apply_depth_fallback(baseline)
    assert result["fallback_applied"] is False
    assert result["selected"] == selected
    assert result["events"] == events
    assert result["complete"] is True


def test_incomplete_baseline_uses_rank4_only_after_top3() -> None:
    selected = _selected(27)
    events = [_event(25 + i, f"p{24 + i}") for i in range(3)]
    ranking = [
        _candidate("p0", 1),
        _candidate("p1", 2),
        _candidate("p2", 3),
        _candidate("new-r4", 4),
        _candidate("new-r5", 5),
    ]
    baseline = {
        "complete": False,
        "selected": selected,
        "events": events,
        "tier1_source": [_intent(ranking)],
        "tier1_author": [],
    }
    result = _apply_depth_fallback(baseline)
    assert result["fallback_applied"] is True
    assert result["complete"] is True
    assert len(result["selected"]) == 28
    assert len(result["events"]) == 4
    assert result["events"][-1]["proposition_id"] == "new-r4"
    assert result["events"][-1]["bm25_rank"] == 4
    assert result["events"][-1]["fallback_depth_extension"] is True


def test_incomplete_baseline_remains_incomplete_when_depth_exhausted() -> None:
    selected = _selected(27)
    events = [_event(25 + i, f"p{24 + i}") for i in range(3)]
    ranking = [
        _candidate(f"p{i}", i + 1)
        for i in range(12)
    ]
    baseline = {
        "complete": False,
        "selected": selected,
        "events": events,
        "tier1_source": [_intent(ranking)],
        "tier1_author": [],
    }
    result = _apply_depth_fallback(baseline)
    assert result["fallback_applied"] is True
    assert result["complete"] is False
    assert len(result["selected"]) == 27
    assert len(result["events"]) == 3
    assert result["fallback_events"] == []
