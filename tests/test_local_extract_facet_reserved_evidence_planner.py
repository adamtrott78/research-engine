from __future__ import annotations

from research_engine.local_extract_facet_reserved_evidence_planner import (
    _classification,
    _reservation_stream,
)
from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    RESIDUAL_ID,
)


def _candidate(pid: str, rank: int) -> dict:
    return {
        "proposition_id": pid,
        "page_number": 2,
        "raw_source_form": f"text {pid}",
        "oracle_rank": rank,
    }


def _intent(
    generator: str,
    index: int,
    values: list[str],
    *,
    facet: str | None = None,
    role: str | None = None,
) -> dict:
    row = {
        "generator_id": generator,
        "intent_index": index,
        "question": f"q{index}",
        "ranking": [
            _candidate(pid, rank)
            for rank, pid in enumerate(values, start=1)
        ],
    }
    if facet is not None:
        row["facet"] = facet
    if role is not None:
        row["anchor_role"] = role
        row["anchor_id"] = f"A{index:02d}"
    return row


def _core() -> list[dict]:
    return [
        {
            "proposition_id": f"c{i}",
            "page_number": 2,
            "raw_source_form": f"core {i}",
        }
        for i in range(1, 25)
    ]


def test_reservation_stream_alternates_source_then_author() -> None:
    tiers = [{
        "tier": 1,
        "source": [
            _intent(CONTROL_ID, 0, ["s0a", "s0b", "s0c"], facet="method"),
            _intent(CONTROL_ID, 1, ["s1a", "s1b", "s1c"], facet="method"),
        ],
        "author": [
            _intent(RESIDUAL_ID, 2, ["a2a", "a2b", "a2c"], role="contribution"),
            _intent(RESIDUAL_ID, 3, ["a3a", "a3b", "a3c"], role="contribution"),
        ],
    }]
    selected, events = _reservation_stream(
        core=_core(),
        tiers=tiers,
        maximum_budget=28,
    )
    assert [row["proposition_id"] for row in selected[24:28]] == [
        "s0a", "a2a", "s1a", "a3a"
    ]
    assert [row["visit_group"] for row in events[:4]] == [
        "source", "author", "source", "author"
    ]


def test_reservation_stream_revisits_only_after_each_group_once() -> None:
    tiers = [{
        "tier": 1,
        "source": [
            _intent(CONTROL_ID, 0, ["s0a", "s0b", "s0c"], facet="method"),
            _intent(CONTROL_ID, 1, ["s1a", "s1b", "s1c"], facet="method"),
        ],
        "author": [],
    }]
    selected, events = _reservation_stream(
        core=_core(),
        tiers=tiers,
        maximum_budget=28,
    )
    assert [row["proposition_id"] for row in selected[24:28]] == [
        "s0a", "s1a", "s0b", "s1b"
    ]
    assert [row["round"] for row in events[:4]] == [1, 1, 2, 2]


def _summary(e1a: bool, controls: int = 7) -> dict:
    return {
        "teacher": {
            "target_e1a_supported": e1a,
            "frozen_control_retention_count": controls,
        }
    }


def test_classification_confirmed_by_budget28() -> None:
    summaries = {
        24: _summary(False),
        26: _summary(False),
        28: _summary(True),
        32: _summary(True),
    }
    result = _classification(
        summaries=summaries,
        safe=True,
        maximum_budget_reached=True,
    )
    assert result["facet_reservation_decision"] == "FACET_RESERVATION_CONFIRMED"
    assert result["qualifying_budget"] == 28


def test_classification_partial_only_at_budget32() -> None:
    summaries = {
        24: _summary(False),
        26: _summary(False),
        28: _summary(False),
        32: _summary(True),
    }
    result = _classification(
        summaries=summaries,
        safe=True,
        maximum_budget_reached=True,
    )
    assert result["facet_reservation_decision"] == "FACET_RESERVATION_PARTIAL"
    assert result["qualifying_budget"] == 32


def test_classification_negative_requires_valid_safe_budget32() -> None:
    summaries = {
        24: _summary(False),
        26: _summary(False),
        28: _summary(False),
        32: _summary(False),
    }
    result = _classification(
        summaries=summaries,
        safe=True,
        maximum_budget_reached=True,
    )
    assert result["facet_reservation_decision"] == "FACET_RESERVATION_NEGATIVE"


def test_invalid_execution_never_becomes_scientific_negative() -> None:
    summaries = {
        24: _summary(False),
        26: _summary(False),
        28: _summary(False),
        32: _summary(False),
    }
    assert _classification(
        summaries=summaries,
        safe=False,
        maximum_budget_reached=True,
    )["facet_reservation_decision"] == "INVALID_EXECUTION"
    assert _classification(
        summaries=summaries,
        safe=True,
        maximum_budget_reached=False,
    )["facet_reservation_decision"] == "INVALID_EXECUTION"
