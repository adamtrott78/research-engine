from __future__ import annotations

from research_engine.local_extract_automatic_intent_exposure import (
    _classify_exposure,
    _minimum_intents_needed,
)


def test_minimum_intents_needed_distinguishes_single_and_distributed() -> None:
    required = {"a", "b"}
    assert _minimum_intents_needed([{"a", "b"}, {"a"}], required) == 1
    assert _minimum_intents_needed([{"a"}, {"b"}], required) == 2
    assert _minimum_intents_needed([{"a"}, set()], required) is None


def _target(
    *,
    single: int | None,
    union: int | None,
) -> dict:
    return {
        "minimum_single_intent_complete_depth": single,
        "minimum_union_complete_depth": union,
    }


def test_classification_shallow_single_precedes_distributed() -> None:
    result = _classify_exposure(
        target=_target(single=3, union=2),
        replay_consistent=True,
    )
    assert result["exposure_classification"] == "SHALLOW_SINGLE_INTENT_EXPOSURE"


def test_classification_shallow_distributed() -> None:
    result = _classify_exposure(
        target=_target(single=5, union=3),
        replay_consistent=True,
    )
    assert result["exposure_classification"] == "SHALLOW_DISTRIBUTED_EXPOSURE"


def test_classification_deep_exposure() -> None:
    result = _classify_exposure(
        target=_target(single=None, union=8),
        replay_consistent=True,
    )
    assert result["exposure_classification"] == "DEEP_EXPOSURE"


def test_classification_static_intents_insufficient() -> None:
    result = _classify_exposure(
        target=_target(single=None, union=None),
        replay_consistent=True,
    )
    assert result["exposure_classification"] == "STATIC_INTENTS_INSUFFICIENT"


def test_replay_inconsistent_blocks_scientific_interpretation() -> None:
    result = _classify_exposure(
        target=_target(single=1, union=1),
        replay_consistent=False,
    )
    assert result["exposure_classification"] == "REPLAY_INCONSISTENT"
