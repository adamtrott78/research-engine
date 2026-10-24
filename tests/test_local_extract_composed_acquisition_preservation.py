from __future__ import annotations

import pytest

from research_engine.local_extract_composed_acquisition_preservation import (
    _apply_exact_swap,
    _classify,
)


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


def test_classify_robust_complementarity() -> None:
    result = _classify(
        primary=_support(e1a=True, e6a=True),
        sensitivity=_support(e1a=True, e6a=True),
        safe=True,
    )
    assert result["composition_decision"] == "COMPOSED_COMPLEMENTARITY_ROBUST"


def test_classify_fragile_complementarity() -> None:
    result = _classify(
        primary=_support(e1a=True, e6a=True),
        sensitivity=_support(e1a=False, e6a=True),
        safe=True,
    )
    assert result["composition_decision"] == "COMPOSED_COMPLEMENTARITY_FRAGILE"


def test_classify_partial() -> None:
    result = _classify(
        primary=_support(e1a=True, e6a=False),
        sensitivity=_support(e1a=False, e6a=False),
        safe=True,
    )
    assert result["composition_decision"] == "COMPOSITION_PARTIAL"


def test_classify_negative_is_explicit() -> None:
    result = _classify(
        primary=_support(e1a=False, e6a=False),
        sensitivity=_support(e1a=False, e6a=False),
        safe=True,
    )
    assert result["composition_decision"] == "COMPOSITION_NEGATIVE"


def test_classify_control_collision_precedes_residuals() -> None:
    result = _classify(
        primary=_support(controls=6, e1a=True, e6a=True),
        sensitivity=_support(controls=6, e1a=True, e6a=True),
        safe=True,
    )
    assert result["composition_decision"] == "COMPOSITION_COLLISION"


def test_classify_invalid_execution_is_not_scientific() -> None:
    result = _classify(
        primary=_support(e1a=True, e6a=True),
        sensitivity=_support(e1a=True, e6a=True),
        safe=False,
    )
    assert result["composition_decision"] == "INVALID_EXECUTION"


def _row(pid: str, page: int) -> dict:
    return {
        "proposition_id": pid,
        "page_number": page,
        "raw_source_form": f"text {pid}",
        "selection_origin": "fixture",
    }


def test_exact_swap_requires_frozen_preconditions() -> None:
    selected = [_row(f"x{i}", 1) for i in range(25)]
    selected.extend([
        _row("page:p4:s33", 4),
        _row("page:p4:s28", 4),
        _row("x27", 2),
    ])
    propositions = [dict(row) for row in selected]
    propositions.append(_row("page:p6:s13", 6))
    swap = {
        "alignment_score": 0.4444444444444444,
        "threshold": 0.4,
    }

    composed, receipt = _apply_exact_swap(
        selected=selected,
        propositions=propositions,
        swap=swap,
    )
    ids = [row["proposition_id"] for row in composed]
    assert len(composed) == 28
    assert "page:p4:s33" not in ids
    assert "page:p4:s28" in ids
    assert "page:p6:s13" in ids
    assert receipt["removed_proposition_id"] == "page:p4:s33"
    assert receipt["added_proposition_id"] == "page:p6:s13"


def test_exact_swap_fails_closed_without_certifier() -> None:
    selected = [_row(f"x{i}", 1) for i in range(27)]
    selected.append(_row("page:p4:s33", 4))
    propositions = [dict(row) for row in selected]
    propositions.append(_row("page:p6:s13", 6))
    swap = {
        "alignment_score": 0.4444444444444444,
        "threshold": 0.4,
    }
    with pytest.raises(RuntimeError, match="certifier"):
        _apply_exact_swap(
            selected=selected,
            propositions=propositions,
            swap=swap,
        )
