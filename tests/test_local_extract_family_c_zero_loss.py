from __future__ import annotations

from research_engine.local_extract_family_c_zero_loss import (
    _coverage_utility,
    _lexicographic_zero_loss_swap,
    _verdict,
)


def _row(pid: str) -> dict:
    return {
        "proposition_id": pid,
        "page_number": 1,
        "raw_source_form": pid,
        "fusion_rank": 1,
        "fusion_score": 1.0,
    }


def _arm(rankings: list[list[str]]) -> dict:
    return {
        "rankings": [
            {
                "intent_index": index,
                "question": f"q-{index}",
                "ranking": [_row(pid) for pid in pids],
            }
            for index, pids in enumerate(rankings)
        ]
    }


def test_coverage_utility_uses_best_rank_per_intent() -> None:
    arm = _arm([
        ["a", "b", "c"],
        ["d", "b", "e"],
    ])
    score = _coverage_utility({"b"}, arm, rrf_k=60)
    expected = ((1 / 62) + (1 / 62)) / 2
    assert abs(score - expected) < 1e-15


def test_zero_loss_swap_improves_residual_without_reducing_control() -> None:
    baseline = [_row("a"), _row("b"), _row("c")]
    control = _arm([
        ["a"],
        ["b"],
    ])
    residual = _arm([
        ["r", "c"],
    ])

    result = _lexicographic_zero_loss_swap(
        baseline=baseline,
        control=control,
        residual=residual,
        rrf_k=60,
    )

    assert result["zero_loss_constraint_pass"] is True
    assert result["final_control_utility"] >= (
        result["baseline_control_utility"] - 1e-15
    )
    assert result["final_residual_utility"] > (
        result["baseline_residual_utility"]
    )


def test_zero_loss_swap_refuses_control_loss_even_for_residual_gain() -> None:
    baseline = [_row("a"), _row("b")]
    control = _arm([
        ["a"],
        ["b"],
    ])
    residual = _arm([
        ["r"],
    ])

    result = _lexicographic_zero_loss_swap(
        baseline=baseline,
        control=control,
        residual=residual,
        rrf_k=60,
    )

    assert result["swap_count"] == 0
    assert {
        row["proposition_id"]
        for row in result["merged"]
    } == {"a", "b"}
    assert result["zero_loss_constraint_pass"] is True


def _candidate(
    support: int,
    residual: int,
    retained: int,
    *,
    page_fraction: float = 1.0,
    zero_loss: bool = True,
) -> dict:
    return {
        "teacher": {
            "supported_atoms_at_24": support,
            "observed_residual_recovery_count": residual,
            "frozen_control_retention_count": retained,
        },
        "nine_paper": {
            "merged24_page_recall_fraction_of_ceiling": page_fraction,
        },
        "algorithm": {
            "zero_loss_constraint_pass": zero_loss,
        },
    }


def test_strong_gate_requires_nine_two_residual_and_all_control() -> None:
    result = _verdict(_candidate(9, 2, 7))
    assert result["protected_augment_decision"] == (
        "PROTECTED_AUGMENT_STRONG"
    )


def test_feasible_gate_requires_eight_one_residual_and_all_control() -> None:
    result = _verdict(_candidate(8, 1, 7))
    assert result["protected_augment_decision"] == (
        "PROTECTED_AUGMENT_FEASIBLE"
    )


def test_gate_rejects_teacher_control_displacement() -> None:
    result = _verdict(_candidate(8, 2, 6))
    assert result["protected_augment_decision"] == (
        "PROTECTED_AUGMENT_NEGATIVE"
    )


def test_gate_rejects_zero_loss_contract_failure() -> None:
    result = _verdict(
        _candidate(9, 2, 7, zero_loss=False)
    )
    assert result["protected_augment_decision"] == (
        "PROTECTED_AUGMENT_NEGATIVE"
    )
