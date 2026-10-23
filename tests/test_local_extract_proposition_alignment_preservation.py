from __future__ import annotations

from research_engine.local_extract_proposition_alignment_preservation import (
    _alignment_certified_swap,
    _best_same_page_certificate,
    _verdict,
)


def _row(
    pid: str,
    text: str,
    *,
    page: int,
    rank: int,
    score: float = 1.0,
) -> dict:
    return {
        "proposition_id": pid,
        "raw_source_form": text,
        "display_text": text,
        "page_number": page,
        "fusion_rank": rank,
        "fusion_score": score,
    }


def test_certificate_requires_same_page_high_overlap() -> None:
    rows = {
        "a": _row("a", "contrastive learning improves wireless accuracy", page=4, rank=1),
        "b": _row("b", "contrastive learning improves wireless accuracy substantially", page=4, rank=2),
        "c": _row("c", "contrastive learning improves wireless accuracy substantially", page=5, rank=3),
    }
    cert = _best_same_page_certificate(
        removed_pid="a",
        retained_baseline_ids={"b", "c"},
        baseline_rows=rows,
    )
    assert cert["certifier_proposition_id"] == "b"
    assert cert["page_number"] == 4
    assert cert["alignment_score"] >= 0.40


def test_dynamic_recompute_prevents_removing_mutual_certifiers() -> None:
    baseline = [
        _row("a", "alpha beta gamma delta", page=1, rank=1),
        _row("b", "alpha beta gamma delta extra", page=1, rank=24),
        _row("c", "unique method details remain protected", page=2, rank=2),
    ]
    residual = [
        _row("r1", "new residual limitation", page=3, rank=1),
        _row("r2", "second residual limitation", page=3, rank=2),
    ]
    result = _alignment_certified_swap(
        baseline=baseline,
        residual_rrf=residual,
        alignment_threshold=0.40,
    )
    assert result["swap_count"] == 1
    assert result["swaps"][0]["removed_proposition_id"] == "b"
    assert result["swaps"][0]["added_proposition_id"] == "r1"
    assert result["certificate_pass"] is True
    assert result["page_preservation_pass"] is True


def test_residual_order_and_worst_baseline_rank_are_deterministic() -> None:
    baseline = [
        _row("a", "shared proposition evidence alpha", page=1, rank=3),
        _row("b", "shared proposition evidence alpha detail", page=1, rank=23),
        _row("c", "another independent evidence unit", page=2, rank=24),
    ]
    residual = [
        _row("r-high", "highest residual", page=3, rank=1),
        _row("r-low", "lower residual", page=3, rank=2),
    ]
    result = _alignment_certified_swap(
        baseline=baseline,
        residual_rrf=residual,
        alignment_threshold=0.40,
    )
    assert result["swaps"][0]["removed_proposition_id"] == "b"
    assert result["swaps"][0]["added_proposition_id"] == "r-high"


def _candidate(
    supported: list[str],
    *,
    page_fraction: float = 1.0,
    safe: bool = True,
) -> dict:
    residual = set(supported) & {"e1.a", "e6.a"}
    control = set(supported) & {
        "e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b"
    }
    return {
        "teacher": {
            "supported_atoms_at_24": len(supported),
            "observed_residual_recovery_count": len(residual),
            "frozen_control_retention_count": len(control),
        },
        "nine_paper": {
            "merged24_page_recall_fraction_of_ceiling": page_fraction,
        },
        "algorithm": {
            "all_papers_certificate_pass": safe,
            "all_papers_page_preserved": safe,
        },
    }


def test_alignment_feasible_requires_all_control_atoms() -> None:
    supported = [
        "e1.a",
        "e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b",
    ]
    assert _verdict(_candidate(supported))["alignment_augment_decision"] == (
        "ALIGNMENT_AUGMENT_FEASIBLE"
    )


def test_alignment_strong_requires_both_residuals_and_nine_total() -> None:
    supported = [
        "e1.a", "e6.a",
        "e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b",
    ]
    assert _verdict(_candidate(supported))["alignment_augment_decision"] == (
        "ALIGNMENT_AUGMENT_STRONG"
    )


def test_alignment_negative_if_control_atom_is_lost() -> None:
    supported = [
        "e1.a",
        "e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b",
        "e1.b",
    ]
    assert _verdict(_candidate(supported))["alignment_augment_decision"] == (
        "ALIGNMENT_AUGMENT_NEGATIVE"
    )
