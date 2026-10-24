from __future__ import annotations

from research_engine.local_extract_predicate_argument_validation import (
    directional_proposition_coverage,
    mutual_proposition_coverage,
    unit_similarity,
    validation_verdict,
    _relation_recovered,
)


def _unit(predicate: str, *arguments: str) -> dict:
    return {
        "predicate_signature": predicate,
        "predicate_surface": predicate,
        "arguments": [
            {
                "dependency": f"arg{index}",
                "text": text,
                "normalized": text,
            }
            for index, text in enumerate(arguments)
        ],
    }


def test_unit_similarity_rewards_matching_predicate_and_arguments() -> None:
    left = _unit("improve", "wireless accuracy", "target domain")
    right = _unit("improve", "target domain", "wireless accuracy")
    other = _unit("reduce", "training time")
    assert unit_similarity(left, right) == 1.0
    assert unit_similarity(left, other) < 0.5


def test_mutual_coverage_is_symmetric_and_requires_units() -> None:
    left = [_unit("improve", "wireless accuracy")]
    right = [_unit("improve", "wireless accuracy")]
    assert mutual_proposition_coverage(left, right) == 1.0
    assert mutual_proposition_coverage(right, left) == 1.0
    assert directional_proposition_coverage([], right) == 0.0


def test_relation_recovery_requires_indicator_and_all_arguments() -> None:
    relation = {
        "TO_indicator": {"text": "correlation"},
        "Arg0": {"text": "occludin expression"},
        "Arg1": {"text": "TER"},
        "Arg2": {"text": "paracellular flux"},
    }
    good = [
        _unit(
            "correlation",
            "occludin expression",
            "TER",
            "paracellular flux",
        )
    ]
    missing = [
        _unit(
            "correlation",
            "occludin expression",
            "TER",
        )
    ]
    assert _relation_recovered(relation, good) is True
    assert _relation_recovered(relation, missing) is False


def test_validation_requires_both_external_gate_families() -> None:
    gates = {
        "spark_heldout": {
            "precision_min": 0.90,
            "recall_min": 0.05,
            "false_positive_rate_max": 0.02,
        },
        "fobie_test": {
            "full_relation_recall_min": 0.50,
            "sentence_with_at_least_one_unit_fraction_min": 0.95,
            "mean_units_per_sentence_max": 8.0,
        },
    }
    spark = {
        "precision": 0.95,
        "recall": 0.10,
        "false_positive_rate": 0.01,
    }
    fobie = {
        "full_relation_recall": 0.60,
        "sentence_with_at_least_one_unit_fraction": 0.99,
        "mean_units_per_sentence": 4.0,
    }
    assert validation_verdict(
        threshold_selected=True,
        spark_heldout=spark,
        fobie_test=fobie,
        gates=gates,
    )["predicate_argument_certificate_decision"] == (
        "PREDICATE_ARGUMENT_CERTIFICATE_VALIDATED"
    )

    assert validation_verdict(
        threshold_selected=True,
        spark_heldout=spark,
        fobie_test={
            **fobie,
            "full_relation_recall": 0.40,
        },
        gates=gates,
    )["predicate_argument_certificate_decision"] == (
        "PREDICATE_ARGUMENT_CERTIFICATE_PARTIAL"
    )

    assert validation_verdict(
        threshold_selected=False,
        spark_heldout=spark,
        fobie_test=fobie,
        gates=gates,
    )["predicate_argument_certificate_decision"] == (
        "PREDICATE_ARGUMENT_CERTIFICATE_REJECTED"
    )
