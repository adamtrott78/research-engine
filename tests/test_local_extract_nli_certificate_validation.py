from __future__ import annotations

from research_engine.local_extract_nli_certificate_validation import (
    average_precision,
    binary_metrics,
    build_spark_pairs,
    roc_auc,
    select_threshold,
    split_topics,
    validation_verdict,
)


def _spark() -> dict:
    return {
        "topic-a": {
            "clusters": [
                {"spans": [
                    {"docSpanText": "alpha method improves accuracy"},
                    {"docSpanText": "accuracy improves with alpha method"},
                ]},
                {"spans": [
                    {"docSpanText": "wireless limitation remains"},
                ]},
            ]
        },
        "topic-b": {
            "clusters": [
                {"spans": [
                    {"docSpanText": "beta dataset has devices"},
                    {"docSpanText": "devices occur in beta dataset"},
                ]},
                {"spans": [
                    {"docSpanText": "future work studies mobility"},
                ]},
            ]
        },
        "topic-c": {
            "clusters": [
                {"spans": [
                    {"docSpanText": "gamma result improves score"},
                    {"docSpanText": "score improves in gamma result"},
                ]},
                {"spans": [
                    {"docSpanText": "classifier uses source labels"},
                ]},
            ]
        },
        "topic-d": {
            "clusters": [
                {"spans": [
                    {"docSpanText": "delta frames form positive pairs"},
                    {"docSpanText": "positive pairs use delta frames"},
                ]},
                {"spans": [
                    {"docSpanText": "target data is tested"},
                ]},
            ]
        },
    }


def test_topic_split_is_hash_deterministic_and_disjoint() -> None:
    calibration, heldout = split_topics(_spark())
    assert len(calibration) == 2
    assert len(heldout) == 2
    assert set(calibration).isdisjoint(heldout)
    assert set(calibration) | set(heldout) == set(_spark())


def test_spark_pair_builder_uses_same_cluster_positives_and_bounded_negatives() -> None:
    rows = build_spark_pairs(_spark(), negative_per_positive=5)
    for split in ("calibration", "heldout"):
        positives = [row for row in rows[split] if row["label"] == 1]
        negatives = [row for row in rows[split] if row["label"] == 0]
        assert positives
        assert len(negatives) <= 5 * len(positives)
        assert len({row["pair_id"] for row in rows[split]}) == len(rows[split])


def test_threshold_selects_lowest_safe_grid_value() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.91, 0.71, 0.49, 0.10]
    result = select_threshold(
        labels,
        scores,
        [0.50, 0.70, 0.90],
        precision_min=0.95,
        recall_min=0.05,
    )
    assert result is not None
    assert result["threshold"] == 0.50
    assert result["precision"] == 1.0


def test_binary_metrics_and_rank_metrics() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    metrics = binary_metrics(labels, scores, 0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0


def test_validation_requires_both_external_gate_families() -> None:
    gates = {
        "spark_heldout": {
            "precision_min": 0.90,
            "recall_min": 0.05,
            "false_positive_rate_max": 0.02,
        },
        "scinli_heldout": {
            "roc_auc_min": 0.65,
            "average_precision_min": 0.40,
        },
    }
    spark = {
        "precision": 0.95,
        "recall": 0.10,
        "false_positive_rate": 0.01,
    }
    scinli = {"roc_auc": 0.80, "average_precision": 0.60}
    assert validation_verdict(
        threshold_selected=True,
        spark_heldout=spark,
        scinli=scinli,
        gates=gates,
    )["nli_certificate_decision"] == "NLI_CERTIFICATE_VALIDATED"

    partial = validation_verdict(
        threshold_selected=True,
        spark_heldout=spark,
        scinli={"roc_auc": 0.60, "average_precision": 0.30},
        gates=gates,
    )
    assert partial["nli_certificate_decision"] == "NLI_CERTIFICATE_PARTIAL"

    rejected = validation_verdict(
        threshold_selected=True,
        spark_heldout={
            "precision": 0.89,
            "recall": 0.20,
            "false_positive_rate": 0.01,
        },
        scinli=scinli,
        gates=gates,
    )
    assert rejected["nli_certificate_decision"] == "NLI_CERTIFICATE_REJECTED"
