from __future__ import annotations

import hashlib
from typing import Any


def pair_seed(pair_identity: str) -> int:
    digest = hashlib.sha256(pair_identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def deterministic_smatch_counts(
    test_graph: str,
    gold_graph: str,
    *,
    pair_identity: str,
    smatch_module: Any | None = None,
) -> tuple[int, int, int]:
    if smatch_module is None:
        import smatch as smatch_module

    if not test_graph or not gold_graph:
        raise ValueError("both AMR graphs are required")

    original_seed = smatch_module.random.seed
    original_iteration_num = smatch_module.iteration_num
    seed = pair_seed(pair_identity)
    smatch_module.match_triple_dict.clear()

    try:
        original_seed(seed)
        smatch_module.random.seed = lambda *args, **kwargs: None
        smatch_module.iteration_num = 5
        match_num, test_num, gold_num = smatch_module.get_amr_match(
            test_graph,
            gold_graph,
        )
        return int(match_num), int(test_num), int(gold_num)
    finally:
        smatch_module.random.seed = original_seed
        smatch_module.iteration_num = original_iteration_num
        smatch_module.match_triple_dict.clear()


def smatch_f1(
    test_graph: str | None,
    gold_graph: str | None,
    *,
    pair_identity: str,
    smatch_module: Any | None = None,
) -> float:
    if not test_graph or not gold_graph:
        return 0.0
    try:
        match_num, test_num, gold_num = deterministic_smatch_counts(
            test_graph,
            gold_graph,
            pair_identity=pair_identity,
            smatch_module=smatch_module,
        )
    except Exception:
        return 0.0

    if smatch_module is None:
        import smatch as smatch_module
    return float(
        smatch_module.compute_f(match_num, test_num, gold_num)[2]
    )


def _gold_triple_count(
    graph: str,
    *,
    amr_module: Any | None = None,
) -> int:
    if amr_module is None:
        import amr as amr_module
    parsed = amr_module.AMR.parse_AMR_line(" ".join(graph.split()))
    if parsed is None:
        raise ValueError("invalid gold AMR")
    instance, attribute, relation = parsed.get_triples()
    return len(instance) + len(attribute) + len(relation)


def score_bio_predictions(
    predictions: list[str | None],
    gold_graphs: list[str],
    *,
    example_ids: list[str],
    smatch_module: Any | None = None,
    amr_module: Any | None = None,
) -> dict[str, Any]:
    if not (
        len(predictions) == len(gold_graphs) == len(example_ids)
    ):
        raise ValueError("Bio-AMR inputs must have equal lengths")

    if smatch_module is None:
        import smatch as smatch_module

    total_match = 0
    total_test = 0
    total_gold = 0
    valid = 0

    for prediction, gold, example_id in zip(
        predictions,
        gold_graphs,
        example_ids,
    ):
        gold_count = _gold_triple_count(
            gold,
            amr_module=amr_module,
        )
        if not prediction:
            total_gold += gold_count
            continue

        try:
            match_num, test_num, gold_num = deterministic_smatch_counts(
                prediction,
                gold,
                pair_identity=f"BIO_AMR_V0_8_TEST:{example_id}",
                smatch_module=smatch_module,
            )
        except Exception:
            total_gold += gold_count
            continue

        valid += 1
        total_match += match_num
        total_test += test_num
        total_gold += gold_num

    precision, recall, f1 = smatch_module.compute_f(
        total_match,
        total_test,
        total_gold,
    )
    count = len(gold_graphs)
    return {
        "example_count": count,
        "valid_prediction_count": valid,
        "valid_prediction_fraction": (
            valid / count if count else 0.0
        ),
        "matching_triple_count": total_match,
        "predicted_triple_count": total_test,
        "gold_triple_count": total_gold,
        "corpus_precision": float(precision),
        "corpus_recall": float(recall),
        "corpus_smatch_f1": float(f1),
    }


def score_spark_pairs(
    pairs: list[dict[str, Any]],
    *,
    graph_by_text: dict[str, str | None],
    smatch_module: Any | None = None,
) -> tuple[list[int], list[float], dict[str, Any]]:
    labels: list[int] = []
    scores: list[float] = []

    for row in pairs:
        left = str(row["left"])
        right = str(row["right"])
        labels.append(int(row["label"]))
        first, second = (
            (left, right) if left <= right else (right, left)
        )
        pair_identity = (
            str(row["topic"]) + "\0" + first + "\0" + second
        )
        scores.append(
            smatch_f1(
                graph_by_text.get(left),
                graph_by_text.get(right),
                pair_identity=pair_identity,
                smatch_module=smatch_module,
            )
        )

    unique_texts = {
        str(row["left"])
        for row in pairs
    } | {
        str(row["right"])
        for row in pairs
    }
    invalid = sum(
        1 for text in unique_texts
        if not graph_by_text.get(text)
    )
    diagnostics = {
        "unique_text_count": len(unique_texts),
        "valid_graph_text_count": len(unique_texts) - invalid,
        "invalid_graph_text_count": invalid,
        "valid_graph_text_fraction": (
            (len(unique_texts) - invalid) / len(unique_texts)
            if unique_texts else 0.0
        ),
    }
    return labels, scores, diagnostics


def validation_verdict(
    *,
    bio_metrics: dict[str, Any],
    threshold_selected: bool,
    spark_heldout: dict[str, Any],
    bio_gate: dict[str, Any],
    spark_gate: dict[str, Any],
) -> dict[str, Any]:
    bio_pass = (
        float(bio_metrics["corpus_smatch_f1"])
        >= float(bio_gate["corpus_smatch_f1_min"])
        and float(bio_metrics["valid_prediction_fraction"])
        >= float(bio_gate["valid_prediction_fraction_min"])
    )

    spark_pass = (
        threshold_selected
        and float(spark_heldout["precision"])
        >= float(spark_gate["precision_min"])
        and float(spark_heldout["recall"])
        >= float(spark_gate["recall_min"])
        and float(spark_heldout["false_positive_rate"])
        <= float(spark_gate["false_positive_rate_max"])
    )

    if not threshold_selected:
        decision = "AMR_GRAPH_CERTIFICATE_REJECTED"
        rationale = "No safe SPARK calibration threshold exists."
    elif bio_pass and spark_pass:
        decision = "AMR_GRAPH_CERTIFICATE_VALIDATED"
        rationale = (
            "Frozen AMR parser fidelity and proposition-redundancy "
            "safety gates both pass."
        )
    elif bio_pass != spark_pass:
        decision = "AMR_GRAPH_CERTIFICATE_PARTIAL"
        rationale = (
            "Exactly one frozen external AMR validation gate family "
            "passes; EXTRACT use remains unauthorized."
        )
    else:
        decision = "AMR_GRAPH_CERTIFICATE_REJECTED"
        rationale = (
            "Neither frozen external AMR validation gate family passes."
        )

    return {
        "amr_graph_certificate_decision": decision,
        "rationale": rationale,
        "bio_amr_pass": bool(bio_pass),
        "spark_pass": bool(spark_pass),
    }
