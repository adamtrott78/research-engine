from __future__ import annotations

import hashlib
import heapq
from itertools import combinations
from typing import Any, Iterable


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _pair_identity(topic: str, left: str, right: str) -> str:
    a, b = _canonical_pair(left, right)
    return hashlib.sha256(
        (topic + "\0" + a + "\0" + b).encode("utf-8")
    ).hexdigest()


def split_topics(topics: Iterable[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(
        {str(topic) for topic in topics},
        key=lambda topic: (hashlib.sha256(topic.encode("utf-8")).hexdigest(), topic),
    )
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def _topic_clusters(topic_row: dict[str, Any]) -> list[list[str]]:
    clusters: list[list[str]] = []
    for cluster in topic_row.get("clusters") or []:
        seen: set[str] = set()
        texts: list[str] = []
        for span in cluster.get("spans") or []:
            text = _clean_text(span.get("docSpanText"))
            if not text or text in seen:
                continue
            seen.add(text)
            texts.append(text)
        if texts:
            clusters.append(sorted(texts))
    return clusters


def _positive_pairs_for_topic(
    topic: str,
    clusters: list[list[str]],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for cluster_index, texts in enumerate(clusters):
        for left, right in combinations(texts, 2):
            if left == right:
                continue
            identity = _pair_identity(topic, left, right)
            rows.setdefault(
                identity,
                {
                    "pair_id": identity,
                    "topic": topic,
                    "left": _canonical_pair(left, right)[0],
                    "right": _canonical_pair(left, right)[1],
                    "label": 1,
                    "cluster_index": cluster_index,
                },
            )
    return [rows[key] for key in sorted(rows)]


def _negative_pair_candidates_for_topic(
    topic: str,
    clusters: list[list[str]],
    positive_ids: set[str],
) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for left_index in range(len(clusters)):
        for right_index in range(left_index + 1, len(clusters)):
            for left in clusters[left_index]:
                for right in clusters[right_index]:
                    if left == right:
                        continue
                    identity = _pair_identity(topic, left, right)
                    if identity in positive_ids or identity in seen:
                        continue
                    seen.add(identity)
                    a, b = _canonical_pair(left, right)
                    yield {
                        "pair_id": identity,
                        "topic": topic,
                        "left": a,
                        "right": b,
                        "label": 0,
                    }


def _lowest_hash_rows(
    rows: Iterable[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        value = int(pair_id, 16)
        item = (-value, pair_id, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
            continue
        if value < -heap[0][0]:
            heapq.heapreplace(heap, item)
    return [
        item[2]
        for item in sorted(
            heap,
            key=lambda item: (-item[0], item[1]),
        )
    ]


def build_spark_pairs(
    spark: dict[str, Any],
    *,
    negative_per_positive: int = 5,
) -> dict[str, Any]:
    calibration_topics, heldout_topics = split_topics(spark)
    split_topics_map = {
        "calibration": set(calibration_topics),
        "heldout": set(heldout_topics),
    }
    output: dict[str, Any] = {
        "calibration_topics": calibration_topics,
        "heldout_topics": heldout_topics,
        "calibration": [],
        "heldout": [],
    }

    all_clusters = {
        str(topic): _topic_clusters(dict(row))
        for topic, row in spark.items()
    }

    for split_name, topic_ids in split_topics_map.items():
        positives: list[dict[str, Any]] = []
        positive_ids: set[str] = set()
        for topic in sorted(topic_ids):
            rows = _positive_pairs_for_topic(topic, all_clusters[topic])
            positives.extend(rows)
            positive_ids.update(str(row["pair_id"]) for row in rows)

        negative_limit = negative_per_positive * len(positives)

        def negative_stream() -> Iterable[dict[str, Any]]:
            for topic in sorted(topic_ids):
                yield from _negative_pair_candidates_for_topic(
                    topic,
                    all_clusters[topic],
                    positive_ids,
                )

        negatives = _lowest_hash_rows(
            negative_stream(),
            negative_limit,
        )
        output[split_name] = sorted(
            positives + negatives,
            key=lambda row: (
                str(row["pair_id"]),
                -int(row["label"]),
            ),
        )
        output[f"{split_name}_positive_count"] = len(positives)
        output[f"{split_name}_negative_count"] = len(negatives)

    return output


def binary_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores length mismatch")
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr),
    }


def select_threshold(
    labels: list[int],
    scores: list[float],
    grid: list[float],
    *,
    precision_min: float,
    recall_min: float,
) -> dict[str, Any] | None:
    for threshold in grid:
        metrics = binary_metrics(labels, scores, float(threshold))
        if (
            metrics["precision"] >= precision_min
            and metrics["recall"] >= recall_min
        ):
            return metrics
    return None


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both classes")

    ordered = sorted(
        zip(scores, labels),
        key=lambda row: row[0],
    )
    positive_rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and ordered[end][0] == ordered[index][0]
        ):
            end += 1
        avg_rank = (rank + (rank + (end - index) - 1)) / 2.0
        positive_rank_sum += avg_rank * sum(
            1 for _, label in ordered[index:end] if label == 1
        )
        rank += end - index
        index = end

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return float(auc)


def average_precision(
    labels: list[int],
    scores: list[float],
) -> float:
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        raise ValueError("average precision requires positives")

    ordered = sorted(
        zip(scores, labels),
        key=lambda row: (-row[0], -row[1]),
    )
    seen_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label != 1:
            continue
        seen_positive += 1
        precision_sum += seen_positive / rank
    return float(precision_sum / positives)


def validation_verdict(
    *,
    threshold_selected: bool,
    spark_heldout: dict[str, Any],
    scinli: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, str]:
    if not threshold_selected:
        return {
            "nli_certificate_decision": "NLI_CERTIFICATE_REJECTED",
            "rationale": "No safe SPARK calibration threshold exists.",
        }

    spark_gate = gates["spark_heldout"]
    scinli_gate = gates["scinli_heldout"]

    spark_precision_ok = (
        float(spark_heldout["precision"])
        >= float(spark_gate["precision_min"])
    )
    spark_pass = (
        spark_precision_ok
        and float(spark_heldout["recall"])
        >= float(spark_gate["recall_min"])
        and float(spark_heldout["false_positive_rate"])
        <= float(spark_gate["false_positive_rate_max"])
    )
    scinli_pass = (
        float(scinli["roc_auc"])
        >= float(scinli_gate["roc_auc_min"])
        and float(scinli["average_precision"])
        >= float(scinli_gate["average_precision_min"])
    )

    if spark_pass and scinli_pass:
        return {
            "nli_certificate_decision": "NLI_CERTIFICATE_VALIDATED",
            "rationale": (
                "Mutual entailment passes both held-out proposition-redundancy "
                "safety and scientific-text entailment discrimination gates."
            ),
        }
    if not spark_precision_ok:
        return {
            "nli_certificate_decision": "NLI_CERTIFICATE_REJECTED",
            "rationale": (
                "Held-out proposition-redundancy precision is below the "
                "predeclared safety floor."
            ),
        }
    return {
        "nli_certificate_decision": "NLI_CERTIFICATE_PARTIAL",
        "rationale": (
            "Only part of the external validation contract passes; learned "
            "certificate use inside EXTRACT remains unauthorized."
        ),
    }
