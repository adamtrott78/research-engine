from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from research_engine.local_extract_nli_certificate_validation import (
    binary_metrics,
    build_spark_pairs,
    select_threshold,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_bytes(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    return git_blob_sha1_bytes(path.read_bytes())


def tokenizer_manifest_sha256(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".cache/"):
            continue
        rows.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("tokenizer manifest is empty")
    canonical = "".join(
        f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), rows


def smatch_seed(pair_identity: str) -> int:
    digest = hashlib.sha256(pair_identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _clear_smatch_cache(smatch_module: Any) -> None:
    cache = getattr(smatch_module, "match_triple_dict", None)
    if cache is not None:
        cache.clear()


def deterministic_smatch_counts(
    predicted_amr: str,
    gold_amr: str,
    *,
    pair_identity: str,
    smatch_module: Any,
) -> tuple[int, int, int]:
    """Run Smatch with the frozen seed-once/internal-reseed-no-op semantics."""
    original_seed = smatch_module.random.seed
    original_seed(smatch_seed(pair_identity))
    _clear_smatch_cache(smatch_module)
    try:
        smatch_module.random.seed = lambda *args, **kwargs: None
        match_num, test_num, gold_num = smatch_module.get_amr_match(
            " ".join(predicted_amr.split()),
            " ".join(gold_amr.split()),
        )
        return int(match_num), int(test_num), int(gold_num)
    finally:
        smatch_module.random.seed = original_seed
        _clear_smatch_cache(smatch_module)


def smatch_f1(match_num: int, test_num: int, gold_num: int) -> float:
    precision = match_num / test_num if test_num else 0.0
    recall = match_num / gold_num if gold_num else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def amr_triple_count(amr_text: str, smatch_module: Any) -> int:
    parsed = smatch_module.amr.AMR.parse_AMR_line(" ".join(amr_text.split()))
    instances, attributes, relations = parsed.get_triples()
    return len(instances) + len(attributes) + len(relations)


def score_spark_graph_pairs(
    rows: list[dict[str, Any]],
    graph_by_text: dict[str, str | None],
    *,
    smatch_module: Any,
) -> tuple[list[int], list[float], int]:
    labels: list[int] = []
    scores: list[float] = []
    invalid_pair_count = 0
    for row in rows:
        labels.append(int(row["label"]))
        left = graph_by_text.get(str(row["left"]))
        right = graph_by_text.get(str(row["right"]))
        if not left or not right:
            scores.append(0.0)
            invalid_pair_count += 1
            continue
        try:
            counts = deterministic_smatch_counts(
                left,
                right,
                pair_identity=str(row["pair_id"]),
                smatch_module=smatch_module,
            )
            scores.append(smatch_f1(*counts))
        except Exception:
            scores.append(0.0)
            invalid_pair_count += 1
    return labels, scores, invalid_pair_count


def aggregate_bio_smatch(
    entries: list[dict[str, str]],
    predictions: list[str | None],
    *,
    smatch_module: Any,
) -> dict[str, Any]:
    if len(entries) != len(predictions):
        raise ValueError("Bio-AMR entries/predictions length mismatch")

    total_match = 0
    total_test = 0
    total_gold = 0
    valid = 0

    for index, (entry, prediction) in enumerate(zip(entries, predictions)):
        gold = str(entry["graph"])
        pair_identity = f"bio-amr:{entry.get('id') or index}"
        if prediction:
            try:
                match_num, test_num, gold_num = deterministic_smatch_counts(
                    prediction,
                    gold,
                    pair_identity=pair_identity,
                    smatch_module=smatch_module,
                )
                total_match += match_num
                total_test += test_num
                total_gold += gold_num
                valid += 1
                continue
            except Exception:
                pass

        # Invalid predictions contribute zero matches and remain in the corpus
        # denominator. Their test-side triple count is zero; gold triples remain.
        total_gold += amr_triple_count(gold, smatch_module)

    precision = total_match / total_test if total_test else 0.0
    recall = total_match / total_gold if total_gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "example_count": len(entries),
        "valid_prediction_count": valid,
        "valid_prediction_fraction": valid / len(entries) if entries else 0.0,
        "matching_triples": total_match,
        "test_triples": total_test,
        "gold_triples": total_gold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validation_verdict(
    *,
    bio_metrics: dict[str, Any],
    spark_threshold: dict[str, Any] | None,
    spark_heldout: dict[str, Any],
    bio_gate: dict[str, Any],
    spark_gate: dict[str, Any],
) -> dict[str, Any]:
    bio_pass = (
        float(bio_metrics["f1"]) >= float(bio_gate["corpus_smatch_f1_min"])
        and float(bio_metrics["valid_prediction_fraction"])
        >= float(bio_gate["valid_prediction_fraction_min"])
    )

    if spark_threshold is None:
        return {
            "amr_graph_certificate_decision": "AMR_GRAPH_CERTIFICATE_REJECTED",
            "bio_amr_pass": bio_pass,
            "spark_pass": False,
            "rationale": "No safe SPARK calibration threshold exists.",
        }

    spark_pass = (
        float(spark_heldout["precision"]) >= float(spark_gate["precision_min"])
        and float(spark_heldout["recall"]) >= float(spark_gate["recall_min"])
        and float(spark_heldout["false_positive_rate"])
        <= float(spark_gate["false_positive_rate_max"])
    )

    if bio_pass and spark_pass:
        decision = "AMR_GRAPH_CERTIFICATE_VALIDATED"
        rationale = "Bio-AMR parser fidelity and SPARK held-out redundancy gates both pass."
    elif bio_pass != spark_pass:
        decision = "AMR_GRAPH_CERTIFICATE_PARTIAL"
        rationale = "Exactly one frozen external AMR gate family passes."
    else:
        decision = "AMR_GRAPH_CERTIFICATE_REJECTED"
        rationale = "Neither frozen external AMR gate family passes."

    return {
        "amr_graph_certificate_decision": decision,
        "bio_amr_pass": bio_pass,
        "spark_pass": spark_pass,
        "rationale": rationale,
    }


def build_and_score_spark(
    spark: dict[str, Any],
    graph_by_text: dict[str, str | None],
    *,
    smatch_module: Any,
    threshold_grid: list[float],
) -> dict[str, Any]:
    pairs = build_spark_pairs(spark, negative_per_positive=5)
    cal_labels, cal_scores, cal_invalid = score_spark_graph_pairs(
        pairs["calibration"],
        graph_by_text,
        smatch_module=smatch_module,
    )
    held_labels, held_scores, held_invalid = score_spark_graph_pairs(
        pairs["heldout"],
        graph_by_text,
        smatch_module=smatch_module,
    )
    threshold = select_threshold(
        cal_labels,
        cal_scores,
        threshold_grid,
        precision_min=0.95,
        recall_min=0.05,
    )
    if threshold is None:
        heldout_metrics = {
            "threshold": None,
            "tp": 0,
            "fp": 0,
            "tn": sum(1 for label in held_labels if label == 0),
            "fn": sum(1 for label in held_labels if label == 1),
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
        }
    else:
        heldout_metrics = binary_metrics(
            held_labels,
            held_scores,
            float(threshold["threshold"]),
        )
    return {
        "pairs": pairs,
        "calibration_scores": cal_scores,
        "heldout_scores": held_scores,
        "calibration_invalid_pair_count": cal_invalid,
        "heldout_invalid_pair_count": held_invalid,
        "threshold": threshold,
        "heldout_metrics": heldout_metrics,
    }
