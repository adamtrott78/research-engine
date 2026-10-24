from __future__ import annotations

from typing import Any, Callable

from research_engine.local_extract_nli_certificate_validation import (
    binary_metrics,
    build_spark_pairs,
    select_threshold,
)
from research_engine.local_extract_wave1 import token_f1


PREDICATE_POS = {"VERB", "ADJ", "NOUN"}
ARGUMENT_DEPS = {
    "nsubj",
    "nsubjpass",
    "dobj",
    "dative",
    "attr",
    "oprd",
    "pobj",
    "ccomp",
    "xcomp",
    "acomp",
    "agent",
    "prep",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _subtree_text(token: Any) -> str:
    rows = [
        child
        for child in token.subtree
        if not bool(getattr(child, "is_space", False))
        and not bool(getattr(child, "is_punct", False))
    ]
    rows.sort(key=lambda child: int(child.i))
    return " ".join(str(child.text) for child in rows).strip()


def extract_predicate_argument_units(
    text: str,
    *,
    nlp: Callable[[str], Any],
    max_units_per_sentence: int = 12,
) -> list[dict[str, Any]]:
    doc = nlp(text)
    units: list[dict[str, Any]] = []

    for token in doc:
        if str(token.pos_) not in PREDICATE_POS:
            continue
        direct_arguments = [
            child
            for child in token.children
            if str(child.dep_) in ARGUMENT_DEPS
        ]
        if not (
            str(token.dep_) == "ROOT"
            or direct_arguments
        ):
            continue
        if not direct_arguments:
            continue

        negation = sorted(
            [
                _norm(getattr(child, "lemma_", child.text))
                for child in token.children
                if str(child.dep_) == "neg"
            ]
        )
        predicate_lemma = _norm(
            getattr(token, "lemma_", token.text)
        )
        predicate_surface = _norm(token.text)
        predicate_signature = " ".join(
            [predicate_lemma, *negation]
        ).strip()

        arguments = []
        for child in sorted(
            direct_arguments,
            key=lambda row: (int(row.i), str(row.dep_)),
        ):
            span = _subtree_text(child)
            if not span:
                continue
            arguments.append(
                {
                    "dependency": str(child.dep_),
                    "text": span,
                    "normalized": _norm(span),
                    "head_lemma": _norm(
                        getattr(child, "lemma_", child.text)
                    ),
                }
            )
        if not arguments:
            continue

        units.append(
            {
                "predicate_token_index": int(token.i),
                "predicate_surface": predicate_surface,
                "predicate_lemma": predicate_lemma,
                "predicate_signature": predicate_signature,
                "negation": negation,
                "arguments": arguments,
                "normalized_tuple": " | ".join(
                    [
                        predicate_signature,
                        *[
                            f"{arg['dependency']}={arg['normalized']}"
                            for arg in arguments
                        ],
                    ]
                ),
            }
        )

    units.sort(
        key=lambda row: (
            int(row["predicate_token_index"]),
            str(row["normalized_tuple"]),
        )
    )
    return units[:max_units_per_sentence]


def _argument_match_mean(
    source_args: list[dict[str, Any]],
    target_args: list[dict[str, Any]],
) -> float:
    if not source_args:
        return 0.0
    if not target_args:
        return 0.0

    pairs: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(source_args):
        for right_index, right in enumerate(target_args):
            pairs.append(
                (
                    token_f1(
                        str(left["normalized"]),
                        str(right["normalized"]),
                    ),
                    left_index,
                    right_index,
                )
            )
    pairs.sort(
        key=lambda row: (-row[0], row[1], row[2])
    )

    used_left: set[int] = set()
    used_right: set[int] = set()
    matched: dict[int, float] = {}
    for score, left_index, right_index in pairs:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matched[left_index] = float(score)

    return float(
        sum(matched.get(index, 0.0) for index in range(len(source_args)))
        / len(source_args)
    )


def unit_similarity(
    source: dict[str, Any],
    target: dict[str, Any],
) -> float:
    predicate_score = token_f1(
        str(source["predicate_signature"]),
        str(target["predicate_signature"]),
    )
    argument_score = _argument_match_mean(
        list(source["arguments"]),
        list(target["arguments"]),
    )
    return float(
        0.50 * predicate_score
        + 0.50 * argument_score
    )


def directional_proposition_coverage(
    source_units: list[dict[str, Any]],
    target_units: list[dict[str, Any]],
) -> float:
    if not source_units or not target_units:
        return 0.0
    best_scores = [
        max(
            unit_similarity(source, target)
            for target in target_units
        )
        for source in source_units
    ]
    return float(sum(best_scores) / len(best_scores))


def mutual_proposition_coverage(
    left_units: list[dict[str, Any]],
    right_units: list[dict[str, Any]],
) -> float:
    return float(
        min(
            directional_proposition_coverage(
                left_units,
                right_units,
            ),
            directional_proposition_coverage(
                right_units,
                left_units,
            ),
        )
    )


def score_spark_pairs(
    pairs: list[dict[str, Any]],
    *,
    nlp: Callable[[str], Any],
    max_units_per_sentence: int = 12,
) -> tuple[list[int], list[float], dict[str, Any]]:
    cache: dict[str, list[dict[str, Any]]] = {}

    def units(text: str) -> list[dict[str, Any]]:
        if text not in cache:
            cache[text] = extract_predicate_argument_units(
                text,
                nlp=nlp,
                max_units_per_sentence=max_units_per_sentence,
            )
        return cache[text]

    labels: list[int] = []
    scores: list[float] = []
    for row in pairs:
        labels.append(int(row["label"]))
        scores.append(
            mutual_proposition_coverage(
                units(str(row["left"])),
                units(str(row["right"])),
            )
        )

    empty_texts = sum(
        1 for value in cache.values() if not value
    )
    unit_counts = [len(value) for value in cache.values()]
    diagnostics = {
        "unique_text_count": len(cache),
        "empty_unit_text_count": empty_texts,
        "text_with_unit_fraction": (
            (len(cache) - empty_texts) / len(cache)
            if cache else 0.0
        ),
        "mean_units_per_text": (
            sum(unit_counts) / len(unit_counts)
            if unit_counts else 0.0
        ),
        "max_units_per_text": (
            max(unit_counts) if unit_counts else 0
        ),
    }
    return labels, scores, diagnostics


def _relation_recovered(
    relation: dict[str, Any],
    units: list[dict[str, Any]],
    *,
    threshold: float = 0.50,
) -> bool:
    indicator = _norm(
        (relation.get("TO_indicator") or {}).get("text")
    )
    arguments = [
        _norm(value.get("text"))
        for key, value in sorted(relation.items())
        if key.startswith("Arg")
        and isinstance(value, dict)
        and _norm(value.get("text"))
    ]
    if not indicator or not arguments:
        return False

    for unit in units:
        predicate_score = token_f1(
            indicator,
            str(unit["predicate_surface"]),
        )
        if predicate_score < threshold:
            continue

        candidates: list[tuple[float, int, int]] = []
        unit_args = list(unit["arguments"])
        for gold_index, gold_text in enumerate(arguments):
            for arg_index, arg in enumerate(unit_args):
                candidates.append(
                    (
                        token_f1(
                            gold_text,
                            str(arg["normalized"]),
                        ),
                        gold_index,
                        arg_index,
                    )
                )
        candidates.sort(
            key=lambda row: (-row[0], row[1], row[2])
        )
        used_gold: set[int] = set()
        used_args: set[int] = set()
        matched: dict[int, float] = {}
        for score, gold_index, arg_index in candidates:
            if gold_index in used_gold or arg_index in used_args:
                continue
            used_gold.add(gold_index)
            used_args.add(arg_index)
            matched[gold_index] = float(score)

        if all(
            matched.get(index, 0.0) >= threshold
            for index in range(len(arguments))
        ):
            return True

    return False


def evaluate_fobie(
    dataset: dict[str, Any],
    *,
    nlp: Callable[[str], Any],
    max_units_per_sentence: int = 12,
    relation_match_threshold: float = 0.50,
) -> dict[str, Any]:
    sentence_count = 0
    sentences_with_units = 0
    unit_count = 0
    relation_count = 0
    recovered_count = 0

    for doc_id in sorted(dataset):
        document = dict(dataset[doc_id])
        for sentence_id in sorted(
            document,
            key=lambda value: (
                int(value) if str(value).isdigit() else str(value)
            ),
        ):
            row = dict(document[sentence_id])
            sentence = str(row.get("sentence") or "")
            units = extract_predicate_argument_units(
                sentence,
                nlp=nlp,
                max_units_per_sentence=max_units_per_sentence,
            )
            sentence_count += 1
            unit_count += len(units)
            if units:
                sentences_with_units += 1

            tradeoffs = (
                row.get("annotations", {}).get("tradeoffs", {})
                or {}
            )
            for relation_id in sorted(tradeoffs):
                relation_count += 1
                if _relation_recovered(
                    dict(tradeoffs[relation_id]),
                    units,
                    threshold=relation_match_threshold,
                ):
                    recovered_count += 1

    return {
        "sentence_count": sentence_count,
        "sentence_with_at_least_one_unit_count": sentences_with_units,
        "sentence_with_at_least_one_unit_fraction": (
            sentences_with_units / sentence_count
            if sentence_count else 0.0
        ),
        "total_unit_count": unit_count,
        "mean_units_per_sentence": (
            unit_count / sentence_count
            if sentence_count else 0.0
        ),
        "tradeoff_relation_count": relation_count,
        "recovered_tradeoff_relation_count": recovered_count,
        "full_relation_recall": (
            recovered_count / relation_count
            if relation_count else 0.0
        ),
        "relation_match_threshold": relation_match_threshold,
        "max_units_per_sentence": max_units_per_sentence,
    }


def validation_verdict(
    *,
    threshold_selected: bool,
    spark_heldout: dict[str, Any],
    fobie_test: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, str]:
    if not threshold_selected:
        return {
            "predicate_argument_certificate_decision":
                "PREDICATE_ARGUMENT_CERTIFICATE_REJECTED",
            "rationale":
                "No safe SPARK calibration threshold exists.",
        }

    spark_gate = gates["spark_heldout"]
    fobie_gate = gates["fobie_test"]

    spark_pass = (
        float(spark_heldout["precision"])
        >= float(spark_gate["precision_min"])
        and float(spark_heldout["recall"])
        >= float(spark_gate["recall_min"])
        and float(spark_heldout["false_positive_rate"])
        <= float(spark_gate["false_positive_rate_max"])
    )
    fobie_pass = (
        float(fobie_test["full_relation_recall"])
        >= float(fobie_gate["full_relation_recall_min"])
        and float(
            fobie_test[
                "sentence_with_at_least_one_unit_fraction"
            ]
        )
        >= float(
            fobie_gate[
                "sentence_with_at_least_one_unit_fraction_min"
            ]
        )
        and float(fobie_test["mean_units_per_sentence"])
        <= float(fobie_gate["mean_units_per_sentence_max"])
    )

    if spark_pass and fobie_pass:
        return {
            "predicate_argument_certificate_decision":
                "PREDICATE_ARGUMENT_CERTIFICATE_VALIDATED",
            "rationale":
                "Predicate/argument evidence units pass both redundancy-safety and scientific relation-fidelity gates.",
        }

    if spark_pass != fobie_pass:
        return {
            "predicate_argument_certificate_decision":
                "PREDICATE_ARGUMENT_CERTIFICATE_PARTIAL",
            "rationale":
                "Exactly one external gate family passes; EXTRACT use remains unauthorized.",
        }

    return {
        "predicate_argument_certificate_decision":
            "PREDICATE_ARGUMENT_CERTIFICATE_REJECTED",
        "rationale":
            "The frozen predicate/argument evidence-unit representation does not pass the external validation contract.",
    }
