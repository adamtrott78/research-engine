from __future__ import annotations

import itertools
from typing import Any

from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _prepare_teacher_pool,
    _rank,
)
from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    RESIDUAL_ID,
)
from research_engine.local_extract_wave7_evaluate import (
    _normalized,
    _rrf_merge,
    _teacher_leak_filter,
    _teacher_support,
)


TARGET_ATOM_ID = "e1.a"
POSITIVE_CONTROL_ATOM_ID = "e6.a"
FROZEN_DEPTHS = (1, 3, 5, 8, 12)


def _generated_teacher_paper(generated: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in generated["papers"]:
        item_id = str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        if item_id == "arxiv:2403.04036":
            rows.append(dict(row))
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one generated teacher paper, got {len(rows)}"
        )
    return rows[0]


def _atom_by_id(
    support_fixture: dict[str, Any],
    atom_id: str,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in support_fixture["atoms"]
        if str(row["id"]) == atom_id
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one atom {atom_id}, got {len(rows)}")
    return rows[0]


def _rank_generator_intents(
    *,
    propositions: list[dict[str, Any]],
    arm: dict[str, Any],
    generator_id: str,
    per_intent_depth: int,
) -> dict[str, Any]:
    accepted = [dict(row) for row in arm.get("accepted") or []]
    rankings = []
    for index, intent in enumerate(accepted):
        question = str(intent["question"])
        ranking = _rank(
            propositions,
            _bm25_scores(question, propositions),
        )[:per_intent_depth]
        rankings.append(
            {
                "generator_id": generator_id,
                "intent_index": index,
                "question": question,
                "ranking": ranking,
            }
        )
    return {
        "generator_id": generator_id,
        "raw_count": int(arm.get("raw_count") or len(accepted)),
        "accepted_count": len(accepted),
        "rankings": rankings,
    }


def _anchor_first_rank(
    *,
    anchor: str,
    expected_page: int,
    ranking: list[dict[str, Any]],
) -> int | None:
    for row in ranking:
        if int(row["page_number"]) != expected_page:
            continue
        text = _normalized(
            str(row.get("raw_source_form") or row.get("display_text") or "")
        )
        if anchor in text:
            return int(row["oracle_rank"])
    return None


def _minimum_intents_needed(
    coverage_by_intent: list[set[str]],
    required_anchors: set[str],
) -> int | None:
    if not required_anchors:
        return 0
    indices = range(len(coverage_by_intent))
    for width in range(1, len(coverage_by_intent) + 1):
        for combo in itertools.combinations(indices, width):
            covered: set[str] = set()
            for index in combo:
                covered.update(coverage_by_intent[index])
            if required_anchors <= covered:
                return width
    return None


def _exposure_for_atom(
    atom: dict[str, Any],
    rankings: list[dict[str, Any]],
    *,
    depths: tuple[int, ...],
) -> dict[str, Any]:
    anchors = [str(value) for value in atom["required_anchors"]]
    expected_page = int(atom["page_number"])
    per_intent = []

    for intent in rankings:
        anchor_rows = []
        complete_rank = 0
        complete = True
        for anchor in anchors:
            first_rank = _anchor_first_rank(
                anchor=anchor,
                expected_page=expected_page,
                ranking=intent["ranking"],
            )
            anchor_rows.append(
                {
                    "anchor": anchor,
                    "first_rank": first_rank,
                }
            )
            if first_rank is None:
                complete = False
            else:
                complete_rank = max(complete_rank, first_rank)
        per_intent.append(
            {
                "generator_id": str(intent["generator_id"]),
                "intent_index": int(intent["intent_index"]),
                "question": str(intent["question"]),
                "all_required_anchors_present": complete,
                "complete_depth": complete_rank if complete else None,
                "anchors": anchor_rows,
            }
        )

    union_first_rank = {}
    for anchor in anchors:
        candidates = [
            row["first_rank"]
            for intent in per_intent
            for row in intent["anchors"]
            if row["anchor"] == anchor and row["first_rank"] is not None
        ]
        union_first_rank[anchor] = min(candidates) if candidates else None

    union_complete = all(
        union_first_rank[anchor] is not None for anchor in anchors
    )
    minimum_union_depth = (
        max(int(union_first_rank[anchor]) for anchor in anchors)
        if union_complete
        else None
    )

    single_depths = [
        int(intent["complete_depth"])
        for intent in per_intent
        if intent["complete_depth"] is not None
    ]
    minimum_single_depth = min(single_depths) if single_depths else None

    required = set(anchors)
    intents_needed_by_depth = {}
    for depth in depths:
        coverage = []
        for intent in per_intent:
            covered = {
                str(row["anchor"])
                for row in intent["anchors"]
                if row["first_rank"] is not None
                and int(row["first_rank"]) <= depth
            }
            coverage.append(covered)
        intents_needed_by_depth[str(depth)] = _minimum_intents_needed(
            coverage,
            required,
        )

    return {
        "atom_id": str(atom["id"]),
        "expected_page": expected_page,
        "required_anchors": anchors,
        "earliest_rank_per_required_anchor": union_first_rank,
        "minimum_single_intent_complete_depth": minimum_single_depth,
        "minimum_union_complete_depth": minimum_union_depth,
        "minimum_distinct_intents_needed_by_depth": intents_needed_by_depth,
        "intents": per_intent,
    }


def _classify_exposure(
    *,
    target: dict[str, Any],
    replay_consistent: bool,
) -> dict[str, Any]:
    if not replay_consistent:
        return {
            "exposure_classification": "REPLAY_INCONSISTENT",
            "interpretation": (
                "Frozen replay did not reproduce the prior valid branch support "
                "or positive-control exposure, so no scientific interpretation "
                "is permitted."
            ),
        }

    single_depth = target["minimum_single_intent_complete_depth"]
    union_depth = target["minimum_union_complete_depth"]

    if single_depth is not None and int(single_depth) <= 3:
        return {
            "exposure_classification": "SHALLOW_SINGLE_INTENT_EXPOSURE",
            "interpretation": (
                "An existing automatic intent exposes all e1.a anchors by "
                "depth<=3; query generation is already sufficiently specific "
                "for this residual and allocation/planning is the next mechanism."
            ),
        }

    if union_depth is not None and int(union_depth) <= 3:
        return {
            "exposure_classification": "SHALLOW_DISTRIBUTED_EXPOSURE",
            "interpretation": (
                "No single shallow intent covers e1.a, but the shallow union does; "
                "the residual is primarily multi-intent evidence assembly."
            ),
        }

    if union_depth is not None and int(union_depth) <= 12:
        return {
            "exposure_classification": "DEEP_EXPOSURE",
            "interpretation": (
                "All e1.a anchors exist in the frozen automatic-intent top-12 "
                "universe but at least one is only exposed beyond depth 3; "
                "source-derived query refinement is justified before an iterative loop."
            ),
        }

    return {
        "exposure_classification": "STATIC_INTENTS_INSUFFICIENT",
        "interpretation": (
            "At least one e1.a anchor is absent from every frozen automatic-intent "
            "top-12 ranking; a fixed coverage planner alone is not sufficient."
        ),
    }


def evaluate_automatic_intent_exposure(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    support_fixture: dict[str, Any],
    generated: dict[str, Any],
    design: dict[str, Any],
    generated_intents_sha256: str,
    workdir: Any,
) -> dict[str, Any]:
    if generated["scientific_state_materialized"] or generated["production_routing_changed"]:
        raise RuntimeError("generated-intents artifact violates scientific boundary")

    if design["design_id"] != "local-extract-automatic-intent-exposure-v1":
        raise RuntimeError("unexpected frozen design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "AUTOMATIC_INTENT_EXPOSURE_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen experiment")

    depths = tuple(int(value) for value in experiment["evaluation_depths"])
    if depths != FROZEN_DEPTHS:
        raise RuntimeError("unexpected frozen evaluation depths")
    frozen = experiment["frozen_inputs"]
    if frozen["retriever"] != "BM25_DIRECT":
        raise RuntimeError("unexpected frozen retriever")
    per_intent_depth = int(frozen["per_intent_depth"])
    if per_intent_depth != 12:
        raise RuntimeError("unexpected frozen per-intent depth")

    teacher_pool = _prepare_teacher_pool(
        fixture=fixture,
        workdir=workdir,
    )
    generated_teacher = _generated_teacher_paper(generated)
    propositions = list(teacher_pool["page_pool"])

    generator_ids = (CONTROL_ID, RESIDUAL_ID)
    generator_rows = {}
    all_rankings = []
    forbidden = list(
        wave7a3_contract["teacher_anti_leak_audit"][
            "forbidden_normalized_substrings"
        ]
    )

    for generator_id in generator_ids:
        arm = dict(generated_teacher["generator_arms"][generator_id])
        ranked = _rank_generator_intents(
            propositions=propositions,
            arm=arm,
            generator_id=generator_id,
            per_intent_depth=per_intent_depth,
        )
        all_rankings.extend(ranked["rankings"])

        # Teacher leak audit is post-hoc only in this diagnostic. It is recorded
        # but does not remove or alter any frozen query or ranking.
        _, leak_rejected = _teacher_leak_filter(
            [dict(row) for row in arm.get("accepted") or []],
            forbidden=forbidden,
        )
        generator_rows[generator_id] = {
            "raw_count": ranked["raw_count"],
            "accepted_intent_count": ranked["accepted_count"],
            "teacher_leak_rejection_count": len(leak_rejected),
            "teacher_leak_rejected": leak_rejected,
            "intents": [
                {
                    "intent_index": int(row["intent_index"]),
                    "question": str(row["question"]),
                }
                for row in ranked["rankings"]
            ],
        }

    target_atom = _atom_by_id(support_fixture, TARGET_ATOM_ID)
    positive_atom = _atom_by_id(support_fixture, POSITIVE_CONTROL_ATOM_ID)
    target_exposure = _exposure_for_atom(
        target_atom,
        all_rankings,
        depths=depths,
    )
    positive_exposure = _exposure_for_atom(
        positive_atom,
        all_rankings,
        depths=depths,
    )

    author_rankings = [
        list(row["ranking"])
        for row in all_rankings
        if row["generator_id"] == RESIDUAL_ID
    ]
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    author_top24 = _rrf_merge(
        author_rankings,
        rrf_k=rrf_k,
        per_intent_depth=per_intent_depth,
        budget=24,
    )
    replay_support = _teacher_support(
        [target_atom, positive_atom],
        author_top24,
    )
    replay_supported = set(replay_support["supported_atoms"])
    author_top24_reproduces_known = {
        TARGET_ATOM_ID,
        POSITIVE_CONTROL_ATOM_ID,
    } <= replay_supported
    positive_control_exposed_top12 = (
        positive_exposure["minimum_union_complete_depth"] is not None
        and int(positive_exposure["minimum_union_complete_depth"]) <= 12
    )
    replay_consistent = bool(
        author_top24_reproduces_known
        and positive_control_exposed_top12
    )

    classification = _classify_exposure(
        target=target_exposure,
        replay_consistent=replay_consistent,
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_AUTOMATIC_INTENT_EXPOSURE_DIAGNOSTIC",
        "design_id": design["design_id"],
        "generated_intents_sha256": generated_intents_sha256,
        "teacher_item_id": str(teacher_pool["item_id"]),
        "retriever": "BM25_DIRECT",
        "evaluation_depths": list(depths),
        **classification,
        "target_atom": target_exposure,
        "positive_control_atom": positive_exposure,
        "generator_diagnostics": generator_rows,
        "replay_consistency": {
            "author_top24_supported_atoms": replay_support["supported_atoms"],
            "author_top24_reproduces_e1a_and_e6a": (
                author_top24_reproduces_known
            ),
            "e6a_union_exposed_within_top12": positive_control_exposed_top12,
            "status": "PASS" if replay_consistent else "FAIL",
        },
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "hard_safety": {
            "teacher_used_for_query_or_ranking": False,
            "teacher_used_only_for_posthoc_evaluation": True,
            "frozen_queries_unchanged": True,
            "frozen_bm25_rankings_only": True,
            "pairwise_semantic_similarity": False,
            "threshold_calibration": False,
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "semantic_control_or_ontology_output_false": True,
            "rejected_mechanism_rescue_false": True,
        },
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "rejected_mechanism_rescue": False,
    }
