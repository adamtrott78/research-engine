from __future__ import annotations

from typing import Any

from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _prepare_teacher_pool,
    _rank,
)
from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    CONTROL_SUPPORTED_ATOMS,
    OBSERVED_RESIDUAL_ATOMS,
    RESIDUAL_ID,
)
from research_engine.local_extract_wave7_evaluate import (
    _rrf_merge,
    _teacher_leak_filter,
    _teacher_support,
)


FROZEN_BUDGETS = (24, 26, 28, 32)
CORE_BUDGET = 24
RESERVATION_DEPTH = 3
TARGET_ATOM_ID = "e1.a"


def _generated_teacher_paper(generated: dict[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in generated["papers"]
        if str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        == "arxiv:2403.04036"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one generated teacher paper, got {len(rows)}"
        )
    return rows[0]


def _rank_intents(
    *,
    propositions: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    generator_id: str,
    depth: int,
    author_roles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, intent in enumerate(accepted):
        question = str(intent["question"])
        ranking = _rank(
            propositions,
            _bm25_scores(question, propositions),
        )[:depth]
        row = {
            "generator_id": generator_id,
            "intent_index": index,
            "question": question,
            "ranking": ranking,
        }
        if generator_id == CONTROL_ID:
            row["facet"] = str(intent["facet"])
        else:
            anchor_id = str(intent["anchor_id"])
            if author_roles is None or anchor_id not in author_roles:
                raise RuntimeError(f"missing role for author anchor {anchor_id}")
            row["anchor_id"] = anchor_id
            row["anchor_role"] = str(author_roles[anchor_id])
        rows.append(row)
    return rows


def _frozen_core(
    *,
    propositions: list[dict[str, Any]],
    source_arm: dict[str, Any],
    forbidden: list[str],
    rrf_k: int,
    per_intent_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = [dict(row) for row in source_arm.get("accepted") or []]
    historical_accepted, rejected = _teacher_leak_filter(
        accepted,
        forbidden=forbidden,
    )
    rankings = [
        _rank(
            propositions,
            _bm25_scores(str(intent["question"]), propositions),
        )
        for intent in historical_accepted
    ]
    core = _rrf_merge(
        rankings,
        rrf_k=rrf_k,
        per_intent_depth=per_intent_depth,
        budget=CORE_BUDGET,
    )
    return core, rejected


def _reservation_groups(
    *,
    source_rankings: list[dict[str, Any]],
    author_rankings: list[dict[str, Any]],
    structural_strata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tiers: list[dict[str, Any]] = []
    for frozen in sorted(structural_strata, key=lambda row: int(row["tier"])):
        source_values = {
            str(value) for value in frozen["source_group"]["values"]
        }
        author_values = {
            str(value) for value in frozen["author_group"]["anchor_role"]
        }
        tiers.append(
            {
                "tier": int(frozen["tier"]),
                "source": [
                    row for row in source_rankings
                    if str(row["facet"]) in source_values
                ],
                "author": [
                    row for row in author_rankings
                    if str(row["anchor_role"]) in author_values
                ],
            }
        )
    return tiers


def _candidate_key(row: dict[str, Any]) -> str:
    return str(row["proposition_id"])


def _append_candidate(
    *,
    intent: dict[str, Any],
    selected: list[dict[str, Any]],
    seen: set[str],
    tier: int,
    round_number: int,
) -> bool:
    for candidate in intent["ranking"][:RESERVATION_DEPTH]:
        pid = _candidate_key(candidate)
        if pid in seen:
            continue
        seen.add(pid)
        row = dict(candidate)
        row["selection_origin"] = "salience_stratified_reservation"
        row["reservation_tier"] = tier
        row["reservation_round"] = round_number
        row["reservation_generator"] = str(intent["generator_id"])
        row["reservation_intent_index"] = int(intent["intent_index"])
        row["reservation_question"] = str(intent["question"])
        row["reservation_bm25_rank"] = int(candidate["oracle_rank"])
        if intent["generator_id"] == CONTROL_ID:
            row["reservation_group_kind"] = "facet"
            row["reservation_group_value"] = str(intent["facet"])
        else:
            row["reservation_group_kind"] = "anchor_role"
            row["reservation_group_value"] = str(intent["anchor_role"])
            row["reservation_anchor_id"] = str(intent["anchor_id"])
        selected.append(row)
        return True
    return False


def _reservation_stream(
    *,
    core: list[dict[str, Any]],
    tiers: list[dict[str, Any]],
    maximum_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in core:
        pid = _candidate_key(source)
        if pid in seen:
            raise RuntimeError("immutable core contains duplicate proposition IDs")
        seen.add(pid)
        row = dict(source)
        row["selection_origin"] = "immutable_source_aware_core"
        selected.append(row)

    events: list[dict[str, Any]] = []
    for tier in tiers:
        source = list(tier["source"])
        author = list(tier["author"])
        round_number = 1
        while len(selected) < maximum_budget:
            any_added = False
            width = max(len(source), len(author))
            for index in range(width):
                for group_name, group in (("source", source), ("author", author)):
                    if index >= len(group) or len(selected) >= maximum_budget:
                        continue
                    intent = group[index]
                    before = len(selected)
                    added = _append_candidate(
                        intent=intent,
                        selected=selected,
                        seen=seen,
                        tier=int(tier["tier"]),
                        round_number=round_number,
                    )
                    if added:
                        any_added = True
                        appended = selected[-1]
                        events.append(
                            {
                                "frontier_rank": len(selected),
                                "tier": int(tier["tier"]),
                                "visit_group": group_name,
                                "round": round_number,
                                "generator_id": str(intent["generator_id"]),
                                "intent_index": int(intent["intent_index"]),
                                "group_kind": str(
                                    appended["reservation_group_kind"]
                                ),
                                "group_value": str(
                                    appended["reservation_group_value"]
                                ),
                                "bm25_rank": int(
                                    appended["reservation_bm25_rank"]
                                ),
                                "proposition_id": str(
                                    appended["proposition_id"]
                                ),
                            }
                        )
                    if len(selected) != before + (1 if added else 0):
                        raise RuntimeError("reservation append accounting mismatch")
            if not any_added:
                break
            round_number += 1
        if len(selected) >= maximum_budget:
            break

    for rank, row in enumerate(selected, start=1):
        row["frontier_rank"] = rank
    return selected, events


def _generic_support(
    atoms: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = _teacher_support(atoms, selected)
    supported = [str(value) for value in raw["supported_atoms"]]
    supported_set = set(supported)
    return {
        "supported_atoms": supported,
        "supported_atom_count": len(supported),
        "support_recall": len(supported) / len(atoms) if atoms else 0.0,
        "frozen_control_atoms_retained": sorted(
            supported_set & CONTROL_SUPPORTED_ATOMS
        ),
        "frozen_control_retention_count": len(
            supported_set & CONTROL_SUPPORTED_ATOMS
        ),
        "observed_residual_atoms_recovered": sorted(
            supported_set & OBSERVED_RESIDUAL_ATOMS
        ),
        "observed_residual_recovery_count": len(
            supported_set & OBSERVED_RESIDUAL_ATOMS
        ),
        "target_e1a_supported": TARGET_ATOM_ID in supported_set,
        "atom_details": raw["atoms"],
    }


def _locatable(
    selected: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
) -> bool:
    source = {
        str(row["proposition_id"]): row
        for row in propositions
    }
    for row in selected:
        original = source.get(str(row["proposition_id"]))
        if original is None:
            return False
        if int(row["page_number"]) != int(original["page_number"]):
            return False
        if str(row["raw_source_form"]) != str(original["raw_source_form"]):
            return False
    return True


def _flagged_intent_keys(
    *,
    source_arm: dict[str, Any],
    author_arm: dict[str, Any],
    forbidden: list[str],
) -> set[tuple[str, int]]:
    flagged: set[tuple[str, int]] = set()
    for generator_id, arm in (
        (CONTROL_ID, source_arm),
        (RESIDUAL_ID, author_arm),
    ):
        accepted = [dict(row) for row in arm.get("accepted") or []]
        _, rejected = _teacher_leak_filter(accepted, forbidden=forbidden)
        rejected_questions = {str(row["question"]) for row in rejected}
        for index, row in enumerate(accepted):
            if str(row["question"]) in rejected_questions:
                flagged.add((generator_id, index))
    return flagged


def _unflagged_candidate_ids(
    rankings: list[dict[str, Any]],
    flagged: set[tuple[str, int]],
) -> set[str]:
    values: set[str] = set()
    for intent in rankings:
        key = (
            str(intent["generator_id"]),
            int(intent["intent_index"]),
        )
        if key in flagged:
            continue
        for row in intent["ranking"][:RESERVATION_DEPTH]:
            values.add(str(row["proposition_id"]))
    return values


def _sensitivity_selected(
    selected: list[dict[str, Any]],
    *,
    flagged: set[tuple[str, int]],
    unflagged_candidate_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    for row in selected:
        if row.get("selection_origin") != "salience_stratified_reservation":
            kept.append(dict(row))
            continue
        key = (
            str(row["reservation_generator"]),
            int(row["reservation_intent_index"]),
        )
        pid = str(row["proposition_id"])
        only_flagged = key in flagged and pid not in unflagged_candidate_ids
        if only_flagged:
            removed.append(pid)
        else:
            kept.append(dict(row))
    return kept, removed


def _classification(
    *,
    summaries: dict[int, dict[str, Any]],
    safe: bool,
    maximum_budget_reached: bool,
) -> dict[str, Any]:
    if not safe or not maximum_budget_reached:
        return {
            "facet_reservation_decision": "INVALID_EXECUTION",
            "qualifying_budget": None,
            "interpretation": (
                "Frozen execution identity, provenance, teacher-blind safety, "
                "or maximum-budget replay did not validate; no scientific "
                "verdict is permitted."
            ),
        }

    for budget in (24, 26, 28):
        teacher = summaries[budget]["teacher"]
        if (
            bool(teacher["target_e1a_supported"])
            and int(teacher["frozen_control_retention_count"]) == 7
        ):
            return {
                "facet_reservation_decision": "FACET_RESERVATION_CONFIRMED",
                "qualifying_budget": budget,
                "interpretation": (
                    "Frozen salience-stratified reservation acquires e1.a "
                    "within the preferred small frontier while the immutable "
                    "source-aware core retains all seven controls."
                ),
            }

    teacher32 = summaries[32]["teacher"]
    if (
        bool(teacher32["target_e1a_supported"])
        and int(teacher32["frozen_control_retention_count"]) == 7
    ):
        return {
            "facet_reservation_decision": "FACET_RESERVATION_PARTIAL",
            "qualifying_budget": 32,
            "interpretation": (
                "Frozen structural reservation acquires e1.a only at the "
                "maximum diagnostic frontier; acquisition is plausible but "
                "cost/compression remains first-order."
            ),
        }

    return {
        "facet_reservation_decision": "FACET_RESERVATION_NEGATIVE",
        "qualifying_budget": None,
        "interpretation": (
            "A valid safe replay reaches budget32 without recovering e1.a; "
            "fixed structural facet/role reservation is insufficient."
        ),
    }


def evaluate_facet_reserved_evidence_planner(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    support_fixture: dict[str, Any],
    generated: dict[str, Any],
    design: dict[str, Any],
    generated_intents_sha256: str,
    workdir: Any,
) -> dict[str, Any]:
    if (
        generated["scientific_state_materialized"]
        or generated["production_routing_changed"]
    ):
        raise RuntimeError("generated-intents artifact violates boundary")

    if design["design_id"] != "local-extract-facet-reserved-evidence-planner-v1":
        raise RuntimeError("unexpected frozen design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "SALIENCE_STRATIFIED_RESERVATION_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen experiment")

    frozen = experiment["frozen_inputs"]
    schedule = experiment["reservation_schedule"]
    if str(frozen["retriever"]) != "BM25_DIRECT":
        raise RuntimeError("unexpected retriever")
    if int(frozen["per_intent_candidate_depth"]) != RESERVATION_DEPTH:
        raise RuntimeError("unexpected reservation depth")
    budgets = tuple(int(value) for value in schedule["diagnostic_budgets"])
    if budgets != FROZEN_BUDGETS:
        raise RuntimeError("unexpected diagnostic budgets")
    if int(schedule["primary_small_ceiling"]) != 28:
        raise RuntimeError("unexpected primary ceiling")
    if int(schedule["maximum_ceiling"]) != 32:
        raise RuntimeError("unexpected maximum ceiling")
    if schedule["no_teacher_filtering_or_calibration"] is not True:
        raise RuntimeError("reservation must remain teacher-blind")
    if schedule["no_post_result_schedule_or_budget_selection"] is not True:
        raise RuntimeError("post-result selection is forbidden")

    teacher_pool = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    if str(teacher_pool["item_id"]) != str(frozen["teacher_control_item"]):
        raise RuntimeError("unexpected teacher control item")
    propositions = list(teacher_pool["page_pool"])
    generated_teacher = _generated_teacher_paper(generated)
    source_arm = dict(generated_teacher["generator_arms"][CONTROL_ID])
    author_arm = dict(generated_teacher["generator_arms"][RESIDUAL_ID])

    per_intent_depth = int(wave7a3_contract["retrieval"]["per_intent_depth"])
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    forbidden = list(
        wave7a3_contract["teacher_anti_leak_audit"][
            "forbidden_normalized_substrings"
        ]
    )

    core, historical_core_rejections = _frozen_core(
        propositions=propositions,
        source_arm=source_arm,
        forbidden=forbidden,
        rrf_k=rrf_k,
        per_intent_depth=per_intent_depth,
    )
    atoms = [dict(row) for row in support_fixture["atoms"]]
    core_support = _generic_support(atoms, core)
    core_identity_pass = bool(
        len(core) == CORE_BUDGET
        and set(core_support["supported_atoms"]) == CONTROL_SUPPORTED_ATOMS
    )

    author_roles = {
        str(row["anchor_id"]): str(row["role"])
        for row in author_arm.get("anchors") or []
    }
    source_rankings = _rank_intents(
        propositions=propositions,
        accepted=[dict(row) for row in source_arm.get("accepted") or []],
        generator_id=CONTROL_ID,
        depth=RESERVATION_DEPTH,
    )
    author_rankings = _rank_intents(
        propositions=propositions,
        accepted=[dict(row) for row in author_arm.get("accepted") or []],
        generator_id=RESIDUAL_ID,
        depth=RESERVATION_DEPTH,
        author_roles=author_roles,
    )
    tiers = _reservation_groups(
        source_rankings=source_rankings,
        author_rankings=author_rankings,
        structural_strata=list(experiment["structural_strata"]),
    )
    full_selected, events = _reservation_stream(
        core=core,
        tiers=tiers,
        maximum_budget=32,
    )

    summaries: dict[int, dict[str, Any]] = {}
    flagged = _flagged_intent_keys(
        source_arm=source_arm,
        author_arm=author_arm,
        forbidden=forbidden,
    )
    unflagged_ids = _unflagged_candidate_ids(
        source_rankings + author_rankings,
        flagged,
    )
    all_locatable = True

    for budget in budgets:
        selected = [dict(row) for row in full_selected[:budget]]
        reached = len(selected) == budget
        locatable = reached and _locatable(selected, propositions)
        all_locatable = all_locatable and locatable
        sensitivity, removed = _sensitivity_selected(
            selected,
            flagged=flagged,
            unflagged_candidate_ids=unflagged_ids,
        )
        teacher = _generic_support(atoms, selected) if reached else {}
        sensitivity_teacher = (
            _generic_support(atoms, sensitivity) if reached else {}
        )
        core_chars = sum(len(str(row["raw_source_form"])) for row in core)
        summaries[budget] = {
            "budget_reached": reached,
            "selected_proposition_count": len(selected),
            "teacher": teacher,
            "incremental_raw_source_chars_vs_top24": (
                sum(len(str(row["raw_source_form"])) for row in selected)
                - core_chars
                if reached
                else None
            ),
            "unique_source_pages": sorted(
                {int(row["page_number"]) for row in selected}
            ),
            "all_propositions_exactly_locatable": locatable,
            "flagged_origin_reservation_count": sum(
                1
                for row in selected
                if row.get("selection_origin")
                == "salience_stratified_reservation"
                and (
                    str(row["reservation_generator"]),
                    int(row["reservation_intent_index"]),
                )
                in flagged
            ),
            "flagged_only_removed_proposition_ids": removed,
            "posthoc_sensitivity_teacher": sensitivity_teacher,
            "selected_propositions": [
                {
                    "proposition_id": str(row["proposition_id"]),
                    "page_number": int(row["page_number"]),
                    "frontier_rank": int(row["frontier_rank"]),
                    "selection_origin": str(row["selection_origin"]),
                    "raw_source_form": str(row["raw_source_form"]),
                    **(
                        {
                            "reservation_tier": int(row["reservation_tier"]),
                            "reservation_round": int(row["reservation_round"]),
                            "reservation_generator": str(
                                row["reservation_generator"]
                            ),
                            "reservation_intent_index": int(
                                row["reservation_intent_index"]
                            ),
                            "reservation_group_kind": str(
                                row["reservation_group_kind"]
                            ),
                            "reservation_group_value": str(
                                row["reservation_group_value"]
                            ),
                            "reservation_bm25_rank": int(
                                row["reservation_bm25_rank"]
                            ),
                        }
                        if row["selection_origin"]
                        == "salience_stratified_reservation"
                        else {}
                    ),
                }
                for row in selected
            ],
        }

    maximum_budget_reached = len(full_selected) >= 32
    safe = bool(
        core_identity_pass
        and all_locatable
        and generated_intents_sha256
        == str(frozen["generated_intents_sha256"])
    )
    decision = _classification(
        summaries=summaries,
        safe=safe,
        maximum_budget_reached=maximum_budget_reached,
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_FACET_RESERVED_EVIDENCE_PLANNER_DIAGNOSTIC",
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "generated_intents_sha256": generated_intents_sha256,
        "reservation_selector_id": str(schedule["id"]),
        "diagnostic_budgets": list(budgets),
        **decision,
        "core": {
            "count": len(core),
            "identity_pass": core_identity_pass,
            "teacher": core_support,
            "historical_teacher_leak_rejections": historical_core_rejections,
        },
        "reservation_events": events,
        "budget_metrics": {
            str(budget): summaries[budget]
            for budget in budgets
        },
        "posthoc_teacher_audit": {
            "flagged_intent_keys": [
                {
                    "generator_id": generator,
                    "intent_index": index,
                }
                for generator, index in sorted(flagged)
            ],
            "used_for_reservation_selection": False,
        },
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "hard_safety": {
            "generated_intents_identity_pass": (
                generated_intents_sha256
                == str(frozen["generated_intents_sha256"])
            ),
            "immutable_core_identity_pass": core_identity_pass,
            "maximum_budget_reached": maximum_budget_reached,
            "all_selected_propositions_exactly_locatable": all_locatable,
            "teacher_used_for_reservation_selection": False,
            "teacher_used_only_for_posthoc_planner_evaluation": True,
            "new_query_generation_false": True,
            "new_model_or_ranker_false": True,
            "pairwise_semantic_similarity_false": True,
            "threshold_calibration_false": True,
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
