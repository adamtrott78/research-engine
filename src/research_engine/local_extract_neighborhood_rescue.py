from __future__ import annotations

import re
from typing import Any

from research_engine.local_extract_facet_reserved_evidence_planner import (
    _generic_support,
    _locatable,
)
from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _prepare_teacher_pool,
    _rank,
)
from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    RESIDUAL_ID,
)


TARGET_E1A = "e1.a"
TARGET_E6A = "e6.a"
FROZEN_CONTROL_COUNT = 7
FIXED_BUDGET = 28
STRESS_REMOVE = "page:p2:s33"
SURVIVING_DEPTH = 12
_PID = re.compile(r"^page:p(?P<page>\d+):s(?P<seq>\d+)$")


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


def _pid_parts(pid: str) -> tuple[int, int]:
    match = _PID.fullmatch(pid)
    if match is None:
        raise RuntimeError(f"unexpected proposition id {pid}")
    return int(match.group("page")), int(match.group("seq"))


def _validate_and_stress_composed(
    *,
    composed_report: dict[str, Any],
    design: dict[str, Any],
    composed_report_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if design["design_id"] != "local-extract-neighborhood-rescue-v1":
        raise RuntimeError("unexpected frozen neighborhood-rescue design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "ADJACENT_PROPOSITION_RESCUE_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen neighborhood-rescue experiment")
    if int(experiment["frozen_inputs"]["fixed_budget"]) != FIXED_BUDGET:
        raise RuntimeError("unexpected fixed budget")
    if int(experiment["frozen_inputs"]["surviving_intent_candidate_depth"]) != SURVIVING_DEPTH:
        raise RuntimeError("unexpected surviving-intent depth")

    expected_sha = str(
        experiment["frozen_inputs"]["composed_result_report_sha256"]
    )
    identity_pass = composed_report_sha256 == expected_sha
    if composed_report["composition_decision"] != "COMPOSED_COMPLEMENTARITY_FRAGILE":
        raise RuntimeError("unexpected composed scientific decision")
    if int(composed_report["fixed_budget"]) != FIXED_BUDGET:
        raise RuntimeError("unexpected composed fixed budget")

    selected = [dict(row) for row in composed_report["selected_propositions"]]
    if len(selected) != FIXED_BUDGET:
        raise RuntimeError("composed report does not contain 28 propositions")
    ids = [str(row["proposition_id"]) for row in selected]
    if len(set(ids)) != FIXED_BUDGET:
        raise RuntimeError("composed report contains duplicate proposition IDs")
    if ids.count(STRESS_REMOVE) != 1:
        raise RuntimeError("expected exact fragile proposition is missing or duplicated")

    removed = [
        str(value)
        for value in composed_report["sensitivity_removed_proposition_ids"]
    ]
    if removed != [STRESS_REMOVE]:
        raise RuntimeError("unexpected frozen sensitivity removal")
    fragile_row = next(
        row for row in selected
        if str(row["proposition_id"]) == STRESS_REMOVE
    )
    if str(fragile_row.get("selection_origin") or "") != "salience_stratified_reservation":
        raise RuntimeError("fragile proposition is not a salience reservation")

    sensitivity = composed_report["posthoc_flagged_origin_sensitivity_teacher"]
    if sensitivity["target_e1a_supported"] is not False:
        raise RuntimeError("frozen sensitivity unexpectedly supports e1.a")
    if TARGET_E6A not in set(sensitivity["supported_atoms"]):
        raise RuntimeError("frozen sensitivity unexpectedly loses e6.a")
    if int(sensitivity["frozen_control_retention_count"]) != FROZEN_CONTROL_COUNT:
        raise RuntimeError("frozen sensitivity does not preserve the seven controls")

    stressed = [
        dict(row)
        for row in selected
        if str(row["proposition_id"]) != STRESS_REMOVE
    ]
    if len(stressed) != FIXED_BUDGET - 1:
        raise RuntimeError("stress replay did not yield exactly 27 propositions")

    return selected, stressed, {
        "composed_report_identity_pass": identity_pass,
        "stress_removed_proposition_id": STRESS_REMOVE,
        "stressed_selected_count": len(stressed),
    }


def _reservation_seeds(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seeds = []
    for index, row in enumerate(selected, start=1):
        pid = str(row["proposition_id"])
        if pid == STRESS_REMOVE:
            continue
        if str(row.get("selection_origin") or "") != "salience_stratified_reservation":
            continue
        seeds.append(
            {
                "frontier_rank": index,
                "proposition_id": pid,
                "page_number": int(row["page_number"]),
            }
        )
    return seeds


def _candidate_neighbors(
    *,
    seeds: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
    stressed_ids: set[str],
) -> list[dict[str, Any]]:
    source = {
        str(row["proposition_id"]): dict(row)
        for row in propositions
    }
    by_page_seq: dict[tuple[int, int], str] = {}
    for pid, row in source.items():
        page, seq = _pid_parts(pid)
        if page != int(row["page_number"]):
            raise RuntimeError(f"proposition/page mismatch for {pid}")
        by_page_seq[(page, seq)] = pid

    candidates: dict[str, dict[str, Any]] = {}
    relation_order = {"predecessor": 0, "successor": 1}

    for seed in seeds:
        page, seq = _pid_parts(str(seed["proposition_id"]))
        if page != int(seed["page_number"]):
            raise RuntimeError("seed page mismatch")
        for relation, neighbor_seq in (
            ("predecessor", seq - 1),
            ("successor", seq + 1),
        ):
            if neighbor_seq < 0:
                continue
            pid = by_page_seq.get((page, neighbor_seq))
            if pid is None or pid in stressed_ids:
                continue
            origin = {
                "seed_proposition_id": str(seed["proposition_id"]),
                "seed_frontier_rank": int(seed["frontier_rank"]),
                "relation": relation,
                "neighbor_distance": 1,
            }
            if pid not in candidates:
                candidates[pid] = {
                    "proposition_id": pid,
                    "page_number": page,
                    "raw_source_form": str(source[pid]["raw_source_form"]),
                    "origins": [origin],
                }
            else:
                candidates[pid]["origins"].append(origin)

    rows = []
    for row in candidates.values():
        canonical = min(
            row["origins"],
            key=lambda value: (
                int(value["seed_frontier_rank"]),
                relation_order[str(value["relation"])],
                str(value["seed_proposition_id"]),
            ),
        )
        row["canonical_seed_proposition_id"] = str(
            canonical["seed_proposition_id"]
        )
        row["seed_frontier_rank"] = int(canonical["seed_frontier_rank"])
        row["neighbor_relation"] = str(canonical["relation"])
        row["neighbor_distance"] = 1
        rows.append(row)
    return rows


def _surviving_rankings(
    *,
    propositions: list[dict[str, Any]],
    generated_teacher: dict[str, Any],
    intent_specs: list[dict[str, Any]],
    depth: int,
) -> list[dict[str, Any]]:
    rows = []
    for spec in intent_specs:
        generator_id = str(spec["generator_id"])
        index = int(spec["intent_index"])
        if generator_id not in (CONTROL_ID, RESIDUAL_ID):
            raise RuntimeError(f"unexpected surviving generator {generator_id}")
        arm = dict(generated_teacher["generator_arms"][generator_id])
        accepted = [dict(value) for value in arm.get("accepted") or []]
        if index < 0 or index >= len(accepted):
            raise RuntimeError(
                f"surviving intent index out of range: {generator_id}:{index}"
            )
        question = str(accepted[index]["question"])
        ranking = _rank(
            propositions,
            _bm25_scores(question, propositions),
        )[:depth]
        rows.append(
            {
                "generator_id": generator_id,
                "intent_index": index,
                "question": question,
                "group": str(spec["group"]),
                "ranking": ranking,
            }
        )
    return rows


def _score_candidates(
    *,
    candidates: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relation_order = {"predecessor": 0, "successor": 1}
    eligible = []
    ineligible = []

    for candidate in candidates:
        pid = str(candidate["proposition_id"])
        hits = []
        for path in rankings:
            matched = next(
                (
                    row
                    for row in path["ranking"]
                    if str(row["proposition_id"]) == pid
                ),
                None,
            )
            if matched is None:
                continue
            hits.append(
                {
                    "generator_id": str(path["generator_id"]),
                    "intent_index": int(path["intent_index"]),
                    "question": str(path["question"]),
                    "group": str(path["group"]),
                    "bm25_rank": int(matched["oracle_rank"]),
                }
            )

        row = dict(candidate)
        row["surviving_path_hits"] = hits
        row["support_count"] = len(hits)
        row["best_rank"] = (
            min(int(hit["bm25_rank"]) for hit in hits)
            if hits
            else None
        )
        if not hits:
            ineligible.append(row)
            continue
        eligible.append(row)

    eligible.sort(
        key=lambda row: (
            -int(row["support_count"]),
            int(row["best_rank"]),
            int(row["seed_frontier_rank"]),
            int(row["neighbor_distance"]),
            relation_order[str(row["neighbor_relation"])],
            str(row["proposition_id"]),
        )
    )
    ineligible.sort(key=lambda row: str(row["proposition_id"]))
    return eligible, ineligible


def _classify(
    *,
    support: dict[str, Any] | None,
    eligible_neighbor_count: int,
    safe: bool,
) -> dict[str, Any]:
    if not safe:
        return {
            "neighborhood_rescue_decision": "INVALID_EXECUTION",
            "interpretation": (
                "Exact input, stress replay, proposition-neighborhood identity, "
                "fixed-budget, provenance, or teacher-boundary safety failed."
            ),
        }

    if eligible_neighbor_count == 0:
        return {
            "neighborhood_rescue_decision": "LOCAL_RESCUE_NO_ELIGIBLE_NEIGHBOR",
            "interpretation": (
                "No immediate neighbor of a surviving reservation seed is present "
                "in any surviving frozen tier1 top12 path."
            ),
        }

    if support is None:
        raise RuntimeError("eligible rescue candidate exists without support report")

    controls = int(support["frozen_control_retention_count"])
    if controls < FROZEN_CONTROL_COUNT:
        return {
            "neighborhood_rescue_decision": "LOCAL_RESCUE_COLLISION",
            "interpretation": (
                "A valid one-slot local rescue loses at least one frozen control atom."
            ),
        }

    supported = set(support["supported_atoms"])
    e1a = TARGET_E1A in supported
    e6a = TARGET_E6A in supported

    if e1a and e6a:
        return {
            "neighborhood_rescue_decision": "LOCAL_RESCUE_CONFIRMED",
            "interpretation": (
                "Immediate teacher-blind source-neighborhood rescue restores e1.a "
                "while retaining e6.a and all seven controls at fixed budget28."
            ),
        }

    if e1a and not e6a:
        return {
            "neighborhood_rescue_decision": "LOCAL_RESCUE_PARTIAL",
            "interpretation": (
                "Local rescue restores e1.a but interferes with the independently "
                "validated e6.a preservation outcome."
            ),
        }

    return {
        "neighborhood_rescue_decision": "LOCAL_RESCUE_NEGATIVE",
        "interpretation": (
            "The valid one-slot immediate-neighbor rescue does not restore e1.a; "
            "local adjacency is insufficient for the known fragile acquisition path."
        ),
    }


def evaluate_neighborhood_rescue(
    *,
    fixture: dict[str, Any],
    support_fixture: dict[str, Any],
    design: dict[str, Any],
    composed_report: dict[str, Any],
    composed_report_sha256: str,
    generated: dict[str, Any],
    generated_intents_sha256: str,
    workdir: Any,
) -> dict[str, Any]:
    experiment = design["selected_discriminating_experiment"]
    frozen = experiment["frozen_inputs"]
    operator = experiment["stress_and_rescue_operator"]

    if generated["scientific_state_materialized"] or generated["production_routing_changed"]:
        raise RuntimeError("generated-intents artifact violates boundary")
    generated_identity_pass = (
        generated_intents_sha256
        == str(frozen["generated_intents_sha256"])
    )

    original_selected, stressed, stress_receipt = _validate_and_stress_composed(
        composed_report=composed_report,
        design=design,
        composed_report_sha256=composed_report_sha256,
    )

    teacher_pool = _prepare_teacher_pool(
        fixture=fixture,
        workdir=workdir,
    )
    if str(teacher_pool["item_id"]) != str(frozen["teacher_control_item"]):
        raise RuntimeError("unexpected teacher control item")
    propositions = [dict(row) for row in teacher_pool["page_pool"]]
    source_ids = {
        str(row["proposition_id"])
        for row in propositions
    }
    all_original_locatable = all(
        str(row["proposition_id"]) in source_ids
        for row in original_selected
    )
    stressed_ids = {
        str(row["proposition_id"])
        for row in stressed
    }

    seeds = _reservation_seeds(original_selected)
    candidates = _candidate_neighbors(
        seeds=seeds,
        propositions=propositions,
        stressed_ids=stressed_ids,
    )
    generated_teacher = _generated_teacher_paper(generated)
    rankings = _surviving_rankings(
        propositions=propositions,
        generated_teacher=generated_teacher,
        intent_specs=[
            dict(row)
            for row in frozen["surviving_tier1_intents"]
        ],
        depth=int(frozen["surviving_intent_candidate_depth"]),
    )
    eligible, ineligible = _score_candidates(
        candidates=candidates,
        rankings=rankings,
    )

    rescued = None
    chosen = None
    support = None
    all_rescued_locatable = False
    selected_count_pass = False

    if eligible:
        chosen = dict(eligible[0])
        source = {
            str(row["proposition_id"]): dict(row)
            for row in propositions
        }
        chosen_pid = str(chosen["proposition_id"])
        chosen_row = dict(source[chosen_pid])
        chosen_row["selection_origin"] = "local_neighborhood_rescue"
        chosen_row["rescue_seed_proposition_id"] = str(
            chosen["canonical_seed_proposition_id"]
        )
        chosen_row["rescue_seed_frontier_rank"] = int(
            chosen["seed_frontier_rank"]
        )
        chosen_row["rescue_neighbor_relation"] = str(
            chosen["neighbor_relation"]
        )
        chosen_row["rescue_support_count"] = int(
            chosen["support_count"]
        )
        chosen_row["rescue_best_rank"] = int(chosen["best_rank"])

        rescued = [dict(row) for row in stressed] + [chosen_row]
        selected_count_pass = len(rescued) == FIXED_BUDGET
        if len({str(row["proposition_id"]) for row in rescued}) != len(rescued):
            raise RuntimeError("rescue introduced duplicate proposition identity")
        all_rescued_locatable = _locatable(rescued, propositions)
        atoms = [dict(row) for row in support_fixture["atoms"]]
        support = _generic_support(atoms, rescued)

    hard_safety = {
        "composed_report_identity_pass": bool(
            stress_receipt["composed_report_identity_pass"]
        ),
        "generated_intents_identity_pass": generated_identity_pass,
        "stress_exact_removal_pass": (
            stress_receipt["stress_removed_proposition_id"] == STRESS_REMOVE
            and int(stress_receipt["stressed_selected_count"]) == FIXED_BUDGET - 1
        ),
        "original_selected_propositions_locatable": all_original_locatable,
        "rescued_exact_count_pass": selected_count_pass if eligible else True,
        "rescued_propositions_locatable": all_rescued_locatable if eligible else True,
        "teacher_used_for_candidate_generation": False,
        "teacher_used_for_candidate_ranking": False,
        "teacher_used_only_for_frozen_stress_and_posthoc_evaluation": True,
        "new_query_generation_false": True,
        "new_model_or_ranker_false": True,
        "pairwise_semantic_similarity_false": True,
        "threshold_calibration_false": True,
        "new_alignment_search_false": True,
        "production_routing_changed_false": True,
        "qwen_verify_changed_false": True,
        "wave7b_started_false": True,
        "semantic_control_or_ontology_output_false": True,
        "rejected_mechanism_rescue_false": True,
    }
    safe = bool(
        hard_safety["composed_report_identity_pass"]
        and hard_safety["generated_intents_identity_pass"]
        and hard_safety["stress_exact_removal_pass"]
        and hard_safety["original_selected_propositions_locatable"]
        and hard_safety["rescued_exact_count_pass"]
        and hard_safety["rescued_propositions_locatable"]
    )

    decision = _classify(
        support=support,
        eligible_neighbor_count=len(eligible),
        safe=safe,
    )

    original_incremental = int(
        composed_report["cost_provenance"][
            "incremental_raw_source_chars_vs_top24"
        ]
    )
    removed_row = next(
        row for row in original_selected
        if str(row["proposition_id"]) == STRESS_REMOVE
    )
    rescued_incremental = None
    unique_pages = sorted(
        {int(row["page_number"]) for row in stressed}
    )
    if rescued is not None and chosen is not None:
        rescued_incremental = (
            original_incremental
            - len(str(removed_row["raw_source_form"]))
            + len(str(chosen["raw_source_form"]))
        )
        unique_pages = sorted(
            {int(row["page_number"]) for row in rescued}
        )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_ADJACENT_PROPOSITION_RESCUE_DIAGNOSTIC",
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "rescue_operator_id": operator["id"],
        "composed_report_sha256": composed_report_sha256,
        "generated_intents_sha256": generated_intents_sha256,
        **decision,
        "stress_replay": stress_receipt,
        "surviving_reservation_seeds": seeds,
        "surviving_intent_rankings": [
            {
                "generator_id": str(row["generator_id"]),
                "intent_index": int(row["intent_index"]),
                "group": str(row["group"]),
                "question": str(row["question"]),
                "top12": [
                    {
                        "proposition_id": str(item["proposition_id"]),
                        "page_number": int(item["page_number"]),
                        "bm25_rank": int(item["oracle_rank"]),
                    }
                    for item in row["ranking"]
                ],
            }
            for row in rankings
        ],
        "eligible_neighbor_candidates": eligible,
        "ineligible_neighbor_candidates": ineligible,
        "chosen_rescue": chosen,
        "rescued_teacher": support,
        "cost_provenance": {
            "selected_proposition_count": len(rescued) if rescued is not None else len(stressed),
            "incremental_raw_source_chars_vs_top24": rescued_incremental,
            "unique_source_pages": unique_pages,
            "all_selected_propositions_exactly_locatable": (
                all_rescued_locatable if eligible else all_original_locatable
            ),
        },
        "selected_propositions": [
            {
                "proposition_id": str(row["proposition_id"]),
                "page_number": int(row["page_number"]),
                "raw_source_form": str(row["raw_source_form"]),
                "selection_origin": str(row.get("selection_origin") or ""),
            }
            for row in (rescued if rescued is not None else stressed)
        ],
        "hard_safety": hard_safety,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "rejected_mechanism_rescue": False,
    }
