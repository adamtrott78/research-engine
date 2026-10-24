from __future__ import annotations

from typing import Any

from research_engine.local_extract_facet_reserved_evidence_planner import (
    _generic_support,
    _locatable,
)
from research_engine.local_extract_oracle_query import _prepare_teacher_pool


TARGET_E1A = "e1.a"
TARGET_E6A = "e6.a"
FROZEN_BUDGET = 28
FROZEN_CONTROL_COUNT = 7
EXPECTED_REMOVE = "page:p4:s33"
EXPECTED_CERTIFIER = "page:p4:s28"
EXPECTED_ADD = "page:p6:s13"
EXPECTED_ALIGNMENT_SCORE = 0.4444444444444444
EXPECTED_ALIGNMENT_THRESHOLD = 0.4


def _alignment_swap_from_result(
    alignment_result: dict[str, Any],
) -> dict[str, Any]:
    if alignment_result["alignment_augment_decision"] != "ALIGNMENT_AUGMENT_FEASIBLE":
        raise RuntimeError("unexpected alignment result decision")
    swap = dict(alignment_result["candidate_selector"]["teacher_paper_swap"])
    expected = {
        "removed_proposition_id": EXPECTED_REMOVE,
        "certifier_proposition_id": EXPECTED_CERTIFIER,
        "added_proposition_id": EXPECTED_ADD,
    }
    for key, value in expected.items():
        if str(swap[key]) != value:
            raise RuntimeError(f"unexpected frozen alignment swap {key}")
    if abs(float(swap["alignment_score"]) - EXPECTED_ALIGNMENT_SCORE) > 1e-15:
        raise RuntimeError("unexpected frozen alignment score")
    if abs(float(swap["threshold"]) - EXPECTED_ALIGNMENT_THRESHOLD) > 1e-15:
        raise RuntimeError("unexpected frozen alignment threshold")
    return swap


def _facet_budget28(
    facet_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if facet_report["facet_reservation_decision"] != "FACET_RESERVATION_CONFIRMED":
        raise RuntimeError("unexpected facet reservation decision")
    if int(facet_report["qualifying_budget"]) != FROZEN_BUDGET:
        raise RuntimeError("unexpected facet qualifying budget")
    hard = facet_report["hard_safety"]
    for key in (
        "generated_intents_identity_pass",
        "immutable_core_identity_pass",
        "maximum_budget_reached",
        "all_selected_propositions_exactly_locatable",
    ):
        if hard.get(key) is not True:
            raise RuntimeError(f"facet report hard-safety failure: {key}")

    budget = dict(facet_report["budget_metrics"][str(FROZEN_BUDGET)])
    selected = [dict(row) for row in budget["selected_propositions"]]
    if len(selected) != FROZEN_BUDGET:
        raise RuntimeError("facet budget28 does not contain exactly 28 propositions")

    core = [
        dict(row)
        for row in facet_report["budget_metrics"]["24"]["selected_propositions"]
    ]
    if len(core) != 24:
        raise RuntimeError("facet immutable core does not contain 24 propositions")
    return selected, budget, core


def _apply_exact_swap(
    *,
    selected: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
    swap: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {
        str(row["proposition_id"]): dict(row)
        for row in propositions
    }
    ids = [str(row["proposition_id"]) for row in selected]

    if ids.count(EXPECTED_REMOVE) != 1:
        raise RuntimeError("frozen removed proposition missing or duplicated")
    if EXPECTED_CERTIFIER not in ids:
        raise RuntimeError("frozen same-page certifier missing")
    if EXPECTED_ADD in ids:
        raise RuntimeError("frozen alignment addition already selected")
    if EXPECTED_ADD not in source:
        raise RuntimeError("frozen alignment addition missing from source pool")

    removed_index = ids.index(EXPECTED_REMOVE)
    removed_row = dict(selected[removed_index])
    certifier_row = next(
        row for row in selected
        if str(row["proposition_id"]) == EXPECTED_CERTIFIER
    )
    if int(removed_row["page_number"]) != int(certifier_row["page_number"]):
        raise RuntimeError("frozen certifier is no longer on the removed proposition page")

    added_row = dict(source[EXPECTED_ADD])
    added_row["selection_origin"] = "validated_alignment_swap"
    added_row["composition_swap_from"] = EXPECTED_REMOVE
    added_row["composition_certifier"] = EXPECTED_CERTIFIER
    added_row["composition_alignment_score"] = float(swap["alignment_score"])
    added_row["composition_alignment_threshold"] = float(swap["threshold"])

    composed = [dict(row) for row in selected]
    composed[removed_index] = added_row
    if len(composed) != FROZEN_BUDGET:
        raise RuntimeError("composition changed the fixed evidence budget")
    if len({str(row["proposition_id"]) for row in composed}) != FROZEN_BUDGET:
        raise RuntimeError("composition introduced duplicate proposition identity")

    return composed, {
        "removed_proposition_id": EXPECTED_REMOVE,
        "removed_page_number": int(removed_row["page_number"]),
        "certifier_proposition_id": EXPECTED_CERTIFIER,
        "certifier_page_number": int(certifier_row["page_number"]),
        "added_proposition_id": EXPECTED_ADD,
        "added_page_number": int(added_row["page_number"]),
        "alignment_score": float(swap["alignment_score"]),
        "alignment_threshold": float(swap["threshold"]),
        "replacement_index_zero_based": removed_index,
    }


def _sensitivity_set(
    *,
    composed: list[dict[str, Any]],
    facet_budget: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    removed = [
        str(value)
        for value in facet_budget["flagged_only_removed_proposition_ids"]
    ]
    removed_set = set(removed)
    sensitivity = [
        dict(row)
        for row in composed
        if str(row["proposition_id"]) not in removed_set
    ]
    return sensitivity, removed


def _classify(
    *,
    primary: dict[str, Any],
    sensitivity: dict[str, Any],
    safe: bool,
) -> dict[str, Any]:
    if not safe:
        return {
            "composition_decision": "INVALID_EXECUTION",
            "interpretation": (
                "Exact-input, swap-precondition, provenance, fixed-budget, or "
                "teacher-blind safety failed; no scientific verdict."
            ),
        }

    primary_supported = set(primary["supported_atoms"])
    sensitivity_supported = set(sensitivity["supported_atoms"])
    primary_controls = int(primary["frozen_control_retention_count"])

    if primary_controls < FROZEN_CONTROL_COUNT:
        return {
            "composition_decision": "COMPOSITION_COLLISION",
            "interpretation": (
                "Serial composition loses at least one frozen control atom and "
                "therefore violates the preservation objective."
            ),
        }

    e1a = TARGET_E1A in primary_supported
    e6a = TARGET_E6A in primary_supported

    if e1a and e6a:
        sensitivity_robust = bool(
            int(sensitivity["frozen_control_retention_count"])
            == FROZEN_CONTROL_COUNT
            and TARGET_E1A in sensitivity_supported
            and TARGET_E6A in sensitivity_supported
        )
        if sensitivity_robust:
            return {
                "composition_decision": "COMPOSED_COMPLEMENTARITY_ROBUST",
                "interpretation": (
                    "Acquisition and preservation compose at budget28 and both "
                    "residual gains survive the flagged-origin sensitivity."
                ),
            }
        return {
            "composition_decision": "COMPOSED_COMPLEMENTARITY_FRAGILE",
            "interpretation": (
                "Acquisition and preservation compose at budget28, but the "
                "combined residual gain is not robust to the frozen "
                "flagged-origin sensitivity."
            ),
        }

    if e1a ^ e6a:
        return {
            "composition_decision": "COMPOSITION_PARTIAL",
            "interpretation": (
                "The seven-atom control core survives, but only one independently "
                "positive residual transports into the composed evidence set."
            ),
        }

    return {
        "composition_decision": "COMPOSITION_NEGATIVE",
        "interpretation": (
            "The seven-atom control core survives, but neither independently "
            "positive residual transports into the composed evidence set."
        ),
    }


def evaluate_composed_acquisition_preservation(
    *,
    fixture: dict[str, Any],
    support_fixture: dict[str, Any],
    design: dict[str, Any],
    facet_report: dict[str, Any],
    facet_report_sha256: str,
    alignment_result: dict[str, Any],
    workdir: Any,
) -> dict[str, Any]:
    if design["design_id"] != "local-extract-composed-acquisition-preservation-v1":
        raise RuntimeError("unexpected frozen composition design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "PLANNER_ALIGNMENT_COMPOSITION_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen composition experiment")
    if int(experiment["frozen_inputs"]["facet_planner_budget"]) != FROZEN_BUDGET:
        raise RuntimeError("unexpected frozen planner budget")

    expected_report_sha = str(
        experiment["frozen_inputs"]["facet_planner_report_sha256"]
    )
    report_identity_pass = facet_report_sha256 == expected_report_sha

    selected, facet_budget, core = _facet_budget28(facet_report)
    swap = _alignment_swap_from_result(alignment_result)

    teacher_pool = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    if str(teacher_pool["item_id"]) != str(
        experiment["frozen_inputs"]["teacher_control_item"]
    ):
        raise RuntimeError("unexpected teacher control item")
    propositions = [dict(row) for row in teacher_pool["page_pool"]]

    composed, swap_receipt = _apply_exact_swap(
        selected=selected,
        propositions=propositions,
        swap=swap,
    )
    sensitivity_selected, sensitivity_removed = _sensitivity_set(
        composed=composed,
        facet_budget=facet_budget,
    )

    atoms = [dict(row) for row in support_fixture["atoms"]]
    primary = _generic_support(atoms, composed)
    sensitivity = _generic_support(atoms, sensitivity_selected)

    all_locatable = _locatable(composed, propositions)
    core_chars = sum(len(str(row["raw_source_form"])) for row in core)
    composed_chars = sum(len(str(row["raw_source_form"])) for row in composed)

    hard_safety = {
        "facet_report_identity_pass": report_identity_pass,
        "planner28_exact_count_pass": len(selected) == FROZEN_BUDGET,
        "composed_exact_count_pass": len(composed) == FROZEN_BUDGET,
        "all_selected_propositions_exactly_locatable": all_locatable,
        "exact_alignment_swap_preconditions_pass": True,
        "teacher_used_for_selection": False,
        "teacher_used_only_for_posthoc_evaluation": True,
        "new_query_generation_false": True,
        "new_model_or_ranker_false": True,
        "new_alignment_search_false": True,
        "pairwise_semantic_similarity_new_false": True,
        "threshold_calibration_false": True,
        "production_routing_changed_false": True,
        "qwen_verify_changed_false": True,
        "wave7b_started_false": True,
        "semantic_control_or_ontology_output_false": True,
        "rejected_mechanism_rescue_false": True,
    }
    safe = all(bool(value) for value in hard_safety.values())

    decision = _classify(
        primary=primary,
        sensitivity=sensitivity,
        safe=safe,
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_COMPOSED_ACQUISITION_PRESERVATION_DIAGNOSTIC",
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "composition_operator_id": experiment["composition_operator"]["id"],
        "facet_report_sha256": facet_report_sha256,
        "fixed_budget": FROZEN_BUDGET,
        **decision,
        "primary_teacher": primary,
        "posthoc_flagged_origin_sensitivity_teacher": sensitivity,
        "sensitivity_removed_proposition_ids": sensitivity_removed,
        "composition_swap": swap_receipt,
        "cost_provenance": {
            "selected_proposition_count": len(composed),
            "incremental_raw_source_chars_vs_top24": composed_chars - core_chars,
            "unique_source_pages": sorted(
                {int(row["page_number"]) for row in composed}
            ),
            "all_selected_propositions_exactly_locatable": all_locatable,
        },
        "selected_propositions": [
            {
                "proposition_id": str(row["proposition_id"]),
                "page_number": int(row["page_number"]),
                "raw_source_form": str(row["raw_source_form"]),
                "selection_origin": str(row.get("selection_origin") or ""),
            }
            for row in composed
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
