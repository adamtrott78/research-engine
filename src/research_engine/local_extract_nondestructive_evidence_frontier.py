from __future__ import annotations

from typing import Any

from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    CONTROL_SUPPORTED_ATOMS,
    OBSERVED_RESIDUAL_ATOMS,
    RESIDUAL_ID,
    _arm_rankings,
    _plain_rankings,
)
from research_engine.local_extract_wave7_evaluate import (
    _prepare_papers,
    _rrf_merge,
    _teacher_support,
)


CANDIDATE_ID = "IMMUTABLE_CORE_APPEND_FRONTIER_V1"
CORE_BUDGET = 24
FROZEN_BUDGETS = (24, 26, 28, 32)


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["proposition_id"]) for row in rows]


def _full_rrf_order(
    rankings: list[list[dict[str, Any]]],
    *,
    rrf_k: int,
    per_intent_depth: int,
) -> list[dict[str, Any]]:
    unique_ids = {
        str(row["proposition_id"])
        for ranking in rankings
        for row in ranking[:per_intent_depth]
    }
    if not unique_ids:
        return []
    return _rrf_merge(
        rankings,
        rrf_k=rrf_k,
        per_intent_depth=per_intent_depth,
        budget=len(unique_ids),
    )


def _append_frontier(
    *,
    core: list[dict[str, Any]],
    residual_order: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    if budget < len(core):
        raise ValueError("frontier budget may not truncate the immutable core")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source in core:
        pid = str(source["proposition_id"])
        if pid in seen:
            raise RuntimeError("immutable core contains duplicate proposition IDs")
        seen.add(pid)
        row = dict(source)
        row["selection_origin"] = "immutable_source_aware_core"
        row["source_fusion_rank"] = int(source["fusion_rank"])
        selected.append(row)

    for source in residual_order:
        if len(selected) >= budget:
            break
        pid = str(source["proposition_id"])
        if pid in seen:
            continue
        seen.add(pid)
        row = dict(source)
        row["selection_origin"] = "append_only_author_anchored_frontier"
        row["source_fusion_rank"] = int(source["fusion_rank"])
        selected.append(row)

    for final_rank, row in enumerate(selected, start=1):
        row["frontier_rank"] = final_rank

    return selected


def _locatable(
    selected: list[dict[str, Any]],
    proposition_rows: list[dict[str, Any]],
) -> bool:
    source = {
        str(row["proposition_id"]): row
        for row in proposition_rows
    }
    for row in selected:
        pid = str(row["proposition_id"])
        original = source.get(pid)
        if original is None:
            return False
        if int(row["page_number"]) != int(original["page_number"]):
            return False
        if str(row["raw_source_form"]) != str(original["raw_source_form"]):
            return False
    return True


def _generic_teacher_support(
    atoms: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    legacy = _teacher_support(atoms, selected)
    supported = [str(value) for value in legacy["supported_atoms"]]
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
        "atom_details": legacy["atoms"],
    }


def _budget_summary(
    *,
    papers: list[dict[str, Any]],
    selected_by_paper: dict[str, list[dict[str, Any]]],
    atoms: list[dict[str, Any]],
    baseline_chars: int,
) -> dict[str, Any]:
    total_expected_pages = 0
    total_coarse_hits = 0
    total_frontier_hits = 0
    page_keys: set[tuple[str, int]] = set()
    total_chars = 0
    all_locatable = True
    teacher: dict[str, Any] | None = None
    paper_rows: list[dict[str, Any]] = []

    for paper in papers:
        item_id = str(paper["item_id"])
        selected = selected_by_paper[item_id]
        expected = {int(value) for value in paper["historical_evidence_pages"]}
        coarse_pages = {int(value) for value in paper["selected_page_numbers"]}
        frontier_pages = {int(row["page_number"]) for row in selected}

        total_expected_pages += len(expected)
        total_coarse_hits += len(expected & coarse_pages)
        total_frontier_hits += len(expected & frontier_pages)
        total_chars += sum(len(str(row["raw_source_form"])) for row in selected)
        page_keys.update((item_id, page) for page in frontier_pages)
        paper_locatable = _locatable(selected, paper["propositions"])
        all_locatable = all_locatable and paper_locatable

        row = {
            "item_id": item_id,
            "selected_proposition_count": len(selected),
            "source_page_numbers": sorted(frontier_pages),
            "historical_page_hits": len(expected & frontier_pages),
            "historical_page_count": len(expected),
            "raw_source_chars": sum(
                len(str(value["raw_source_form"])) for value in selected
            ),
            "exactly_locatable": paper_locatable,
            "selected_propositions": [
                {
                    "proposition_id": str(value["proposition_id"]),
                    "page_number": int(value["page_number"]),
                    "frontier_rank": int(value["frontier_rank"]),
                    "selection_origin": str(value["selection_origin"]),
                    "source_fusion_rank": int(value["source_fusion_rank"]),
                    "raw_source_form": str(value["raw_source_form"]),
                }
                for value in selected
            ],
        }
        if paper.get("teacher_claims"):
            teacher = _generic_teacher_support(atoms, selected)
            row["teacher_support"] = teacher
        paper_rows.append(row)

    coarse_ceiling = (
        total_coarse_hits / total_expected_pages
        if total_expected_pages
        else 1.0
    )
    page_recall = (
        total_frontier_hits / total_expected_pages
        if total_expected_pages
        else 1.0
    )
    teacher = teacher or {
        "supported_atoms": [],
        "supported_atom_count": 0,
        "support_recall": 0.0,
        "frozen_control_atoms_retained": [],
        "frozen_control_retention_count": 0,
        "observed_residual_atoms_recovered": [],
        "observed_residual_recovery_count": 0,
        "atom_details": [],
    }
    return {
        "teacher": teacher,
        "nine_paper": {
            "historical_evidence_bearing_page_count": total_expected_pages,
            "coarse_selected_page_hits": total_coarse_hits,
            "coarse_selected_page_recall_ceiling": coarse_ceiling,
            "frontier_page_hits": total_frontier_hits,
            "historical_evidence_bearing_page_recall": page_recall,
            "frontier_page_recall_fraction_of_coarse_ceiling": (
                page_recall / coarse_ceiling if coarse_ceiling else 1.0
            ),
            "unique_source_page_count": len(page_keys),
            "total_raw_source_chars": total_chars,
            "incremental_raw_source_chars_vs_top24": (
                total_chars - baseline_chars
            ),
            "all_propositions_exactly_locatable": all_locatable,
        },
        "papers": paper_rows,
    }


def _frontier_verdict(
    metrics: dict[int, dict[str, Any]],
    *,
    safe: bool,
) -> dict[str, Any]:
    def values(budget: int) -> tuple[int, int, int]:
        teacher = metrics[budget]["teacher"]
        return (
            int(teacher["supported_atom_count"]),
            int(teacher["frozen_control_retention_count"]),
            int(teacher["observed_residual_recovery_count"]),
        )

    confirmed_at: int | None = None
    for budget in (24, 26, 28):
        support, retained, residual = values(budget)
        if safe and retained == 7 and residual == 2 and support >= 9:
            confirmed_at = budget
            break
    if confirmed_at is not None:
        return {
            "frontier_decision": "ADDITIVE_FRONTIER_CONFIRMED",
            "qualifying_budget": confirmed_at,
            "interpretation": (
                "A frozen small append-only frontier exposes both observed "
                "complementary residual atoms without altering the immutable "
                "source-aware core."
            ),
        }

    partial_at: int | None = None
    if safe:
        for budget in (24, 26, 28):
            _, retained, residual = values(budget)
            if retained == 7 and residual >= 1:
                partial_at = budget
                break
        if partial_at is None:
            _, retained32, residual32 = values(32)
            if retained32 == 7 and residual32 == 2:
                partial_at = 32

    if partial_at is not None:
        return {
            "frontier_decision": "ADDITIVE_FRONTIER_PARTIAL",
            "qualifying_budget": partial_at,
            "interpretation": (
                "Static budget collision contributes to the residual but does "
                "not fully explain it within the preferred small frontier."
            ),
        }

    return {
        "frontier_decision": "ADDITIVE_FRONTIER_NEGATIVE",
        "qualifying_budget": None,
        "interpretation": (
            "The frozen append-only branch union does not establish small "
            "static budget collision as the primary remaining mechanism."
        ),
    }


def evaluate_nondestructive_evidence_frontier(
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
        raise RuntimeError("generated-intents artifact violates scientific boundary")

    if design["design_id"] != "local-extract-nondestructive-evidence-frontier-v1":
        raise RuntimeError("unexpected frozen design")
    experiment = design["selected_discriminating_experiment"]
    selector = experiment["selector"]
    if experiment["id"] != "NONDESTRUCTIVE_EVIDENCE_FRONTIER_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen experiment")
    if selector["id"] != CANDIDATE_ID:
        raise RuntimeError("unexpected frozen selector")
    budgets = tuple(int(value) for value in selector["diagnostic_budgets"])
    if budgets != FROZEN_BUDGETS:
        raise RuntimeError("unexpected frozen diagnostic budgets")
    if int(selector["primary_small_frontier_ceiling"]) != 28:
        raise RuntimeError("unexpected small frontier ceiling")
    if int(selector["maximum_diagnostic_ceiling"]) != 32:
        raise RuntimeError("unexpected maximum diagnostic ceiling")
    if selector["teacher_blind"] is not True:
        raise RuntimeError("selector must remain teacher-blind")
    if selector["no_pairwise_similarity_score"] is not True:
        raise RuntimeError("pairwise similarity is forbidden")
    if selector["no_threshold_calibration"] is not True:
        raise RuntimeError("threshold calibration is forbidden")
    if selector["no_post_result_budget_selection"] is not True:
        raise RuntimeError("post-result budget selection is forbidden")

    per_intent_depth = int(wave7a3_contract["retrieval"]["per_intent_depth"])
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    forbidden = list(
        wave7a3_contract["teacher_anti_leak_audit"][
            "forbidden_normalized_substrings"
        ]
    )

    papers = _prepare_papers(
        fixture=fixture,
        generated=generated,
        workdir=workdir,
    )
    atoms = [dict(row) for row in support_fixture["atoms"]]

    selected_by_budget: dict[int, dict[str, list[dict[str, Any]]]] = {
        budget: {} for budget in budgets
    }
    paper_diagnostics: list[dict[str, Any]] = []
    total_leak_rejected = 0
    all_core_prefix_preserved = True
    all_residual_prefix24_match_frozen = True

    for paper in papers:
        control = _arm_rankings(
            paper=paper,
            generator_id=CONTROL_ID,
            forbidden=forbidden,
            per_intent_depth=per_intent_depth,
        )
        residual = _arm_rankings(
            paper=paper,
            generator_id=RESIDUAL_ID,
            forbidden=forbidden,
            per_intent_depth=per_intent_depth,
        )
        total_leak_rejected += len(control["teacher_leak_rejected"])
        total_leak_rejected += len(residual["teacher_leak_rejected"])

        control_rankings = _plain_rankings(control)
        residual_rankings = _plain_rankings(residual)
        core = _rrf_merge(
            control_rankings,
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=CORE_BUDGET,
        )
        frozen_residual24 = _rrf_merge(
            residual_rankings,
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=CORE_BUDGET,
        )
        residual_order = _full_rrf_order(
            residual_rankings,
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
        )
        prefix_match = _ids(residual_order[:CORE_BUDGET]) == _ids(
            frozen_residual24
        )
        all_residual_prefix24_match_frozen = (
            all_residual_prefix24_match_frozen and prefix_match
        )

        item_id = str(paper["item_id"])
        core_ids = _ids(core)
        budget_counts: dict[str, int] = {}
        for budget in budgets:
            selected = _append_frontier(
                core=core,
                residual_order=residual_order,
                budget=budget,
            )
            prefix_preserved = _ids(selected[: len(core)]) == core_ids
            all_core_prefix_preserved = (
                all_core_prefix_preserved and prefix_preserved
            )
            selected_by_budget[budget][item_id] = selected
            budget_counts[str(budget)] = len(selected)

        paper_diagnostics.append(
            {
                "item_id": item_id,
                "immutable_core_count": len(core),
                "residual_full_order_count": len(residual_order),
                "residual_prefix24_matches_frozen": prefix_match,
                "control_post_leak_intents": control["post_leak_count"],
                "residual_post_leak_intents": residual["post_leak_count"],
                "control_teacher_leak_rejected": control[
                    "teacher_leak_rejected"
                ],
                "residual_teacher_leak_rejected": residual[
                    "teacher_leak_rejected"
                ],
                "selected_count_by_budget": budget_counts,
            }
        )

    top24_chars = sum(
        len(str(row["raw_source_form"]))
        for selected in selected_by_budget[24].values()
        for row in selected
    )
    metrics = {
        budget: _budget_summary(
            papers=papers,
            selected_by_paper=selected_by_budget[budget],
            atoms=atoms,
            baseline_chars=top24_chars,
        )
        for budget in budgets
    }

    all_locatable = all(
        bool(metrics[budget]["nine_paper"]["all_propositions_exactly_locatable"])
        for budget in budgets
    )
    safe = bool(
        all_core_prefix_preserved
        and all_residual_prefix24_match_frozen
        and all_locatable
    )
    decision = _frontier_verdict(metrics, safe=safe)

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_NONDESTRUCTIVE_EVIDENCE_FRONTIER_DIAGNOSTIC",
        "design_id": design["design_id"],
        "generated_intents_sha256": generated_intents_sha256,
        "candidate_selector_id": CANDIDATE_ID,
        "diagnostic_budgets": list(budgets),
        **decision,
        "frontier_metrics": {
            str(budget): metrics[budget]
            for budget in budgets
        },
        "paper_diagnostics": paper_diagnostics,
        "teacher_leak_reject_count": total_leak_rejected,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "hard_safety": {
            "teacher_blind_selector": True,
            "immutable_core_prefix_preserved_all_papers": (
                all_core_prefix_preserved
            ),
            "residual_prefix24_matches_frozen_all_papers": (
                all_residual_prefix24_match_frozen
            ),
            "all_propositions_exactly_locatable": all_locatable,
            "no_pairwise_similarity_score": True,
            "no_threshold_calibration": True,
            "no_post_result_budget_selection": True,
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "ontology_or_semantic_control_in_extract_false": True,
            "rejected_mechanism_rescue_false": True,
        },
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "ontology_or_semantic_control_in_extract": False,
        "rejected_mechanism_rescue": False,
    }
