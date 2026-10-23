from __future__ import annotations

from typing import Any

from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    CONTROL_RRF_ID,
    QUOTA_ID,
    RESIDUAL_ID,
    RESIDUAL_RRF_ID,
    _arm_rankings,
    _arm_summary,
    _plain_rankings,
    _quota_merge,
)
from research_engine.local_extract_wave1 import token_f1
from research_engine.local_extract_wave7_evaluate import (
    _prepare_papers,
    _rrf_merge,
)


CANDIDATE_ID = "DYNAMIC_SAME_PAGE_ALIGNMENT_SWAP_V1"


def _best_same_page_certificate(
    *,
    removed_pid: str,
    retained_baseline_ids: set[str],
    baseline_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    removed = baseline_rows[removed_pid]
    removed_page = int(removed["page_number"])
    best_score = 0.0
    best_pid: str | None = None
    for pid in sorted(retained_baseline_ids):
        if pid == removed_pid:
            continue
        row = baseline_rows[pid]
        if int(row["page_number"]) != removed_page:
            continue
        score = token_f1(
            str(removed["raw_source_form"]),
            str(row["raw_source_form"]),
        )
        if (
            best_pid is None
            or score > best_score
            or (abs(score - best_score) <= 1e-15 and pid < best_pid)
        ):
            best_score = float(score)
            best_pid = pid
    return {
        "removed_proposition_id": removed_pid,
        "certifier_proposition_id": best_pid,
        "alignment_score": float(best_score),
        "page_number": removed_page,
    }


def _page_set(
    selected_ids: list[str],
    rows: dict[str, dict[str, Any]],
) -> set[int]:
    return {int(rows[pid]["page_number"]) for pid in selected_ids}


def _alignment_certified_swap(
    *,
    baseline: list[dict[str, Any]],
    residual_rrf: list[dict[str, Any]],
    alignment_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= alignment_threshold <= 1.0:
        raise ValueError("alignment_threshold must be in [0,1]")

    baseline_rows = {
        str(row["proposition_id"]): dict(row)
        for row in baseline
    }
    rows = dict(baseline_rows)
    for row in residual_rrf:
        rows.setdefault(str(row["proposition_id"]), dict(row))

    original_baseline_ids = set(baseline_rows)
    baseline_pages = _page_set(
        [str(row["proposition_id"]) for row in baseline],
        rows,
    )
    residual_order = [
        str(row["proposition_id"])
        for row in residual_rrf
        if str(row["proposition_id"]) not in original_baseline_ids
    ]

    selected = [str(row["proposition_id"]) for row in baseline]
    selected_set = set(selected)
    swaps: list[dict[str, Any]] = []

    while True:
        remaining_baseline = original_baseline_ids & selected_set
        removal_candidates: list[dict[str, Any]] = []
        for removed_pid in sorted(remaining_baseline):
            certificate = _best_same_page_certificate(
                removed_pid=removed_pid,
                retained_baseline_ids=remaining_baseline - {removed_pid},
                baseline_rows=baseline_rows,
            )
            if float(certificate["alignment_score"]) < alignment_threshold:
                continue
            removal_candidates.append(certificate)

        if not removal_candidates:
            break

        added_pid = next(
            (pid for pid in residual_order if pid not in selected_set),
            None,
        )
        if added_pid is None:
            break

        removal_candidates.sort(
            key=lambda row: (
                -int(
                    baseline_rows[
                        str(row["removed_proposition_id"])
                    ]["fusion_rank"]
                ),
                str(row["removed_proposition_id"]),
            )
        )

        chosen: dict[str, Any] | None = None
        for certificate in removal_candidates:
            removed_pid = str(certificate["removed_proposition_id"])
            candidate_ids = [
                added_pid if pid == removed_pid else pid
                for pid in selected
            ]
            if not baseline_pages <= _page_set(candidate_ids, rows):
                continue
            chosen = certificate
            break

        if chosen is None:
            break

        removed_pid = str(chosen["removed_proposition_id"])
        index = selected.index(removed_pid)
        selected[index] = added_pid
        selected_set = set(selected)
        swaps.append(
            {
                **chosen,
                "added_proposition_id": added_pid,
                "swap_index": len(swaps) + 1,
                "removed_frozen_fusion_rank": int(
                    baseline_rows[removed_pid]["fusion_rank"]
                ),
                "added_residual_fusion_rank": int(
                    rows[added_pid]["fusion_rank"]
                ),
                "threshold": float(alignment_threshold),
            }
        )

    merged: list[dict[str, Any]] = []
    for rank, pid in enumerate(selected, start=1):
        row = dict(rows[pid])
        row["fusion_rank"] = rank
        row["fusion_score"] = float(row.get("fusion_score") or 0.0)
        row["selection_origin"] = (
            "frozen_source_aware_baseline"
            if pid in original_baseline_ids
            else "residual_alignment_slot"
        )
        merged.append(row)

    final_pages = _page_set(selected, rows)
    certificate_pass = all(
        float(row["alignment_score"]) >= alignment_threshold
        for row in swaps
    )
    return {
        "merged": merged,
        "swap_count": len(swaps),
        "swaps": swaps,
        "alignment_threshold": float(alignment_threshold),
        "certificate_pass": certificate_pass,
        "page_preservation_pass": bool(baseline_pages <= final_pages),
        "baseline_pages": sorted(baseline_pages),
        "final_pages": sorted(final_pages),
    }


def _verdict(candidate: dict[str, Any]) -> dict[str, Any]:
    teacher = candidate["teacher"]
    nine = candidate["nine_paper"]
    algorithm = candidate["algorithm"]
    support = int(teacher["supported_atoms_at_24"])
    residual = int(teacher["observed_residual_recovery_count"])
    retained = int(teacher["frozen_control_retention_count"])
    page_fraction = float(
        nine["merged24_page_recall_fraction_of_ceiling"]
    )
    safe = bool(
        algorithm["all_papers_certificate_pass"]
        and algorithm["all_papers_page_preserved"]
    )

    if (
        support >= 9
        and residual == 2
        and retained == 7
        and page_fraction >= 0.90
        and safe
    ):
        return {
            "alignment_augment_decision": "ALIGNMENT_AUGMENT_STRONG",
            "rationale": (
                "Direct proposition-alignment certificates preserve all frozen "
                "control support while recovering both observed residual atoms."
            ),
        }

    if (
        support >= 8
        and residual >= 1
        and retained == 7
        and page_fraction >= 0.90
        and safe
    ):
        return {
            "alignment_augment_decision": "ALIGNMENT_AUGMENT_FEASIBLE",
            "rationale": (
                "Direct proposition-alignment certificates preserve all frozen "
                "control support while adding at least one observed residual atom."
            ),
        }

    return {
        "alignment_augment_decision": "ALIGNMENT_AUGMENT_NEGATIVE",
        "rationale": (
            "The frozen proposition-alignment certificate does not establish "
            "coverage-preserving residual augmentation."
        ),
    }


def evaluate_proposition_alignment_preservation(
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
        raise RuntimeError(
            "generated-intents artifact violates scientific boundary"
        )

    selected = design["selected_first_diagnostic"]
    algorithm = selected["algorithm"]
    if (
        selected["id"]
        != "PROPOSITION_ALIGNMENT_CERTIFIED_AUGMENTATION_DIAGNOSTIC_V1"
    ):
        raise RuntimeError("unexpected frozen diagnostic")
    if algorithm["id"] != CANDIDATE_ID:
        raise RuntimeError("unexpected frozen algorithm")
    if int(algorithm["final_budget"]) != 24:
        raise RuntimeError("unexpected final budget")

    threshold = float(algorithm["alignment_threshold"])
    per_intent_depth = 12
    budget = int(algorithm["final_budget"])
    rrf_k = int(
        wave7a3_contract["retrieval"]["fusion"]["k"]
    )
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

    selected_by_arm = {
        CANDIDATE_ID: {},
        CONTROL_RRF_ID: {},
        RESIDUAL_RRF_ID: {},
        QUOTA_ID: {},
    }
    algorithm_by_paper: dict[str, dict[str, Any]] = {}
    total_leak_rejected = 0

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

        control_rrf = _rrf_merge(
            _plain_rankings(control),
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=budget,
        )
        residual_rrf = _rrf_merge(
            _plain_rankings(residual),
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=budget,
        )
        candidate = _alignment_certified_swap(
            baseline=control_rrf,
            residual_rrf=residual_rrf,
            alignment_threshold=threshold,
        )
        quota = _quota_merge(
            control_rrf,
            residual_rrf,
            control_quota=20,
            residual_quota=4,
            budget=budget,
        )

        item_id = str(paper["item_id"])
        selected_by_arm[CANDIDATE_ID][item_id] = candidate["merged"]
        selected_by_arm[CONTROL_RRF_ID][item_id] = control_rrf
        selected_by_arm[RESIDUAL_RRF_ID][item_id] = residual_rrf
        selected_by_arm[QUOTA_ID][item_id] = quota
        algorithm_by_paper[item_id] = {
            key: value
            for key, value in candidate.items()
            if key != "merged"
        }

    arms = {
        arm_id: _arm_summary(
            papers=papers,
            selected_by_paper=rows,
            atoms=atoms,
        )
        for arm_id, rows in selected_by_arm.items()
    }

    candidate_arm = arms[CANDIDATE_ID]
    teacher_paper = next(
        row
        for row in candidate_arm["papers"]
        if "teacher_support" in row
    )
    teacher_item_id = str(teacher_paper["item_id"])
    candidate_arm["algorithm"] = {
        **algorithm_by_paper[teacher_item_id],
        "all_papers_certificate_pass": all(
            bool(row["certificate_pass"])
            for row in algorithm_by_paper.values()
        ),
        "all_papers_page_preserved": all(
            bool(row["page_preservation_pass"])
            for row in algorithm_by_paper.values()
        ),
        "paper_metrics": algorithm_by_paper,
    }

    decision = _verdict(candidate_arm)

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_PROPOSITION_ALIGNMENT_PRESERVATION",
        "design_id": design["design_id"],
        "baseline_result_sha": design["baseline_result_sha"],
        "generated_intents_sha256": generated_intents_sha256,
        "candidate_selector_id": CANDIDATE_ID,
        **decision,
        "arms": arms,
        "teacher_leak_reject_count": total_leak_rejected,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "hard_safety": {
            "teacher_blind_selector": True,
            "all_papers_certificate_pass": bool(
                candidate_arm["algorithm"][
                    "all_papers_certificate_pass"
                ]
            ),
            "all_papers_page_preserved": bool(
                candidate_arm["algorithm"][
                    "all_papers_page_preserved"
                ]
            ),
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "ontology_or_semantic_control_in_extract_false": True,
            "fixed_quota_promoted_false": True,
        },
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "ontology_or_semantic_control_in_extract": False,
        "fixed_quota_promoted": False,
    }
