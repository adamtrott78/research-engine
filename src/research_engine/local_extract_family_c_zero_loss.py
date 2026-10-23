from __future__ import annotations

from typing import Any

from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    CONTROL_RRF_ID,
    CONTROL_SUPPORTED_ATOMS,
    QUOTA_ID,
    RESIDUAL_ID,
    RESIDUAL_RRF_ID,
    OBSERVED_RESIDUAL_ATOMS,
    _arm_rankings,
    _arm_summary,
    _plain_rankings,
    _quota_merge,
)
from research_engine.local_extract_wave7_evaluate import (
    _prepare_papers,
    _rrf_merge,
)


CANDIDATE_ID = "LEXICOGRAPHIC_ZERO_LOSS_SWAP_V1"
EPSILON = 1e-15


def _coverage_utility(
    selected_ids: set[str],
    arm: dict[str, Any],
    *,
    rrf_k: int,
) -> float:
    rankings = list(arm["rankings"])
    if not rankings:
        return 0.0
    total = 0.0
    for intent in rankings:
        best = 0.0
        for rank, row in enumerate(intent["ranking"], start=1):
            if str(row["proposition_id"]) in selected_ids:
                best = 1.0 / (rrf_k + rank)
                break
        total += best
    return total / len(rankings)


def _candidate_rows(
    baseline: list[dict[str, Any]],
    residual: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {str(row["proposition_id"]): dict(row) for row in baseline}
    for intent in residual["rankings"]:
        for row in intent["ranking"]:
            rows.setdefault(str(row["proposition_id"]), dict(row))
    return rows


def _ranked_output(
    selected_ids: list[str],
    rows: dict[str, dict[str, Any]],
    *,
    control: dict[str, Any],
    residual: dict[str, Any],
    rrf_k: int,
) -> list[dict[str, Any]]:
    output = []
    selected_set = set(selected_ids)
    control_utility = _coverage_utility(selected_set, control, rrf_k=rrf_k)
    residual_utility = _coverage_utility(selected_set, residual, rrf_k=rrf_k)
    for rank, pid in enumerate(selected_ids, start=1):
        row = dict(rows[pid])
        row["fusion_rank"] = rank
        row["fusion_score"] = float(
            control_utility + residual_utility
        )
        output.append(row)
    return output


def _lexicographic_zero_loss_swap(
    *,
    baseline: list[dict[str, Any]],
    control: dict[str, Any],
    residual: dict[str, Any],
    rrf_k: int,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    rows = _candidate_rows(baseline, residual)
    selected = [str(row["proposition_id"]) for row in baseline]
    selected_set = set(selected)

    baseline_control = _coverage_utility(
        selected_set,
        control,
        rrf_k=rrf_k,
    )
    current_control = baseline_control
    current_residual = _coverage_utility(
        selected_set,
        residual,
        rrf_k=rrf_k,
    )
    swaps: list[dict[str, Any]] = []

    while True:
        best: dict[str, Any] | None = None
        residual_candidates = sorted(set(rows) - selected_set)
        for added_pid in residual_candidates:
            for removed_pid in sorted(selected_set):
                candidate_ids = (
                    selected_set - {removed_pid}
                ) | {added_pid}
                control_utility = _coverage_utility(
                    candidate_ids,
                    control,
                    rrf_k=rrf_k,
                )
                if control_utility < baseline_control - epsilon:
                    continue
                residual_utility = _coverage_utility(
                    candidate_ids,
                    residual,
                    rrf_k=rrf_k,
                )
                residual_gain = residual_utility - current_residual
                if residual_gain <= epsilon:
                    continue

                proposal = {
                    "added_proposition_id": added_pid,
                    "removed_proposition_id": removed_pid,
                    "control_utility": float(control_utility),
                    "residual_utility": float(residual_utility),
                    "residual_gain": float(residual_gain),
                }
                if best is None:
                    best = proposal
                    continue

                key = (
                    -proposal["residual_gain"],
                    -proposal["control_utility"],
                    proposal["added_proposition_id"],
                    proposal["removed_proposition_id"],
                )
                best_key = (
                    -best["residual_gain"],
                    -best["control_utility"],
                    best["added_proposition_id"],
                    best["removed_proposition_id"],
                )
                if key < best_key:
                    best = proposal

        if best is None:
            break

        index = selected.index(best["removed_proposition_id"])
        selected[index] = best["added_proposition_id"]
        selected_set = set(selected)
        current_control = float(best["control_utility"])
        current_residual = float(best["residual_utility"])
        swaps.append(
            {
                **best,
                "swap_index": len(swaps) + 1,
                "control_utility_floor": float(baseline_control),
            }
        )

    merged = _ranked_output(
        selected,
        rows,
        control=control,
        residual=residual,
        rrf_k=rrf_k,
    )
    return {
        "merged": merged,
        "baseline_control_utility": float(baseline_control),
        "final_control_utility": float(current_control),
        "baseline_residual_utility": float(
            _coverage_utility(
                {
                    str(row["proposition_id"])
                    for row in baseline
                },
                residual,
                rrf_k=rrf_k,
            )
        ),
        "final_residual_utility": float(current_residual),
        "zero_loss_constraint_pass": bool(
            current_control >= baseline_control - epsilon
            and all(
                float(row["control_utility"])
                >= baseline_control - epsilon
                for row in swaps
            )
        ),
        "swap_count": len(swaps),
        "swaps": swaps,
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
    zero_loss = bool(algorithm["zero_loss_constraint_pass"])

    if (
        support >= 9
        and residual == 2
        and retained == 7
        and page_fraction >= 0.90
        and zero_loss
    ):
        return {
            "protected_augment_decision": "PROTECTED_AUGMENT_STRONG",
            "rationale": (
                "The lexicographic selector preserves all frozen control "
                "support while recovering both observed residual atoms."
            ),
        }

    if (
        support >= 8
        and residual >= 1
        and retained == 7
        and page_fraction >= 0.90
        and zero_loss
    ):
        return {
            "protected_augment_decision": "PROTECTED_AUGMENT_FEASIBLE",
            "rationale": (
                "The lexicographic selector preserves all frozen control "
                "support and adds at least one observed residual atom."
            ),
        }

    return {
        "protected_augment_decision": "PROTECTED_AUGMENT_NEGATIVE",
        "rationale": (
            "The frozen teacher-blind zero-loss selector does not establish "
            "coverage-preserving residual augmentation."
        ),
    }


def evaluate_family_c_zero_loss(
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
    if selected["id"] != "ZERO_LOSS_RESIDUAL_AUGMENTATION_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen diagnostic")
    if algorithm["id"] != CANDIDATE_ID:
        raise RuntimeError("unexpected frozen algorithm")
    if int(algorithm["final_budget"]) != 24:
        raise RuntimeError("unexpected final budget")

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
        total_leak_rejected += len(
            control["teacher_leak_rejected"]
        )
        total_leak_rejected += len(
            residual["teacher_leak_rejected"]
        )

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
        candidate = _lexicographic_zero_loss_swap(
            baseline=control_rrf,
            control=control,
            residual=residual,
            rrf_k=rrf_k,
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
        "all_papers_zero_loss": all(
            bool(row["zero_loss_constraint_pass"])
            for row in algorithm_by_paper.values()
        ),
        "paper_metrics": algorithm_by_paper,
    }
    candidate_arm["algorithm"]["zero_loss_constraint_pass"] = bool(
        candidate_arm["algorithm"]["all_papers_zero_loss"]
    )

    decision = _verdict(candidate_arm)

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_FAMILY_C_ZERO_LOSS_DIAGNOSTIC",
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
            "all_papers_zero_loss": bool(
                candidate_arm["algorithm"]["all_papers_zero_loss"]
            ),
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "ontology_or_semantic_control_in_extract_false": True,
            "structured_gap_judge_executed_false": True,
        },
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "ontology_or_semantic_control_in_extract": False,
        "structured_gap_judge_executed": False,
    }
