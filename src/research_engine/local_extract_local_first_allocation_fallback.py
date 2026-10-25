from __future__ import annotations

from typing import Any

SOURCE_DESIGN_ID = "local-extract-corrected-all-nine-recovery-comparison-v1"
SOURCE_ARCHITECTURE_ID = "CORRECTED_ALL_NINE_STRUCTURAL_RECOVERY_COMPARISON_V1"
SOURCE_EXPERIMENT_ID = "ALL_NINE_FIXED_BASELINE_RECOVERY_COMPARISON_V1"
DESIGN_ID = "local-extract-local-first-allocation-fallback-hybrid-v1"
ARCHITECTURE_ID = "LOCAL_FIRST_ALLOCATION_FALLBACK_HYBRID_V1"
EXPERIMENT_ID = "LOCAL_FIRST_ALLOCATION_FALLBACK_REALIZATION_V1"
OPERATOR_ID = "LOCAL_FIRST_ELIGIBILITY_FALLBACK_V1"


def _choose_action(
    *,
    local: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    """Choose an arm using only candidate presence, never evaluation fields."""
    local_chosen = local.get("chosen")
    if local_chosen is not None:
        chosen = dict(local_chosen)
        return {
            "selected_arm": "LOCAL",
            "selected_proposition_id": str(chosen["proposition_id"]),
            "selected_raw_source_chars": len(str(chosen["raw_source_form"])),
            "local_candidate_present": True,
            "allocation_invoked": False,
        }

    allocation_chosen = allocation.get("chosen")
    if allocation_chosen is not None:
        chosen = dict(allocation_chosen)
        return {
            "selected_arm": "ALLOCATION_FALLBACK",
            "selected_proposition_id": str(chosen["proposition_id"]),
            "selected_raw_source_chars": len(str(chosen["raw_source_form"])),
            "local_candidate_present": False,
            "allocation_invoked": True,
        }

    return {
        "selected_arm": "NONE",
        "selected_proposition_id": None,
        "selected_raw_source_chars": 0,
        "local_candidate_present": False,
        "allocation_invoked": True,
    }


def _verdict(
    *,
    safe: bool,
    expected_local_exact: int,
    expected_allocation_only_exact: int,
    local_exact_preserved: int,
    allocation_only_captured: int,
    hybrid_exact_recovery_count: int,
) -> dict[str, str]:
    if not safe:
        return {
            "hybrid_realization_decision": "INVALID_EXECUTION",
            "interpretation": (
                "Exact source-report identity, frozen source metrics, teacher-blind "
                "action ordering, one-slot budget, or lazy-cost invariants failed; "
                "no scientific verdict."
            ),
        }

    if local_exact_preserved < expected_local_exact:
        return {
            "hybrid_realization_decision": "HYBRID_REGRESSION",
            "interpretation": (
                "At least one exact recovery from the frozen local arm was lost "
                "under serial local-first failover."
            ),
        }

    expected_hybrid = expected_local_exact + expected_allocation_only_exact
    if (
        local_exact_preserved == expected_local_exact
        and allocation_only_captured == expected_allocation_only_exact
        and hybrid_exact_recovery_count == expected_hybrid
    ):
        return {
            "hybrid_realization_decision": "HYBRID_COMPLEMENTARITY_REALIZED",
            "interpretation": (
                "The complementary structural capabilities are realized by the "
                "teacher-blind local-first/no-local-to-allocation failover policy."
            ),
        }

    if (
        local_exact_preserved == expected_local_exact
        and allocation_only_captured == 0
        and hybrid_exact_recovery_count == expected_local_exact
    ):
        return {
            "hybrid_realization_decision": "HYBRID_COLLAPSES_TO_LOCAL",
            "interpretation": (
                "The local arm is preserved but the allocation-only capability is "
                "not realized by the executable fallback condition."
            ),
        }

    return {
        "hybrid_realization_decision": "INVALID_EXECUTION",
        "interpretation": (
            "Observed replay state does not match any frozen valid outcome "
            "partition; no scientific verdict."
        ),
    }


def evaluate_local_first_allocation_fallback(
    *,
    source_report: dict[str, Any],
    source_report_sha256: str,
    design: dict[str, Any],
) -> dict[str, Any]:
    if design.get("design_id") != DESIGN_ID:
        raise RuntimeError("unexpected local-first hybrid design")
    if design.get("selected_architecture", {}).get("id") != ARCHITECTURE_ID:
        raise RuntimeError("unexpected local-first hybrid architecture")

    experiment = design["selected_discriminating_experiment"]
    if experiment.get("id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected local-first hybrid experiment")
    operator = experiment["hybrid_operator"]
    if operator.get("id") != OPERATOR_ID:
        raise RuntimeError("unexpected local-first hybrid operator")

    frozen = experiment["frozen_inputs"]
    source_frozen = design["frozen_evidence"]["corrected_all_nine_result"]
    trace_frozen = design["frozen_evidence"]["trace_summary"]

    expected_paper_count = int(frozen["expected_paper_count"])
    expected_stress_count = int(frozen["expected_stress_count"])
    fixed_total_budget = int(frozen["fixed_total_budget"])
    recovery_budget = int(frozen["recovery_budget_after_stress"])

    source_identity = {
        "source_report_sha256_pass": source_report_sha256
        == str(frozen["corrected_report_sha256"]),
        "source_design_id_pass": source_report.get("design_id") == SOURCE_DESIGN_ID,
        "source_architecture_id_pass": source_report.get("architecture_id")
        == SOURCE_ARCHITECTURE_ID,
        "source_experiment_id_pass": source_report.get("experiment_id")
        == SOURCE_EXPERIMENT_ID,
        "source_verdict_pass": source_report.get("cross_paper_recovery_decision")
        == str(frozen["expected_source_verdict"]),
        "source_paper_count_pass": int(source_report.get("paper_count", -1))
        == expected_paper_count,
        "source_valid_paper_count_pass": int(
            source_report.get("valid_paper_count", -1)
        )
        == expected_paper_count,
        "source_stress_count_pass": int(
            source_report.get("valid_stress_case_count", -1)
        )
        == expected_stress_count,
        "source_scientific_state_materialized_false": source_report.get(
            "scientific_state_materialized"
        )
        is False,
        "source_production_routing_changed_false": source_report.get(
            "production_routing_changed"
        )
        is False,
        "source_qwen_verify_changed_false": source_report.get("qwen_verify_changed")
        is False,
        "source_wave7b_started_false": source_report.get("wave7b_started") is False,
        "source_semantic_control_or_ontology_output_false": source_report.get(
            "semantic_control_or_ontology_output"
        )
        is False,
        "source_rejected_mechanism_rescue_false": source_report.get(
            "rejected_mechanism_rescue"
        )
        is False,
    }

    source_counts = {
        "local_exact_recovery_count": int(
            source_report.get("local_exact_removed_proposition_recovery_count", -1)
        ),
        "allocation_exact_recovery_count": int(
            source_report.get(
                "allocation_exact_removed_proposition_recovery_count", -1
            )
        ),
        "local_only_exact_recovery_count": int(
            source_report.get("local_only_exact_recovery_count", -1)
        ),
        "allocation_only_exact_recovery_count": int(
            source_report.get("allocation_only_exact_recovery_count", -1)
        ),
        "both_exact_recovery_count": int(
            source_report.get("both_exact_recovery_count", -1)
        ),
        "neither_exact_recovery_count": int(
            source_report.get("neither_exact_recovery_count", -1)
        ),
        "allocation_depth_exact_recovery_count": int(
            source_report.get(
                "allocation_exact_recoveries_from_depth_extension_count", -1
            )
        ),
    }
    expected_source_counts = {
        "local_exact_recovery_count": int(
            source_frozen["local_exact_recovery_count"]
        ),
        "allocation_exact_recovery_count": int(
            source_frozen["allocation_exact_recovery_count"]
        ),
        "local_only_exact_recovery_count": int(
            source_frozen["local_only_exact_recovery_count"]
        ),
        "allocation_only_exact_recovery_count": int(
            source_frozen["allocation_only_exact_recovery_count"]
        ),
        "both_exact_recovery_count": int(source_frozen["both_exact_recovery_count"]),
        "neither_exact_recovery_count": int(
            source_frozen["neither_exact_recovery_count"]
        ),
        "allocation_depth_exact_recovery_count": int(
            source_frozen["allocation_depth_exact_recovery_count"]
        ),
    }
    source_counts_pass = source_counts == expected_source_counts

    papers = [dict(row) for row in source_report.get("papers") or []]
    actions: list[dict[str, Any]] = []
    action_sources: list[tuple[dict[str, Any], dict[str, Any]]] = []
    hybrid_candidate_count = 0
    hybrid_candidate_raw_chars = 0
    hybrid_selected_raw_chars = 0
    stressed_budget_pass = True

    # Selection pass: do not read removed proposition identity, exact-recovery
    # flags, historical evidence pages, or any other evaluation oracle here.
    for paper in papers:
        item_id = str(paper["item_id"])
        for stress in paper.get("stresses") or []:
            local = dict(stress["local"])
            allocation = dict(stress["allocation"])
            action = _choose_action(local=local, allocation=allocation)

            stressed_count = int(stress.get("stressed_selected_count", -1))
            stressed_budget_pass = stressed_budget_pass and (
                stressed_count == fixed_total_budget - recovery_budget
            )

            hybrid_candidate_count += int(local["generated_candidate_count"])
            hybrid_candidate_raw_chars += int(local["candidate_raw_source_chars"])
            if action["allocation_invoked"]:
                hybrid_candidate_count += int(allocation["candidate_count"])
                hybrid_candidate_raw_chars += int(
                    allocation["candidate_raw_source_chars"]
                )
            hybrid_selected_raw_chars += int(action["selected_raw_source_chars"])

            resulting_budget = stressed_count + int(
                action["selected_proposition_id"] is not None
            )
            actions.append(
                {
                    "item_id": item_id,
                    "stress_index": int(stress["stress_index"]),
                    **action,
                    "stressed_selected_count": stressed_count,
                    "resulting_selected_count": resulting_budget,
                }
            )
            action_sources.append((stress, actions[-1]))

    local_selected_case_count = sum(
        int(row["selected_arm"] == "LOCAL") for row in actions
    )
    allocation_fallback_case_count = sum(
        int(row["allocation_invoked"]) for row in actions
    )
    no_candidate_case_count = sum(
        int(row["selected_arm"] == "NONE") for row in actions
    )

    one_slot_budget_pass = bool(
        stressed_budget_pass
        and all(
            int(row["resulting_selected_count"])
            in {fixed_total_budget - recovery_budget, fixed_total_budget}
            for row in actions
        )
        and all(
            int(row["resulting_selected_count"]) == fixed_total_budget
            for row in actions
            if row["selected_proposition_id"] is not None
        )
    )

    trace_counts_pass = bool(
        len(actions) == expected_stress_count
        and local_selected_case_count
        == int(trace_frozen["local_eligible_case_count"])
        and allocation_fallback_case_count
        == int(trace_frozen["local_first_fallback_invocation_count"])
        and no_candidate_case_count
        == int(trace_frozen["local_empty_case_count"])
        - int(trace_frozen["local_first_fallback_invocation_count"])
    )
    trace_cost_pass = bool(
        hybrid_candidate_count == int(trace_frozen["local_first_candidate_count"])
        and hybrid_candidate_raw_chars
        == int(trace_frozen["local_first_candidate_raw_source_chars"])
        and hybrid_selected_raw_chars
        == int(trace_frozen["local_first_selected_raw_source_chars"])
    )

    # Evaluation pass: teacher/oracle fields become visible only after every
    # action has already been fixed above.
    posthoc_evaluation: list[dict[str, Any]] = []
    hybrid_exact_recovery_count = 0
    local_exact_preserved = 0
    allocation_only_captured = 0
    source_local_exact_seen = 0
    source_allocation_only_seen = 0
    for stress, action in action_sources:
        removed_id = str(stress["removed_proposition_id"])
        local_exact = bool(stress["local"]["exact_removed_proposition_recovery"])
        allocation_exact = bool(
            stress["allocation"]["exact_removed_proposition_recovery"]
        )
        hybrid_exact = action["selected_proposition_id"] == removed_id
        source_local_exact_seen += int(local_exact)
        allocation_only = allocation_exact and not local_exact
        source_allocation_only_seen += int(allocation_only)
        hybrid_exact_recovery_count += int(hybrid_exact)
        local_exact_preserved += int(local_exact and hybrid_exact)
        allocation_only_captured += int(allocation_only and hybrid_exact)
        posthoc_evaluation.append(
            {
                "item_id": action["item_id"],
                "stress_index": action["stress_index"],
                "removed_proposition_id": removed_id,
                "source_local_exact_recovery": local_exact,
                "source_allocation_exact_recovery": allocation_exact,
                "hybrid_exact_recovery": hybrid_exact,
            }
        )

    selection_trace_teacher_blind = all(
        not any(
            token in key
            for token in ("removed", "exact", "historical", "teacher", "oracle")
        )
        for row in actions
        for key in row
    )
    source_exact_totals_pass = bool(
        source_local_exact_seen == expected_source_counts["local_exact_recovery_count"]
        and source_allocation_only_seen
        == expected_source_counts["allocation_only_exact_recovery_count"]
    )

    allocation_everywhere_candidates = int(
        trace_frozen["allocation_everywhere_candidate_count"]
    )
    allocation_everywhere_chars = int(
        trace_frozen["allocation_everywhere_candidate_raw_source_chars"]
    )
    cost = {
        "hybrid_candidate_count_examined": hybrid_candidate_count,
        "hybrid_candidate_raw_source_chars_examined": hybrid_candidate_raw_chars,
        "hybrid_selected_raw_source_chars": hybrid_selected_raw_chars,
        "allocation_everywhere_candidate_count": allocation_everywhere_candidates,
        "allocation_everywhere_candidate_raw_source_chars": allocation_everywhere_chars,
        "candidate_count_ratio_vs_allocation_everywhere": (
            hybrid_candidate_count / allocation_everywhere_candidates
        ),
        "candidate_char_ratio_vs_allocation_everywhere": (
            hybrid_candidate_raw_chars / allocation_everywhere_chars
        ),
    }

    hard_safety = {
        **source_identity,
        "source_counts_pass": source_counts_pass,
        "source_exact_totals_replayed_pass": source_exact_totals_pass,
        "selection_trace_teacher_blind": selection_trace_teacher_blind,
        "one_slot_budget_invariant_pass": one_slot_budget_pass,
        "trace_case_counts_pass": trace_counts_pass,
        "trace_cost_pass": trace_cost_pass,
        "removed_proposition_used_for_selection": False,
        "exact_recovery_flags_used_for_selection": False,
        "historical_page_oracle_used_for_selection": False,
        "teacher_used_for_selection": False,
        "cross_arm_score_used": False,
        "threshold_search_used": False,
        "new_query_generation": False,
        "new_model_or_ranker": False,
        "two_slot_budget_increase": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
    }

    safe = bool(
        all(source_identity.values())
        and source_counts_pass
        and source_exact_totals_pass
        and selection_trace_teacher_blind
        and one_slot_budget_pass
        and trace_counts_pass
        and trace_cost_pass
    )
    decision = _verdict(
        safe=safe,
        expected_local_exact=int(source_frozen["local_exact_recovery_count"]),
        expected_allocation_only_exact=int(
            source_frozen["allocation_only_exact_recovery_count"]
        ),
        local_exact_preserved=local_exact_preserved,
        allocation_only_captured=allocation_only_captured,
        hybrid_exact_recovery_count=hybrid_exact_recovery_count,
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_FIRST_ALLOCATION_FALLBACK_REALIZATION",
        "design_id": DESIGN_ID,
        "architecture_id": ARCHITECTURE_ID,
        "experiment_id": EXPERIMENT_ID,
        "operator_id": OPERATOR_ID,
        "source_report_sha256": source_report_sha256,
        **decision,
        "paper_count": len(papers),
        "stress_case_count": len(actions),
        "hybrid_exact_recovery_count": hybrid_exact_recovery_count,
        "known_local_exact_recoveries_preserved_count": local_exact_preserved,
        "known_allocation_only_recoveries_captured_count": allocation_only_captured,
        "hybrid_local_selected_case_count": local_selected_case_count,
        "hybrid_allocation_fallback_case_count": allocation_fallback_case_count,
        "hybrid_no_candidate_case_count": no_candidate_case_count,
        "source_counts": source_counts,
        "cost": cost,
        "actions": actions,
        "posthoc_evaluation": posthoc_evaluation,
        "hard_safety": hard_safety,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
    }
