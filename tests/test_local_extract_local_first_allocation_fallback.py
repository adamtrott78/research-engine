from __future__ import annotations

from research_engine.local_extract_local_first_allocation_fallback import (
    _choose_action,
    _verdict,
    evaluate_local_first_allocation_fallback,
)


def _candidate(pid: str, raw: str) -> dict:
    return {
        "proposition_id": pid,
        "raw_source_form": raw,
        "page_number": 1,
    }


def test_choose_action_prefers_local_without_consulting_allocation() -> None:
    result = _choose_action(
        local={"chosen": _candidate("local", "aa")},
        allocation={"chosen": _candidate("allocation", "bbb")},
    )
    assert result["selected_arm"] == "LOCAL"
    assert result["selected_proposition_id"] == "local"
    assert result["allocation_invoked"] is False


def test_choose_action_falls_back_only_when_local_empty() -> None:
    result = _choose_action(
        local={"chosen": None},
        allocation={"chosen": _candidate("allocation", "bbb")},
    )
    assert result["selected_arm"] == "ALLOCATION_FALLBACK"
    assert result["selected_proposition_id"] == "allocation"
    assert result["allocation_invoked"] is True


def test_choose_action_can_return_no_candidate() -> None:
    result = _choose_action(
        local={"chosen": None},
        allocation={"chosen": None},
    )
    assert result["selected_arm"] == "NONE"
    assert result["selected_proposition_id"] is None
    assert result["allocation_invoked"] is True


def test_verdict_regression_precedes_realization() -> None:
    result = _verdict(
        safe=True,
        expected_local_exact=3,
        expected_allocation_only_exact=1,
        local_exact_preserved=2,
        allocation_only_captured=1,
        hybrid_exact_recovery_count=3,
    )
    assert result["hybrid_realization_decision"] == "HYBRID_REGRESSION"


def test_verdict_collapse_is_distinct_from_realization() -> None:
    result = _verdict(
        safe=True,
        expected_local_exact=3,
        expected_allocation_only_exact=1,
        local_exact_preserved=3,
        allocation_only_captured=0,
        hybrid_exact_recovery_count=3,
    )
    assert result["hybrid_realization_decision"] == "HYBRID_COLLAPSES_TO_LOCAL"


def _design() -> dict:
    return {
        "design_id": "local-extract-local-first-allocation-fallback-hybrid-v1",
        "selected_architecture": {
            "id": "LOCAL_FIRST_ALLOCATION_FALLBACK_HYBRID_V1"
        },
        "frozen_evidence": {
            "corrected_all_nine_result": {
                "local_exact_recovery_count": 2,
                "allocation_exact_recovery_count": 1,
                "local_only_exact_recovery_count": 2,
                "allocation_only_exact_recovery_count": 1,
                "both_exact_recovery_count": 0,
                "neither_exact_recovery_count": 1,
                "allocation_depth_exact_recovery_count": 0,
            },
            "trace_summary": {
                "local_eligible_case_count": 3,
                "local_empty_case_count": 1,
                "local_first_fallback_invocation_count": 1,
                "local_first_trace_exact_recovery_count": 3,
                "local_first_candidate_count": 10,
                "local_first_candidate_raw_source_chars": 100,
                "local_first_selected_raw_source_chars": 14,
                "allocation_everywhere_candidate_count": 100,
                "allocation_everywhere_candidate_raw_source_chars": 1000,
            },
        },
        "selected_discriminating_experiment": {
            "id": "LOCAL_FIRST_ALLOCATION_FALLBACK_REALIZATION_V1",
            "frozen_inputs": {
                "corrected_report_sha256": "abc",
                "expected_paper_count": 1,
                "expected_stress_count": 4,
                "expected_source_verdict": "COMPLEMENTARY_STRUCTURAL_RECOVERY",
                "recovery_budget_after_stress": 1,
                "fixed_total_budget": 28,
            },
            "hybrid_operator": {
                "id": "LOCAL_FIRST_ELIGIBILITY_FALLBACK_V1"
            },
        },
    }


def _stress(
    index: int,
    *,
    removed: str,
    local_chosen: dict | None,
    allocation_chosen: dict | None,
    local_exact: bool,
    allocation_exact: bool,
    local_count: int,
    local_chars: int,
    allocation_count: int = 0,
    allocation_chars: int = 0,
) -> dict:
    return {
        "stress_index": index,
        "stressed_selected_count": 27,
        "removed_proposition_id": removed,
        "removed_on_historical_evidence_page": False,
        "local": {
            "chosen": local_chosen,
            "generated_candidate_count": local_count,
            "candidate_raw_source_chars": local_chars,
            "exact_removed_proposition_recovery": local_exact,
        },
        "allocation": {
            "chosen": allocation_chosen,
            "candidate_count": allocation_count,
            "candidate_raw_source_chars": allocation_chars,
            "exact_removed_proposition_recovery": allocation_exact,
        },
    }


def _source_report() -> dict:
    stresses = [
        _stress(
            0,
            removed="p0",
            local_chosen=_candidate("p0", "aa"),
            allocation_chosen=_candidate("x0", "x"),
            local_exact=True,
            allocation_exact=False,
            local_count=1,
            local_chars=10,
        ),
        _stress(
            1,
            removed="p1",
            local_chosen=_candidate("p1", "bbb"),
            allocation_chosen=_candidate("x1", "x"),
            local_exact=True,
            allocation_exact=False,
            local_count=2,
            local_chars=20,
        ),
        _stress(
            2,
            removed="p2",
            local_chosen=_candidate("x2", "cccc"),
            allocation_chosen=_candidate("x3", "x"),
            local_exact=False,
            allocation_exact=False,
            local_count=3,
            local_chars=30,
        ),
        _stress(
            3,
            removed="p3",
            local_chosen=None,
            allocation_chosen=_candidate("p3", "ddddd"),
            local_exact=False,
            allocation_exact=True,
            local_count=0,
            local_chars=0,
            allocation_count=4,
            allocation_chars=40,
        ),
    ]
    return {
        "design_id": "local-extract-corrected-all-nine-recovery-comparison-v1",
        "architecture_id": "CORRECTED_ALL_NINE_STRUCTURAL_RECOVERY_COMPARISON_V1",
        "experiment_id": "ALL_NINE_FIXED_BASELINE_RECOVERY_COMPARISON_V1",
        "cross_paper_recovery_decision": "COMPLEMENTARY_STRUCTURAL_RECOVERY",
        "paper_count": 1,
        "valid_paper_count": 1,
        "valid_stress_case_count": 4,
        "local_exact_removed_proposition_recovery_count": 2,
        "allocation_exact_removed_proposition_recovery_count": 1,
        "local_only_exact_recovery_count": 2,
        "allocation_only_exact_recovery_count": 1,
        "both_exact_recovery_count": 0,
        "neither_exact_recovery_count": 1,
        "allocation_exact_recoveries_from_depth_extension_count": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "rejected_mechanism_rescue": False,
        "papers": [{"item_id": "paper:1", "stresses": stresses}],
    }


def test_evaluate_realizes_complementarity_with_teacher_blind_action_trace() -> None:
    report = evaluate_local_first_allocation_fallback(
        source_report=_source_report(),
        source_report_sha256="abc",
        design=_design(),
    )
    assert report["hybrid_realization_decision"] == (
        "HYBRID_COMPLEMENTARITY_REALIZED"
    )
    assert report["hybrid_exact_recovery_count"] == 3
    assert report["known_local_exact_recoveries_preserved_count"] == 2
    assert report["known_allocation_only_recoveries_captured_count"] == 1
    assert report["hybrid_local_selected_case_count"] == 3
    assert report["hybrid_allocation_fallback_case_count"] == 1
    assert report["cost"]["hybrid_candidate_count_examined"] == 10
    assert report["cost"]["hybrid_candidate_raw_source_chars_examined"] == 100
    assert report["cost"]["hybrid_selected_raw_source_chars"] == 14
    assert report["hard_safety"]["selection_trace_teacher_blind"] is True
    for action in report["actions"]:
        assert not any(
            token in key
            for key in action
            for token in ("removed", "exact", "historical", "teacher", "oracle")
        )
