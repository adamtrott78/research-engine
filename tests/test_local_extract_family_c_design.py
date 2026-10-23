from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-family-c-iterative-residual-design.v1.json"
)


def _load() -> dict:
    return json.loads(DESIGN.read_text())


def test_family_c_design_preserves_composition_negative() -> None:
    design = _load()
    exclusion = design["authoritative_exclusion_evidence"]
    assert exclusion["workflow_run_id"] == 40032590597
    assert exclusion["decision"] == "COMPOSITION_NEGATIVE"


def test_family_c_first_diagnostic_is_zero_generation_and_lexicographic() -> None:
    design = _load()
    selected = design["selected_first_diagnostic"]
    assert selected["id"] == "ZERO_LOSS_RESIDUAL_AUGMENTATION_DIAGNOSTIC_V1"
    assert selected["new_model_calls"] == 0
    assert selected["new_model_downloads"] == 0
    algorithm = selected["algorithm"]
    assert algorithm["id"] == "LEXICOGRAPHIC_ZERO_LOSS_SWAP_V1"
    assert algorithm["teacher_blind"] is True
    assert algorithm["final_budget"] == 24
    assert algorithm["fixed_swap_limit"] is None
    assert algorithm["no_post_result_tuning"] is True
    assert "U_C(new_set) >= U_C(baseline_set)" in algorithm["feasibility"]


def test_family_c_gap_judge_is_contingent_not_authorized() -> None:
    design = _load()
    families = {row["id"]: row for row in design["architecture_families"]}
    controller = families["FAMILY_C2_STRUCTURED_SCIENTIFIC_GAP_JUDGE"][
        "controller_contract"
    ]
    assert controller["maximum_gap_items"] == 4
    assert "strict schema-constrained JSON" in controller["safeguards"]
    assert design["boundaries"]["qwen_gap_judge_execution_authorized"] is False
    assert design["boundaries"]["trained_critic_authorized"] is False


def test_family_c_does_not_promote_quota_or_ontology() -> None:
    design = _load()
    rejected = {row["id"]: row for row in design["rejected_or_deferred"]}
    assert rejected["PROMOTE_FIXED_20_4_QUOTA"]["status"] == "FORBIDDEN_POST_HOC"
    assert rejected["KNOWLEDGE_GRAPH_OR_ONTOLOGY_CONTROLLER"]["status"] == "OUT_OF_SCOPE"
    boundaries = design["boundaries"]
    assert boundaries["ontology_or_semantic_control_in_extract"] is False
    assert boundaries["production_standard_codex_extract_changed"] is False
    assert boundaries["accepted_qwen_verify_changed"] is False
    assert boundaries["wave7b"] is False


def test_family_c_positive_gate_requires_full_control_retention() -> None:
    design = _load()
    gates = design["selected_first_diagnostic"]["predeclared_interpretation"]
    assert any(
        "retains all 7/7" in item
        for item in gates["PROTECTED_AUGMENT_STRONG"]
    )
    assert any(
        "retains all 7/7" in item
        for item in gates["PROTECTED_AUGMENT_FEASIBLE"]
    )
