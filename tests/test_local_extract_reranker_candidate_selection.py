from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-proposition-reranker-candidate-selection.v1.json"
)


def test_candidate_selection_is_minimal_and_capability_distinct() -> None:
    contract = json.loads(CONTRACT.read_text())
    models = contract["candidate_models"]
    assert [row["id"] for row in models] == [
        "BGE_M3",
        "QWEN3_RERANKER_0_6B",
    ]
    assert contract["execution_boundaries"]["maximum_candidate_model_families_after_durable_approval"] == 2
    assert contract["execution_boundaries"]["load_models_sequentially"] is True

    bge_arms = {
        row["id"]: row
        for row in models[0]["one_model_load_produces_arms"]
    }
    assert set(bge_arms) == {
        "BGE_M3_DENSE",
        "BGE_M3_SPARSE",
        "BGE_M3_COLBERT",
        "BGE_M3_HYBRID_040_020_040",
    }
    assert bge_arms["BGE_M3_HYBRID_040_020_040"][
        "weights_frozen_from_official_example"
    ] == [0.4, 0.2, 0.4]


def test_bakeoff_keeps_candidate_pool_and_budget_fixed() -> None:
    contract = json.loads(CONTRACT.read_text())
    protocol = contract["scoring_protocol"]
    assert protocol["candidate_pool"] == (
        "all exact propositions on pages reached by current top-12 coarse regions"
    )
    assert protocol["channels"] == [
        "global",
        "facet:method",
        "facet:dataset",
        "facet:evaluation",
        "facet:robustness",
    ]
    assert "top_k=12" in protocol["selection"]
    assert any("no changing top_k" == item for item in protocol["no_tuning_after_results"])
    assert any(
        "same maximum 12 selected propositions per paper" == item
        for item in contract["fixed_controls"]
    )


def test_semantic_support_is_additive_not_metric_replacement() -> None:
    contract = json.loads(CONTRACT.read_text())
    semantic = contract["evaluation"]["semantic_support_track"]
    design = semantic["design"]
    assert any(
        "frozen before any neural ranking output exists" in item
        for item in design
    )
    assert any("atom-support recall@12" in item for item in design)
    assert any(
        "Do not replace or retune the legacy lexical metric" in item
        for item in design
    )
    assert any("SciFact-style" in item for item in design)


def test_no_model_download_or_wave7_is_authorized_by_selection_checkpoint() -> None:
    contract = json.loads(CONTRACT.read_text())
    bounds = contract["execution_boundaries"]
    assert bounds["model_downloads_authorized_by_this_contract"] is False
    assert bounds["production_extract_change_authorized"] is False
    assert bounds["qwen_verify_change_authorized"] is False
    assert bounds["wave7_authorized"] is False
    assert bounds["generation_model_family_bakeoff_authorized"] is False
    assert bounds["gitrunner_substrate_change_authorized"] is False


def test_material_gain_is_measured_against_wave6_to_available_ceiling() -> None:
    contract = json.loads(CONTRACT.read_text())
    baseline = contract["motivating_result"]
    assert (
        baseline["wave6_grouped_mean_f1"]
        < baseline["selected_page_pool_grouped_ceiling_f1"]
    )
    assert (
        baseline["wave6_atom_mean_f1"]
        < baseline["selected_page_pool_atom_ceiling_f1"]
    )
    interpretation = contract["predeclared_interpretation"]
    assert "25%" in interpretation["material_gain"]
    assert "both grouped and atomized" in interpretation["material_gain"]
    assert "at least two of the three" in interpretation["case_recovery"]
