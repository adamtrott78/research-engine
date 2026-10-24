from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-composed-acquisition-preservation-design.v1.json"
)


def test_composition_design_is_corrected_and_frozen() -> None:
    design = json.loads(DESIGN.read_text())
    assert design["design_id"] == "local-extract-composed-acquisition-preservation-v1"
    assert design["status"] == "DESIGN_CORRECTED_REVIEW_REQUIRED_NO_IMPLEMENTATION_AUTHORIZED"
    experiment = design["selected_discriminating_experiment"]
    assert experiment["id"] == "PLANNER_ALIGNMENT_COMPOSITION_DIAGNOSTIC_V1"
    assert experiment["frozen_inputs"]["facet_planner_budget"] == 28
    assert experiment["composition_operator"]["id"] == (
        "EXACT_PLANNER28_THEN_VALIDATED_ALIGNMENT_SWAP_V1"
    )
    assert experiment["composition_operator"]["no_new_alignment_search"] is True
    assert experiment["composition_operator"]["no_threshold_search"] is True

    outcomes = set(experiment["frozen_interpretation"])
    assert outcomes == {
        "COMPOSED_COMPLEMENTARITY_ROBUST",
        "COMPOSED_COMPLEMENTARITY_FRAGILE",
        "COMPOSITION_PARTIAL",
        "COMPOSITION_NEGATIVE",
        "COMPOSITION_COLLISION",
        "INVALID_EXECUTION",
    }
    correction = design["review_correction"]
    assert correction["selector_changed"] is False
    assert correction["composition_operator_changed"] is False
    assert correction["frozen_inputs_changed"] is False
    assert correction["budget_changed"] is False
    assert correction["teacher_boundary_changed"] is False
    assert correction["metrics_changed"] is False
    assert correction["interpretation_table_changed"] is True
    assert correction["implementation_started_before_correction"] is False
