from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-facet-reserved-evidence-planner-design.v1.json"
)


def test_facet_planner_design_preserves_frozen_science() -> None:
    design = json.loads(DESIGN.read_text())
    assert design["design_id"] == "local-extract-facet-reserved-evidence-planner-v1"
    experiment = design["selected_discriminating_experiment"]
    assert experiment["id"] == "SALIENCE_STRATIFIED_RESERVATION_DIAGNOSTIC_V1"
    assert experiment["frozen_inputs"]["retriever"] == "BM25_DIRECT"
    assert experiment["frozen_inputs"]["per_intent_candidate_depth"] == 3
    assert experiment["reservation_schedule"]["diagnostic_budgets"] == [
        24, 26, 28, 32
    ]
    assert experiment["reservation_schedule"]["primary_small_ceiling"] == 28
    assert experiment["reservation_schedule"]["maximum_ceiling"] == 32
    assert experiment["reservation_schedule"][
        "no_teacher_filtering_or_calibration"
    ] is True
    assert design["hard_boundaries"]["teacher_used_for_selection"] is False
    assert design["hard_boundaries"]["new_query_generation"] is False
    assert design["hard_boundaries"]["new_model_or_ranker"] is False
    assert design["hard_boundaries"]["pairwise_semantic_similarity"] is False


def test_invalid_execution_is_not_scientific_negative() -> None:
    design = json.loads(DESIGN.read_text())
    experiment = design["selected_discriminating_experiment"]
    frozen = experiment["frozen_interpretation"]
    assert "INVALID_EXECUTION" in frozen
    negative = " ".join(frozen["FACET_RESERVATION_NEGATIVE"])
    invalid = " ".join(frozen["INVALID_EXECUTION"])
    assert "valid safe replay" in negative
    assert "no scientific verdict" in invalid
    assert design["review_correction"]["other_science_changed"] is False
