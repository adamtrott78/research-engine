from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-nondestructive-evidence-frontier-design.v1.json"
)


def _load() -> dict:
    return json.loads(DESIGN.read_text())


def test_frontier_design_freezes_residual_first_discriminator() -> None:
    design = _load()
    experiment = design["selected_discriminating_experiment"]
    selector = experiment["selector"]
    assert design["design_id"] == (
        "local-extract-nondestructive-evidence-frontier-v1"
    )
    assert experiment["id"] == "NONDESTRUCTIVE_EVIDENCE_FRONTIER_DIAGNOSTIC_V1"
    assert selector["id"] == "IMMUTABLE_CORE_APPEND_FRONTIER_V1"
    assert selector["diagnostic_budgets"] == [24, 26, 28, 32]
    assert selector["primary_small_frontier_ceiling"] == 28
    assert selector["maximum_diagnostic_ceiling"] == 32
    assert selector["teacher_blind"] is True
    assert selector["no_pairwise_similarity_score"] is True
    assert selector["no_threshold_calibration"] is True
    assert selector["no_post_result_budget_selection"] is True
    assert experiment["new_generation_calls"] == 0
    assert experiment["new_model_downloads"] == 0
    assert experiment["new_model_training"] is False


def test_frontier_design_preserves_positive_and_negative_evidence() -> None:
    design = _load()
    positive = design["preserved_positive_evidence"]
    assert positive["checkpoint"] == "ALIGNMENT_AUGMENT_FEASIBLE"
    assert positive["supported_atoms"] == 8
    assert positive["frozen_control_atoms_retained"] == 7
    assert positive["observed_residual_atoms_recovered"] == ["e6.a"]
    assert set(design["rejected_mechanisms"]) == {
        "NLI_CERTIFICATE_REJECTED",
        "PREDICATE_ARGUMENT_CERTIFICATE_REJECTED",
        "AMR_GRAPH_CERTIFICATE_REJECTED",
    }


def test_frontier_design_has_frozen_interpretation_and_stopping_rule() -> None:
    experiment = _load()["selected_discriminating_experiment"]
    assert set(experiment["frozen_interpretation"]) == {
        "ADDITIVE_FRONTIER_CONFIRMED",
        "ADDITIVE_FRONTIER_PARTIAL",
        "ADDITIVE_FRONTIER_NEGATIVE",
    }
    assert set(experiment["decision_table"]) == {
        "ADDITIVE_FRONTIER_CONFIRMED",
        "ADDITIVE_FRONTIER_PARTIAL",
        "ADDITIVE_FRONTIER_NEGATIVE",
        "OPERATIONAL_FAILURE_BEFORE_SCIENCE",
    }
    assert "Exactly one scientific execution" in experiment["stopping_rule"]
    assert "no production promotion" in (
        experiment["decision_table"]["ADDITIVE_FRONTIER_CONFIRMED"]
    )
