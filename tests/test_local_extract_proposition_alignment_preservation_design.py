from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN = ROOT / "benchmarks" / "document_intelligence" / "local-extract-proposition-alignment-preservation-design.v1.json"


def _load() -> dict:
    return json.loads(DESIGN.read_text())


def test_alignment_design_is_materially_different_and_bounded() -> None:
    design = _load()
    selected = design["selected_first_diagnostic"]
    algorithm = selected["algorithm"]
    assert selected["id"] == "PROPOSITION_ALIGNMENT_CERTIFIED_AUGMENTATION_DIAGNOSTIC_V1"
    assert algorithm["id"] == "DYNAMIC_SAME_PAGE_ALIGNMENT_SWAP_V1"
    assert algorithm["alignment_threshold"] == 0.40
    assert algorithm["certificate_scope"] == "same source page only"
    assert algorithm["dynamic_recompute"] is True
    assert algorithm["final_budget"] == 24
    assert algorithm["teacher_blind"] is True
    assert selected["new_model_calls"] == 0
    assert selected["new_model_downloads"] == 0


def test_alignment_design_preserves_rejected_evidence_and_boundaries() -> None:
    design = _load()
    names = {row["result"] for row in design["exclusion_evidence"]}
    assert {"WAVE7A3_REJECT", "COMPOSITION_NEGATIVE", "PROTECTED_AUGMENT_NEGATIVE"} <= names
    boundaries = design["boundaries"]
    assert boundaries["wave7b"] is False
    assert boundaries["production_standard_codex_extract_changed"] is False
    assert boundaries["accepted_qwen_verify_changed"] is False
    assert boundaries["new_model_training_authorized"] is False
    assert boundaries["new_model_download_authorized"] is False
    assert boundaries["ontology_or_semantic_control_in_extract"] is False
    assert boundaries["fixed_quota_promotion"] is False


def test_alignment_threshold_was_source_only_sparse_calibration() -> None:
    calibration = _load()["source_only_calibration"]
    assert calibration["proposition_count"] == 216
    assert calibration["threshold"] == 0.40
    assert calibration["certified_count"] == 20
    assert abs(calibration["certified_fraction"] - (20 / 216)) < 1e-12
    assert sum(calibration["per_paper_certified_counts"].values()) == 20
