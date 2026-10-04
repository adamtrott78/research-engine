import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SELECTION = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "residual-candidate-selection.v1.json"
)


def _selection():
    return json.loads(SELECTION.read_text(encoding="utf-8"))


def test_candidate_selection_is_benchmark_only():
    data = _selection()

    assert data["status"] == "SELECTED_FOR_BENCHMARK_NOT_INSTALLED"
    assert data["promotion_boundary"]["no_production_change_from_selection"] is True
    assert data["promotion_boundary"]["no_localization_matrix_change_yet"] is True
    assert data["promotion_boundary"]["no_job_tier_change"] is True
    assert data["promotion_boundary"]["no_daemon_lifecycle_change"] is True
    assert data["promotion_boundary"]["no_scientific_state_materialization"] is True


def test_minimum_candidate_set_is_exactly_three_primary_families():
    data = _selection()
    candidates = data["candidates"]

    assert [row["id"] for row in candidates] == [
        "paddleocr-pp-ocrv6-medium",
        "paddleocr-vl-1.6",
        "qwen3.5-4b",
    ]
    assert [row["priority"] for row in candidates] == [1, 2, 3]


def test_ocr_candidate_is_residual_scoped_not_global():
    data = _selection()
    row = data["candidates"][0]

    assert row["role"] == "PRIMARY_TARGETED_OCR_LAYOUT_RECOVERY"
    assert row["runtime_selection"]["downloads_now"] is False
    assert "healthy born-digital pages" in row["forbidden_uses"]
    assert "whole-corpus OCR rerun" in row["forbidden_uses"]
    assert row["resource_policy"]["target_peak_vram_mib_lte"] <= 8192


def test_document_vlm_is_pinned_and_region_scoped():
    data = _selection()
    row = data["candidates"][1]

    assert row["id"] == "paddleocr-vl-1.6"
    assert row["pinned_revision"] == (
        "c5630abae1d940eafe0697512a0325494b02ab42"
    )
    assert row["runtime_selection"]["downloads_now"] is False
    assert "every-page VLM parsing" in row["forbidden_uses"]
    assert row["license_review"]["model_license"] == "Apache-2.0"
    assert row["resource_policy"]["target_peak_vram_mib_lte"] <= 14336


def test_general_semantic_candidate_is_bounded_and_pinned():
    data = _selection()
    row = data["candidates"][2]

    assert row["id"] == "qwen3.5-4b"
    assert row["pinned_revision"] == (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert row["runtime_selection"]["downloads_now"] is False
    assert row["license_review"]["model_license"] == "Apache-2.0"
    assert row["resource_policy"]["peak_vram_mib_lte"] <= 16384
    assert any(
        "<=4 authoritative PDF pages" in value
        for value in row["semantic_constraints"]
    )
    assert any(
        "radius-0 cited authoritative page" in value
        for value in row["semantic_constraints"]
    )


def test_visual_fallback_does_not_duplicate_general_vlm_by_default():
    data = _selection()
    row = data["candidates"][2]

    assert "PaddleOCR-VL-1.6 fails" in row["visual_fallback_condition"]
    sequence = "\n".join(data["runtime_sequence"])
    assert "do not run both by default" in sequence


def test_deferred_models_cannot_sneak_into_initial_runtime_sequence():
    data = _selection()
    deferred = {
        row["candidate"]
        for row in data["rejected_or_deferred_alternatives"]
        if row["decision"] == "DEFER"
    }
    sequence = "\n".join(data["runtime_sequence"])

    assert "Qwen/Qwen3-VL-8B-Instruct-FP8" in deferred
    assert "Qwen/Qwen3-8B-AWQ" in deferred
    assert "Qwen/Qwen3-VL-8B-Instruct-FP8" not in sequence
    assert "Qwen/Qwen3-8B-AWQ" not in sequence


def test_model_downloads_wait_for_missing_source_case_manifests():
    data = _selection()

    assert data["runtime_sequence"][0] == (
        "Do not download or install any selected model until the three missing "
        "source-case manifests are durable."
    )
