import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "residual-source-audit-contract.v1.json"
)


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_source_audit_contract_is_pre_model_and_bounded():
    data = _contract()
    required = data["required_state"]

    assert data["status"] == "BOUNDED_SOURCE_AUDIT_ONLY"
    assert required["research_daemon"] == "inactive/dead PID 0 before, throughout, and after"
    assert required["network_use"] == "SOURCE_PDF_DOWNLOADS_ONLY"
    assert required["new_candidate_model_downloads"] == 0
    assert required["new_candidate_model_inference_calls"] == 0
    assert required["scientific_state_materialized"] is False


def test_source_audit_contract_covers_exactly_three_registered_cases():
    data = _contract()

    assert data["source_cases"] == [
        "scan-mixed-quality-v1",
        "semantic-visuals-v1",
        "malformed-layout-v1",
    ]


def test_source_audit_contract_uses_only_existing_docling_tiers():
    data = _contract()

    assert data["allowed_inference"] == [
        "Docling native model-free PDF parse",
        (
            "Docling standard_no_ocr using the already-installed validated "
            "layout/TableFormer cache only"
        ),
    ]


def test_source_audit_contract_fails_closed_before_candidate_prefetch():
    data = _contract()
    forbidden = "\n".join(data["forbidden_actions"])
    layout = data["deterministic_audit_acceptance"]["malformed_layout_v1"]

    assert "install or download any newly selected" in forbidden
    assert "invoke Codex or any semantic model" in forbidden
    assert "relax residual opener thresholds" in forbidden
    assert any("decision is REJECT" in value for value in layout)
    assert "separate bounded runtime/model prefetch gate" in data["next_on_pass"]


def test_source_audit_thresholds_match_residual_contract():
    data = _contract()
    thresholds = data["opener_thresholds"]

    assert thresholds["scan_native_nonspace_characters_lt"] == 32
    assert thresholds["scan_figure_bbox_page_coverage_gte"] == 0.50
    assert thresholds["low_confidence_replacement_control_ratio_gt"] == 0.01
    assert thresholds["low_confidence_alphabetic_token_ratio_lt"] == 0.45
    assert thresholds["low_confidence_native_standard_token_f1_lt"] == 0.80
    assert thresholds["layout_duplicate_block_ratio_gt"] == 0.20
    assert thresholds["layout_max_text_block_overlap_fraction_gte"] == 0.50
    assert thresholds["layout_native_standard_token_f1_gte"] == 0.85
    assert thresholds["layout_native_standard_ordered_bigram_f1_lt"] == 0.70
    assert thresholds["layout_interleaved_column_switches_gte"] == 3
