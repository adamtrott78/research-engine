import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CASES = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "cases"
)

EXPECTED = {
    "scan-mixed-quality-v1": "scan-mixed-quality-v1.json",
    "semantic-visuals-v1": "semantic-visuals-v1.json",
    "malformed-layout-v1": "malformed-layout-v1.json",
}


def _case(case_id):
    return json.loads((CASES / EXPECTED[case_id]).read_text(encoding="utf-8"))


def test_exactly_required_new_residual_cases_are_registered():
    loaded = {case_id: _case(case_id) for case_id in EXPECTED}

    assert set(loaded) == {
        "scan-mixed-quality-v1",
        "semantic-visuals-v1",
        "malformed-layout-v1",
    }
    for case_id, data in loaded.items():
        assert data["case_id"] == case_id
        assert data["governing_contract"] == "../residual-escalation-contract.v1.json"


def test_source_hashes_are_not_fabricated_before_bounded_prefetch():
    for case_id in EXPECTED:
        data = _case(case_id)
        source = data["source"]
        assert source["authoritative_sha256"] is None
        assert source["authoritative_sha256_status"] == (
            "PENDING_BOUNDED_LAMBDA_PREFETCH"
        )
        assert source["pdf_url"].startswith("https://")


def test_scan_case_has_distinct_textless_and_low_confidence_targets():
    data = _case("scan-mixed-quality-v1")
    residuals = {
        page["intended_residual"]
        for page in data["target_pages"]
    }

    assert residuals == {
        "SCAN_IMAGE_ONLY_OR_TEXTLESS_PAGE",
        "LOW_CONFIDENCE_TEXT_OR_OCR_RECOVERY",
    }
    assert any(
        "Production opener must independently confirm"
        in page["acceptance_on_lambda"]
        for page in data["target_pages"]
    )


def test_semantic_visual_case_has_chart_and_relational_diagram():
    data = _case("semantic-visuals-v1")
    regions = data["target_regions"]

    assert [region["page_index_zero_based"] for region in regions] == [7, 3]
    assert {region["content_type"] for region in regions} == {
        "quantitative multi-panel line chart",
        "relational network-architecture diagram",
    }
    assert all(
        region["answer_status"] == "PENDING_HUMAN_REGION_ANNOTATION"
        for region in regions
    )


def test_layout_case_must_fail_closed_if_native_opener_does_not_fire():
    data = _case("malformed-layout-v1")
    policy = data["replacement_policy"]

    assert policy["fail_closed"] is True
    assert policy["threshold_relaxation_for_case_selection"] is False
    assert policy["model_inference_before_opener"] is False
    assert "Select a replacement" in policy["action_if_no_target_page_opens"]


def test_case_manifests_do_not_open_model_execution():
    for case_id in EXPECTED:
        data = _case(case_id)
        assert data["model_gate"].startswith("No ")
        assert any(
            "PDF bytes" in statement or "PDF bytes" in statement
            for statement in data["authority_and_transport"]
        )
