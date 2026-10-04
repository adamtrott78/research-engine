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


def test_source_hashes_are_pinned_after_bounded_lambda_audit():
    expected = {
        "scan-mixed-quality-v1": (
            "3948255a723380d8844bab31c60298cee00da9334efa377f1d596b306792b6a1",
            28,
        ),
        "semantic-visuals-v1": (
            "1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a",
            12,
        ),
        "malformed-layout-v1": (
            "f2630182ccfabd6fc89ac20270b4daf15a0ca849298a7db1e5d124ca443051f1",
            14,
        ),
    }
    for case_id in EXPECTED:
        data = _case(case_id)
        source = data["source"]
        assert source["authoritative_sha256"] == expected[case_id][0]
        assert source["authoritative_sha256_status"] == (
            "PINNED_FROM_BOUNDED_LAMBDA_SOURCE_AUDIT"
        )
        assert source["authoritative_page_count"] == expected[case_id][1]
        assert source["pdf_url"].startswith("https://")
        assert data["bounded_source_audit"]["decision"] == "PASS"


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
    assert all(region["locator_status"] == "LINKED_FIGURE" for region in regions)
    assert all(len(region["pdf_bbox"]) == 4 for region in regions)
    assert data["bounded_source_audit"]["region_locators_ready_for_model_prefetch"] is True


def test_layout_case_must_fail_closed_if_native_opener_does_not_fire():
    data = _case("malformed-layout-v1")
    policy = data["replacement_policy"]

    assert policy["fail_closed"] is True
    assert policy["threshold_relaxation_for_case_selection"] is False
    assert policy["model_inference_before_opener"] is False
    assert "Select a replacement" in policy["action_if_no_target_page_opens"]
    assert data["bounded_source_audit"]["replacement_required"] is False
    assert all(page["audit_evidence"]["layout_opener_fired"] for page in data["target_pages"])


def test_case_manifests_do_not_open_model_execution():
    for case_id in EXPECTED:
        data = _case(case_id)
        assert "bounded model prefetch may proceed" in data["model_gate"]
        assert "inference remains closed" in data["model_gate"]
        assert any(
            "PDF bytes" in statement or "PDF bytes" in statement
            for statement in data["authority_and_transport"]
        )
