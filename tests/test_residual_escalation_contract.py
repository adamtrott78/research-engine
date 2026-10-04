import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "residual-escalation-contract.v1.json"
)

EXPECTED_RESIDUALS = {
    "SCAN_IMAGE_ONLY_OR_TEXTLESS_PAGE",
    "LOW_CONFIDENCE_TEXT_OR_OCR_RECOVERY",
    "FORMULA_RESIDUAL_AFTER_CODEFORMULAV2",
    "TABLE_RESIDUAL_AFTER_FAST_ACCURATE_PADDLE",
    "UNCAPTIONED_OR_SEMANTICALLY_RICH_FIGURE_CHART",
    "LAYOUT_OR_READING_ORDER_ANOMALY",
    "SEMANTIC_EXTRACT_AMBIGUITY",
    "SEMANTIC_VERIFY_AMBIGUITY",
}


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_residual_contract_covers_required_families_once():
    data = _contract()
    rows = data["residual_families"]
    ids = [row["id"] for row in rows]

    assert set(ids) == EXPECTED_RESIDUALS
    assert len(ids) == len(set(ids))


def test_residual_contract_locks_authoritative_pdf_escalation_ladder():
    data = _contract()
    tiers = [row["id"] for row in data["escalation_ladder"]]

    assert tiers == [
        "DETERMINISTIC_NATIVE_STRUCTURE",
        "VALIDATED_SPECIALIZED_LOCAL",
        "TARGETED_LOCAL_RECOVERY",
        "REMOTE_CODEX_AUTHORITATIVE_PDF",
        "HUMAN_REVIEW",
    ]
    invariants = "\n".join(data["authority_invariants"])
    assert "original content-addressed PDF remains scientific authority" in invariants
    assert "EXTRACT and VERIFY remain STANDARD Codex jobs" in invariants
    assert "broad parser bakeoff remains closed" in invariants


def test_every_residual_has_opening_quality_provenance_failure_resource_and_control_gates():
    data = _contract()

    for row in data["residual_families"]:
        assert row["opener"]["production_signal"]
        assert row["opener"]["oracle_forbidden_in_routing"] is True
        assert row["cheapest_eligible_local_recovery"]
        assert row["quality_gate"]
        assert row["provenance"]
        assert "source_pdf_sha256" in row["provenance"]
        assert row["fallthrough"]
        assert row["resource_ceiling"]
        assert row["semantic_control"]
        assert row["candidate_family_after_contract"]


def test_new_model_resource_envelope_fits_one_a4500_with_headroom():
    data = _contract()
    limits = data["global_resource_envelope"]["new_model_owner_limits"]

    assert limits["max_single_owner_peak_vram_mib"] <= 16384
    assert limits["max_single_owner_peak_rss_mib"] <= 32768
    assert limits["max_cold_model_load_seconds"] <= 120
    assert limits["max_pending_per_family"] == 8
    assert limits["duplicate_model_owners_per_family"] == 1
    assert limits["remote_api_fallback_allowed"] is False
    assert limits["model_downloads_during_promotion_gate"] is False


def test_semantic_accounting_keeps_tokens_and_plan_allowance_separate():
    data = _contract()
    accounting = data["semantic_accounting_contract"]

    assert "input_tokens" in accounting["exact_usage_fields"]
    assert "cached_input_tokens" in accounting["exact_usage_fields"]
    assert "reasoning_output_tokens" in accounting["exact_usage_fields"]
    assert "censored empirical observation" in accounting["allowance_observation"]
    assert "never infer exact allowance savings from token counts" in (
        accounting["allowance_observation"]
    )


def test_minimum_new_corpus_is_three_targeted_sources_not_new_bakeoff():
    data = _contract()
    cases = data["minimum_additional_benchmark_cases"]
    new_sources = [row for row in cases if row["new_source_required"]]

    assert [row["case_id"] for row in new_sources] == [
        "scan-mixed-quality-v1",
        "semantic-visuals-v1",
        "malformed-layout-v1",
    ]
    assert any(
        row["case_id"] == "existing-formula-table-residuals"
        and row["new_source_required"] is False
        for row in cases
    )
    assert any(
        row["case_id"] == "preserved-semantic-ambiguity-set"
        and row["new_source_required"] is False
        for row in cases
    )


def test_concrete_new_model_selection_is_closed_until_contract_is_durable():
    data = _contract()
    gate = data["candidate_selection_gate"]

    assert gate["status"] == "CLOSED_UNTIL_THIS_CONTRACT_IS_DURABLE"
    assert gate["concrete_models_named_here"] is False
    assert len(gate["families_to_consider_after_checkpoint"]) == 3
    exclusions = "\n".join(gate["exclusions"])
    assert "no global OCR" in exclusions
    assert "no VLM on every page" in exclusions
    assert "no model/runtime installation before contract durability" in exclusions


def test_semantic_fallbacks_require_contemporaneous_authoritative_controls():
    data = _contract()
    by_id = {row["id"]: row for row in data["residual_families"]}

    extract = by_id["SEMANTIC_EXTRACT_AMBIGUITY"]
    verify = by_id["SEMANTIC_VERIFY_AMBIGUITY"]

    assert "contemporaneous" in extract["semantic_control"].lower()
    assert "authoritative PDF" in extract["semantic_control"]
    assert extract["quality_gate"]["fabricated_or_unlocatable_evidence_tolerance"] == 0

    assert "contemporaneous" in verify["semantic_control"].lower()
    assert "radius-0" in verify["semantic_control"]
    assert verify["quality_gate"]["verified_decision_agreement_min"] == 1.0
    assert verify["quality_gate"]["accepted_evidence_ids_exact_agreement_min"] == 1.0
