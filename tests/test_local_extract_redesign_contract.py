import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-redesign-contract.v1.json"
)


def load_contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_is_pre_download_and_pinned_to_accepted_baseline():
    payload = load_contract()
    assert payload["schema_version"] == 1
    assert payload["task"] == "research-engine-local-extract-redesign-v1"
    assert payload["status"] == "CONTRACT_ONLY_NO_NEW_MODEL_DOWNLOADS"
    assert (
        payload["baseline_research_engine_sha"]
        == "e5a9b4c361ac790562fac5cf2f35f5a54f2bf58d"
    )


def test_local_evidence_agent_is_hierarchical_and_codex_is_last():
    payload = load_contract()
    architecture = payload["architecture"]
    assert architecture["id"] == "LOCAL_EVIDENCE_AGENT_V1"
    tiers = architecture["stages"]
    assert [stage["tier"] for stage in tiers] == list(range(8))
    assert tiers[-1]["id"] == "STANDARD_CODEX_FALLBACK"
    assert tiers[0]["id"] == "DOCUMENT_SUBSTRATE"
    assert any(stage["id"] == "LOCAL_SCIENTIFIC_VERIFICATION" for stage in tiers)
    assert any(stage["id"] == "LOCAL_COMPLETENESS_AND_CONSISTENCY_AUDIT" for stage in tiers)


def test_first_wave_reuses_qwen_only_in_a_narrower_role():
    payload = load_contract()
    policy = payload["first_wave_candidate_policy"]
    assert any(
        "existing certified Qwen3.5-4B" in item
        for item in policy["reuse_without_download"]
    )
    assert "NuExtract3 4B" in policy["heavyweight_candidates_deferred"]
    assert "Lift 9B" in policy["heavyweight_candidates_deferred"]
    assert "Qwen3.6-35B-A3B" in policy["heavyweight_candidates_deferred"]
    assert any(
        "prior Qwen3.5-4B grounded-candidate EXTRACT architecture as-is" in item
        for item in payload["exclusions"]
    )


def test_promotion_requires_exact_grounding_and_codex_minority_path():
    payload = load_contract()
    hard = payload["gates"]["hard_safety"]
    production = payload["gates"]["production_promotion"]
    assert "fabricated_or_unlocatable_quote_count == 0" in hard
    assert "exact_quote_locatability == 1.0 for every locally accepted claim" in hard
    assert "job_local_completion_fraction >= 0.80" in production
    assert "unsupported_claim_rate == 0 on promotion corpus" in production
    assert "projected remote input-token reduction >= 0.50" in production


def test_codex_fallback_becomes_versioned_teacher_data():
    payload = load_contract()
    teacher = payload["teacher_data_policy"]
    assert "STANDARD Codex fallback outputs" in teacher["source"]
    assert any("paper-level holdouts" in teacher["holdout"] for _ in [0])
    assert any(
        "do not train on unverified" in item
        for item in teacher["exclusions"]
    )
