from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave2 import (
    aggregate_region_units,
    decide_wave2,
    guard_region_payload,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave2-contract.v1.json"
)


def _region(region_id: str = "pdf:p2:000100-000200") -> dict:
    return {
        "region_id": region_id,
        "page_number": 2,
        "start_char": 100,
        "end_char": 200,
        "raw_source_form": "Exact authoritative source text.",
        "display_text": "Exact authoritative source text.",
    }


def test_wave2_contract_keeps_retrieval_fixed() -> None:
    contract = json.loads(CONTRACT.read_text())
    fixed = set(contract["experimental_control"]["fixed"])
    assert "FACET_INTERLEAVED_BM25_V1" in fixed
    assert "top_k_regions=12" in fixed
    assert contract["architecture"]["budgeting"]["claims_per_region_max"] == 1
    assert (
        contract["architecture"]["budgeting"]["max_output_tokens_per_region_call"]
        == 320
    )
    assert contract["execution"]["new_privileged_authority_required"] is False
    assert contract["execution"]["gitrunner_substrate_change_required"] is False


def test_region_claim_binds_candidate_identity_deterministically() -> None:
    region = _region()
    guarded = guard_region_payload(
        {
            "decision": "CLAIM",
            "type": "method",
            "normalized_form": "The method uses an exact grounded source region.",
            "confidence": 0.9,
        },
        region,
    )
    assert guarded["status"] == "CLAIM_ACCEPTED"
    evidence = guarded["evidence"][0]
    assert evidence["candidate_id"] == region["region_id"]
    assert evidence["raw_source_form"] == region["raw_source_form"]
    assert evidence["page_number"] == region["page_number"]


def test_generated_candidate_identity_is_rejected_instead_of_trusted() -> None:
    guarded = guard_region_payload(
        {
            "decision": "CLAIM",
            "candidate_id": "made-up",
            "type": "method",
            "normalized_form": "The method uses an exact grounded source region.",
            "confidence": 0.9,
        },
        _region(),
    )
    assert guarded["status"] == "HARD_REJECT"
    assert guarded["evidence"] == []
    assert guarded["reasons"] == ["CLAIM_SCHEMA"]


def test_valid_claim_survives_unrelated_hard_reject() -> None:
    valid = guard_region_payload(
        {
            "decision": "CLAIM",
            "type": "result",
            "normalized_form": "The experiment reports a grounded result.",
            "confidence": 0.8,
        },
        _region("pdf:p2:000100-000200"),
    )
    units = [
        {
            "candidate_id": "pdf:p2:000100-000200",
            "rank": 1,
            "guard": valid,
        },
        {
            "candidate_id": "pdf:p3:000100-000200",
            "rank": 2,
            "guard": {
                "status": "HARD_REJECT",
                "reasons": ["INVALID_DECISION"],
                "evidence": [],
            },
        },
    ]
    aggregated = aggregate_region_units(units)
    assert aggregated["status"] == "RESIDUAL_REQUIRED"
    assert aggregated["accepted_claim_count"] == 1
    assert len(aggregated["hard_rejects"]) == 1


def test_exact_duplicate_is_dropped_without_erasing_claim_set() -> None:
    first = guard_region_payload(
        {
            "decision": "CLAIM",
            "type": "dataset",
            "normalized_form": "The dataset contains grounded RF captures.",
            "confidence": 0.9,
        },
        _region("pdf:p2:000100-000200"),
    )
    second = guard_region_payload(
        {
            "decision": "CLAIM",
            "type": "dataset",
            "normalized_form": "The dataset contains grounded RF captures.",
            "confidence": 0.8,
        },
        _region("pdf:p3:000100-000200"),
    )
    aggregated = aggregate_region_units(
        [
            {
                "candidate_id": "pdf:p2:000100-000200",
                "rank": 1,
                "guard": first,
            },
            {
                "candidate_id": "pdf:p3:000100-000200",
                "rank": 2,
                "guard": second,
            },
        ]
    )
    assert aggregated["status"] == "PASS_LOCAL"
    assert aggregated["accepted_claim_count"] == 1
    assert aggregated["exact_duplicate_claims_dropped"] == 1


def test_wave2_research_floor_and_semantic_gate_remain_strict() -> None:
    report = {
        "hard_safety": {
            "quotes": True,
            "pages": True,
        },
        "metrics": {
            "job_local_completion_fraction": 0.75,
        },
        "teacher_subset_agreement": {
            "qwen_precision_at_f1_0_5": 1.0,
            "teacher_claim_coverage_at_f1_0_5": 0.5,
            "page_agreement_on_passing_matches": 1.0,
            "type_agreement_on_passing_matches": 1.0,
        },
    }
    assert decide_wave2(report) == "PARTIAL"

    report["metrics"]["job_local_completion_fraction"] = 0.49
    assert decide_wave2(report) == "REJECT"
