import json
from pathlib import Path

import pytest

from research_engine.local_extract_wave1 import (
    decide_wave1,
    evidence_page_recall,
    exact_source_regions,
    guard_claims,
    rank_regions,
    source_urls,
    teacher_agreement,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave1-corpus.v1.json"
)


def test_fixture_preserves_nine_paper_historical_oracle():
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert value["source_run_id"] == "rf-open-world-corpus-expansion-v1"
    assert len(value["papers"]) == 9
    assert sum(
        paper["historical_evidence_count"]
        for paper in value["papers"]
    ) == 71
    assert sum(
        len(paper["historical_evidence_pages"])
        for paper in value["papers"]
    ) == 43
    assert sum(
        paper["total_pdf_pages"]
        for paper in value["papers"]
    ) == 65
    assert sum(
        paper["historical_context_chars"]
        for paper in value["papers"]
    ) == 322102
    assert value["historical"]["extract_input_tokens"] == 223262
    teacher = [
        paper for paper in value["papers"]
        if paper.get("teacher_claims")
    ]
    assert len(teacher) == 1
    assert len(teacher[0]["teacher_claims"]) == 6


def test_arxiv_urls_are_bounded_and_version_aware():
    urls = source_urls("2306.13895")
    assert urls[0] == "https://arxiv.org/pdf/2306.13895"
    assert "https://arxiv.org/pdf/2306.13895v1" in urls
    assert "https://arxiv.org/pdf/2306.13895v10.pdf" in urls
    assert len(urls) == 22
    assert all(url.startswith("https://arxiv.org/pdf/") for url in urls)


def test_region_ranking_covers_expected_page():
    pages = [
        {
            "page_number": 1,
            "text": "Background\nThe weather was mild.\n",
        },
        {
            "page_number": 4,
            "text": (
                "Results\n"
                "The detector rejects previously unseen RF transmitters "
                "with high accuracy.\n"
            ),
        },
    ]
    regions = exact_source_regions(pages, max_chars=96)
    ranked = rank_regions(
        "RF unknown transmitter rejection result accuracy",
        regions,
        top_k=2,
    )
    assert ranked[0]["page_number"] == 4
    assert evidence_page_recall([4], ranked[:1]) == 1.0


def test_guard_copies_exact_quote_and_rejects_unknown_candidates():
    region = {
        "region_id": "pdf:p2:000000-000100",
        "page_number": 2,
        "start_char": 0,
        "end_char": 100,
        "raw_source_form": "The method improves open-set rejection.",
        "display_text": "The method improves open-set rejection.",
        "section_hint": "Results",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
    }
    result = guard_claims(
        {
            "claims": [
                {
                    "candidate_id": region["region_id"],
                    "type": "result",
                    "normalized_form": (
                        "The method improves open-set rejection."
                    ),
                    "confidence": 0.9,
                }
            ]
        },
        [region],
        max_claims=12,
        max_claims_per_region=2,
    )
    assert result["status"] == "PASS_GROUNDED"
    assert (
        result["evidence"][0]["raw_source_form"]
        == region["raw_source_form"]
    )

    denied = guard_claims(
        {
            "claims": [
                {
                    "candidate_id": "invented",
                    "type": "result",
                    "normalized_form": "Invented unsupported claim.",
                    "confidence": 0.9,
                }
            ]
        },
        [region],
        max_claims=12,
        max_claims_per_region=2,
    )
    assert denied["status"] == "FAIL_CLOSED"
    assert "CLAIM_0:UNKNOWN_CANDIDATE" in denied["reasons"]


def test_teacher_agreement_scores_semantic_page_and_type_match():
    teacher = [
        {
            "normalized_form": "The method improves unknown rejection.",
            "page_number": 4,
            "type": "result",
        }
    ]
    qwen = [
        {
            "normalized_form": "The method improves unknown rejection.",
            "page_number": 4,
            "type": "result",
        }
    ]
    result = teacher_agreement(teacher, qwen)
    assert result["qwen_precision_at_f1_0_5"] == 1.0
    assert result["teacher_claim_coverage_at_f1_0_5"] == 1.0
    assert result["page_agreement_on_passing_matches"] == 1.0
    assert result["type_agreement_on_passing_matches"] == 1.0


def test_wave1_decision_is_bounded_to_three_values():
    base = {
        "hard_safety": {
            "fabricated_or_unlocatable_quote_count_zero": True,
            "exact_quote_locatability_one": True,
            "page_provenance_coverage_one": True,
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
        },
        "metrics": {
            "job_local_completion_fraction": 1.0,
            "historical_evidence_page_recall_at_12": 0.9,
            "projected_remote_input_token_reduction": 0.8,
        },
        "teacher_subset_agreement": {
            "qwen_precision_at_f1_0_5": 0.96,
            "teacher_claim_coverage_at_f1_0_5": 0.85,
            "page_agreement_on_passing_matches": 1.0,
            "type_agreement_on_passing_matches": 1.0,
        },
    }
    assert decide_wave1(base) == "PROMISING"

    partial = json.loads(json.dumps(base))
    partial["teacher_subset_agreement"][
        "qwen_precision_at_f1_0_5"
    ] = 0.8
    assert decide_wave1(partial) == "PARTIAL"

    rejected = json.loads(json.dumps(base))
    rejected["metrics"]["job_local_completion_fraction"] = 0.4
    assert decide_wave1(rejected) == "REJECT"

    unsafe = json.loads(json.dumps(base))
    unsafe["hard_safety"][
        "fabricated_or_unlocatable_quote_count_zero"
    ] = False
    assert decide_wave1(unsafe) == "REJECT"


def test_guard_rejects_invalid_type_and_region_fanout():
    region = {
        "region_id": "r1",
        "page_number": 1,
        "start_char": 0,
        "end_char": 20,
        "raw_source_form": "Exact source sentence.",
    }
    invalid = guard_claims(
        {
            "claims": [
                {
                    "candidate_id": "r1",
                    "type": "made_up",
                    "normalized_form": "A sufficiently long claim.",
                    "confidence": 0.5,
                }
            ]
        },
        [region],
        max_claims=12,
        max_claims_per_region=2,
    )
    assert invalid["status"] == "FAIL_CLOSED"

    fanout = guard_claims(
        {
            "claims": [
                {
                    "candidate_id": "r1",
                    "type": "method",
                    "normalized_form": f"Distinct atomic claim number {idx}.",
                    "confidence": 0.5,
                }
                for idx in range(3)
            ]
        },
        [region],
        max_claims=12,
        max_claims_per_region=2,
    )
    assert fanout["status"] == "FAIL_CLOSED"


@pytest.mark.parametrize("bad", [0, -1])
def test_rank_regions_requires_positive_top_k(bad):
    with pytest.raises(ValueError):
        rank_regions("query", [], top_k=bad)
