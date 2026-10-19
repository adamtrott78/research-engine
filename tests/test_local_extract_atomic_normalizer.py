import copy

from research_engine.local_extract_atomic_normalizer import (
    build_normalization_prompt,
    build_normalization_request,
    guard_and_canonicalize_normalization,
)


def region():
    return {
        "region_id": "pdf:p2:000100-000420",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "page_number": 2,
        "start_char": 100,
        "end_char": 420,
        "section_hint": "Results",
        "raw_source_form": (
            "The proposed method rejects unknown devices at 90.28 percent "
            "while maintaining approximately 90 percent known-class accuracy."
        ),
    }


def test_request_is_narrow_and_does_not_delegate_provenance():
    request = build_normalization_request(
        region(),
        local_context="Nearby context only.",
    )

    assert request["candidate"]["candidate_id"] == region()["region_id"]
    assert request["candidate"]["raw_source_form"] == region()["raw_source_form"]
    assert request["candidate"]["page_number"] == 2
    assert request["bounded_context"] == "Nearby context only."
    prompt = build_normalization_prompt(request)
    assert "You are not performing retrieval" in prompt
    assert "Do not discover or quote new source text" in prompt


def test_guard_copies_quote_and_page_from_deterministic_region():
    payload = {
        "claims": [
            {
                "candidate_id": region()["region_id"],
                "type": "experimental-result",
                "normalized_form": (
                    "The method reports 90.28% unknown-device rejection "
                    "with about 90% known-class accuracy."
                ),
                "confidence": 0.91,
                # These fields are deliberately malicious/noisy. The guard
                # must ignore them rather than trusting model provenance.
                "raw_source_form": "fabricated quote",
                "page_number": 999,
            }
        ]
    }

    result = guard_and_canonicalize_normalization(region(), payload)

    assert result["status"] == "PASS_GROUNDED"
    claim = result["canonical_claims"][0]
    assert claim["type"] == "experimental_result"
    assert claim["raw_source_form"] == region()["raw_source_form"]
    assert claim["page_number"] == 2
    assert claim["source_region"]["start_char"] == 100
    assert claim["source_region"]["end_char"] == 420
    assert result["model_provenance_fields_accepted"] is False


def test_guard_rejects_candidate_reference_outside_supplied_span():
    payload = {
        "claims": [
            {
                "candidate_id": "pdf:p99:evil",
                "type": "result",
                "normalized_form": "Unsupported claim.",
                "confidence": 0.9,
            }
        ]
    }

    result = guard_and_canonicalize_normalization(region(), payload)

    assert result["status"] == "FALLTHROUGH_CODEX"
    assert "CLAIM_0:CANDIDATE_ID_MISMATCH" in result["reasons"]
    assert result["canonical_claims"] is None


def test_guard_allows_abstention():
    result = guard_and_canonicalize_normalization(
        region(),
        {"claims": []},
    )

    assert result["status"] == "PASS_GROUNDED"
    assert result["canonical_claims"] == []


def test_guard_collapses_exact_duplicate_claims_conservatively():
    row = {
        "candidate_id": region()["region_id"],
        "type": "result",
        "normalized_form": "The method reports unknown-device rejection.",
        "confidence": 0.95,
    }
    low = copy.deepcopy(row)
    low["confidence"] = 0.7

    result = guard_and_canonicalize_normalization(
        region(),
        {"claims": [row, low]},
    )

    assert result["status"] == "PASS_GROUNDED"
    assert len(result["canonical_claims"]) == 1
    assert result["canonical_claims"][0]["confidence"] == 0.7


def test_guard_fails_closed_on_excess_or_malformed_claims():
    payload = {
        "claims": [
            {
                "candidate_id": region()["region_id"],
                "type": "result",
                "normalized_form": f"claim {index}",
                "confidence": 0.5,
            }
            for index in range(5)
        ]
    }

    result = guard_and_canonicalize_normalization(region(), payload)
    assert result["status"] == "FALLTHROUGH_CODEX"
    assert "TOO_MANY_CLAIMS:5" in result["reasons"]

    malformed = guard_and_canonicalize_normalization(
        region(),
        {
            "claims": [
                {
                    "candidate_id": region()["region_id"],
                    "type": "???",
                    "normalized_form": "",
                    "confidence": 2,
                }
            ]
        },
    )
    assert malformed["status"] == "FALLTHROUGH_CODEX"
    assert malformed["canonical_claims"] is None
