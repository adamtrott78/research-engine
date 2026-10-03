from research_engine.orchestration import JobType
from research_engine.paper_package_ab import (
    agreement_pass,
    extract_agreement,
    quote_token_recall,
    token_f1,
    verify_agreement,
)


def test_token_f1_exact_and_partial():
    assert token_f1("alpha beta gamma", "alpha beta gamma") == 1.0
    score = token_f1("alpha beta gamma", "alpha beta")
    assert 0.79 < score < 0.81


def test_quote_token_recall_allows_extra_context():
    assert quote_token_recall(
        "evaluated method detects unseen transmitters",
        "The evaluated method detects unseen transmitters under dynamic RF conditions.",
    ) == 1.0
    assert quote_token_recall("alpha beta gamma", "alpha beta") == 2 / 3


def test_extract_agreement_exact_match_passes():
    historical = {
        "evidence": [
            {
                "id": "old:1",
                "type": "RESULT",
                "normalized_form": "The method detects unseen transmitters.",
                "raw_source_form": "detects unseen transmitters",
                "page_number": 3,
                "confidence": 0.9,
            }
        ]
    }
    replay = {
        "evidence": [
            {
                "id": "new:1",
                "type": "RESULT",
                "normalized_form": "The method detects unseen transmitters.",
                "raw_source_form": "detects unseen transmitters",
                "page_number": 3,
                "confidence": 0.8,
            }
        ]
    }
    agreement = extract_agreement(historical, replay)
    assert agreement["matched_normalized_f1"] == 1.0
    assert agreement["historical_coverage_at_f1_0_5"] == 1.0
    assert agreement["replay_precision_at_f1_0_5"] == 1.0
    assert agreement["page_agreement_on_passing_matches"] == 1.0
    assert agreement["type_agreement_on_passing_matches"] == 1.0
    assert agreement_pass(JobType.EXTRACT, agreement) is True


def test_extract_agreement_detects_page_regression():
    historical = {
        "evidence": [
            {
                "id": "old:1",
                "type": "RESULT",
                "normalized_form": "same claim",
                "raw_source_form": "same quote",
                "page_number": 2,
            }
        ]
    }
    replay = {
        "evidence": [
            {
                "id": "new:1",
                "type": "RESULT",
                "normalized_form": "same claim",
                "raw_source_form": "same quote",
                "page_number": 8,
            }
        ]
    }
    agreement = extract_agreement(historical, replay)
    assert agreement["historical_coverage_at_f1_0_5"] == 1.0
    assert agreement["page_agreement_on_passing_matches"] == 0.0
    assert agreement_pass(JobType.EXTRACT, agreement) is False


def test_extract_agreement_penalizes_missing_and_extra_evidence():
    historical = {
        "evidence": [
            {
                "type": "RESULT",
                "normalized_form": "alpha finding",
                "raw_source_form": "alpha finding",
                "page_number": 1,
            },
            {
                "type": "RESULT",
                "normalized_form": "beta finding",
                "raw_source_form": "beta finding",
                "page_number": 2,
            },
        ]
    }
    replay = {
        "evidence": [
            {
                "type": "RESULT",
                "normalized_form": "alpha finding",
                "raw_source_form": "alpha finding",
                "page_number": 1,
            },
            {
                "type": "RESULT",
                "normalized_form": "unrelated material",
                "raw_source_form": "unrelated material",
                "page_number": 4,
            },
            {
                "type": "RESULT",
                "normalized_form": "extra unrelated",
                "raw_source_form": "extra unrelated",
                "page_number": 5,
            },
        ]
    }
    agreement = extract_agreement(historical, replay)
    assert agreement["historical_coverage_at_f1_0_5"] == 0.5
    assert agreement["replay_precision_at_f1_0_5"] == 1 / 3
    assert agreement_pass(JobType.EXTRACT, agreement) is False


def test_verify_agreement_requires_decision_and_exact_authoritative_ids():
    historical = {
        "verified": True,
        "item_id": "paper:1",
        "accepted_evidence_ids": ["e:1", "e:2"],
        "issues": [],
    }
    replay = {
        "verified": True,
        "item_id": "paper:1",
        "accepted_evidence_ids": ["e:2", "e:1"],
        "issues": [],
    }
    agreement = verify_agreement(historical, replay)
    assert agreement["verified_equal"] is True
    assert agreement["accepted_evidence_ids_equal"] is True
    assert agreement["issues_equal"] is True
    assert agreement_pass(JobType.VERIFY, agreement) is True

    replay["accepted_evidence_ids"] = ["e:1"]
    disagreement = verify_agreement(historical, replay)
    assert disagreement["accepted_evidence_ids_equal"] is False
    assert agreement_pass(JobType.VERIFY, disagreement) is False


def test_execution_metadata_does_not_affect_agreement():
    historical = {
        "verified": False,
        "item_id": "paper:1",
        "accepted_evidence_ids": [],
        "issues": ["quote unsupported"],
        "_execution": {"usage": {"input_tokens": 999}},
        "_materialization": {"event": "historical"},
    }
    replay = {
        "verified": False,
        "item_id": "paper:1",
        "accepted_evidence_ids": [],
        "issues": ["quote unsupported"],
        "_execution": {"usage": {"input_tokens": 1}},
    }
    agreement = verify_agreement(historical, replay)
    assert agreement["verified_equal"] is True
    assert agreement["accepted_evidence_ids_equal"] is True
    assert agreement["issues_equal"] is True
