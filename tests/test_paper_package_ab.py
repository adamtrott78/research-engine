from research_engine.orchestration import JobType
from research_engine.paper_package_ab import (
    PaperPackageABPosthoc,
    PaperPackageABPreflight,
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



def test_preflight_uses_extract_extractor_for_catalog_coverage():
    class Planner:
        def plan(self, **_kwargs):
            return {
                "source_artifacts": [
                    {"sha256": "abc"},
                ]
            }

        def completed_jobs(self, *_args, **_kwargs):
            return []

    class Extractor:
        def __init__(self):
            self.calls = []

        def coverage(self, shas):
            self.calls.append(list(shas))
            return {
                "requested": 1,
                "native_covered": 1,
                "standard_covered": 1,
                "missing_native": [],
                "missing_standard": [],
            }

    service = object.__new__(PaperPackageABPreflight)
    extractor = Extractor()
    service.planner = Planner()
    service.package_extract_extractor = extractor

    result = service.run(run_id="run", expected_papers=1)

    assert extractor.calls == [["abc"]]
    assert result["decision"] == "PASS"
    assert result["package_coverage"]["native_covered"] == 1



def test_posthoc_compares_package_and_pdf_controls_without_model_calls():
    service = object.__new__(PaperPackageABPosthoc)
    service.store = None
    events = [
        {
            "id": 1,
            "calibration_type": "PAPER_PACKAGE_EXTRACT_AB",
            "source_job_id": "extract:1",
            "replay_input_tokens": 900,
            "replay_cached_input_tokens": 100,
            "replay_output_tokens": 50,
            "replay_reasoning_output_tokens": 20,
            "wall_time_ms": 1000,
            "result_summary": {
                "evidence": [
                    {
                        "type": "RESULT",
                        "normalized_form": "same result",
                        "raw_source_form": "same quote",
                        "page_number": 2,
                    }
                ]
            },
            "created_at": "2026-10-03T00:00:00Z",
        },
        {
            "id": 2,
            "calibration_type": "PAPER_PACKAGE_EXTRACT_AB_CONTROL_PDF",
            "source_job_id": "extract:1",
            "replay_input_tokens": 1000,
            "replay_cached_input_tokens": 100,
            "replay_output_tokens": 52,
            "replay_reasoning_output_tokens": 21,
            "wall_time_ms": 1100,
            "result_summary": {
                "evidence": [
                    {
                        "type": "RESULT",
                        "normalized_form": "same result",
                        "raw_source_form": "same quote",
                        "page_number": 2,
                    }
                ]
            },
            "created_at": "2026-10-03T00:00:01Z",
        },
        {
            "id": 3,
            "calibration_type": "PAPER_PACKAGE_VERIFY_AB",
            "source_job_id": "verify:1",
            "replay_input_tokens": 800,
            "replay_cached_input_tokens": 50,
            "replay_output_tokens": 30,
            "replay_reasoning_output_tokens": 10,
            "wall_time_ms": 900,
            "result_summary": {
                "verified": True,
                "accepted_evidence_ids": ["e:1"],
                "issues": [],
            },
            "created_at": "2026-10-03T00:00:02Z",
        },
        {
            "id": 4,
            "calibration_type": "PAPER_PACKAGE_VERIFY_AB_CONTROL_PDF",
            "source_job_id": "verify:1",
            "replay_input_tokens": 1000,
            "replay_cached_input_tokens": 50,
            "replay_output_tokens": 31,
            "replay_reasoning_output_tokens": 11,
            "wall_time_ms": 950,
            "result_summary": {
                "verified": True,
                "accepted_evidence_ids": ["e:1"],
                "issues": [],
            },
            "created_at": "2026-10-03T00:00:03Z",
        },
        {
            "id": 5,
            "calibration_type": "PAPER_PACKAGE_VERIFY_AB",
            "source_job_id": "verify:2",
            "replay_input_tokens": 700,
            "replay_cached_input_tokens": 25,
            "replay_output_tokens": 20,
            "replay_reasoning_output_tokens": 5,
            "wall_time_ms": 800,
            "result_summary": {
                "verified": False,
                "accepted_evidence_ids": [],
                "issues": ["unsupported"],
            },
            "created_at": "2026-10-03T00:00:04Z",
        },
    ]
    service._events = lambda _run_id: events

    result = service.run(run_id="run")

    assert result["decision"] == "PASS_CONTEMPORANEOUS_EQUIVALENCE"
    assert result["package_events"] == 3
    assert result["controlled_pairs"] == 2
    assert result["controlled_pair_passes"] == 2
    assert result["controlled_pair_agreement_rate"] == 1.0
    assert result["controlled_pdf_input_tokens"] == 2000
    assert result["controlled_package_input_tokens"] == 1700
    assert result["controlled_input_reduction_fraction"] == 0.15
    controlled = [row for row in result["rows"] if row["controlled_pair"]]
    assert all(row["package_vs_pdf_agreement_pass"] is True for row in controlled)


def test_posthoc_detects_contemporaneous_context_difference():
    service = object.__new__(PaperPackageABPosthoc)
    service.store = None
    service._events = lambda _run_id: [
        {
            "id": 1,
            "calibration_type": "PAPER_PACKAGE_VERIFY_AB",
            "source_job_id": "verify:1",
            "replay_input_tokens": 800,
            "replay_cached_input_tokens": 0,
            "replay_output_tokens": 20,
            "replay_reasoning_output_tokens": 0,
            "wall_time_ms": 100,
            "result_summary": {
                "verified": True,
                "accepted_evidence_ids": ["e:1"],
                "issues": [],
            },
            "created_at": "2026-10-03T00:00:00Z",
        },
        {
            "id": 2,
            "calibration_type": "PAPER_PACKAGE_VERIFY_AB_CONTROL_PDF",
            "source_job_id": "verify:1",
            "replay_input_tokens": 1000,
            "replay_cached_input_tokens": 0,
            "replay_output_tokens": 20,
            "replay_reasoning_output_tokens": 0,
            "wall_time_ms": 100,
            "result_summary": {
                "verified": False,
                "accepted_evidence_ids": [],
                "issues": ["unsupported"],
            },
            "created_at": "2026-10-03T00:00:01Z",
        },
    ]

    result = service.run(run_id="run")

    assert result["decision"] == "MEASURED_CONTEXT_DIFFERENCE"
    assert result["controlled_pairs"] == 1
    assert result["controlled_pair_passes"] == 0
    assert result["controlled_pair_agreement_rate"] == 0.0
