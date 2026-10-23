from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave4 import exact_propositions
from research_engine.local_extract_wave5 import (
    SELECTOR_ID,
    decide_wave5,
    deterministic_proposition_selection,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave5-contract.v1.json"
)


def _region(raw: str) -> dict:
    return {
        "region_id": "pdf:p4:001000-001900",
        "page_number": 4,
        "start_char": 1000,
        "end_char": 1000 + len(raw),
        "raw_source_form": raw,
        "display_text": raw,
        "section_hint": "Evaluation",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "rank": 1,
    }


def test_deterministic_selector_chooses_query_relevant_exact_proposition() -> None:
    rows = exact_propositions(
        _region(
            "The dataset contains ten devices and three receivers. "
            "Wireless channel shift reduces classifier accuracy substantially."
        )
    )
    selected = deterministic_proposition_selection(
        "wireless channel shift classifier accuracy",
        rows,
    )
    assert selected is not None
    assert selected["proposition_id"] == rows[1]["proposition_id"]
    assert selected["raw_source_form"] == rows[1]["raw_source_form"]
    assert selected["selector_id"] == SELECTOR_ID
    assert isinstance(selected["selector_score"], float)


def test_deterministic_selector_is_stable_on_zero_score_tie() -> None:
    rows = exact_propositions(
        _region(
            "First unrelated complete scientific statement. "
            "Second unrelated complete scientific statement."
        )
    )
    selected = deterministic_proposition_selection(
        "zzzz_nonexistent_query_token",
        rows,
    )
    assert selected is not None
    assert selected["proposition_id"] == rows[0]["proposition_id"]


def test_deterministic_selector_skips_only_empty_complete_set() -> None:
    assert deterministic_proposition_selection("topic", []) is None


def test_wave5_contract_changes_only_proposal_selection() -> None:
    contract = json.loads(CONTRACT.read_text())
    fixed = set(contract["experimental_control"]["fixed"])
    changed = set(contract["experimental_control"]["changed"])

    assert "Wave-2 exact_source_regions(max_chars=650) region construction" in fixed
    assert "FACET_INTERLEAVED_BM25_V1 region ranking algorithm and facet queries" in fixed
    assert "Wave-4 deterministic exact proposition enumeration" in fixed
    assert "Wave-4 Qwen normalization prompt" in fixed
    assert "Wave-4 normalization JSON schema and strict guard" in fixed
    assert (
        "preserved teacher_agreement lexical scorer and F1>=0.5 pass threshold"
        in fixed
    )
    assert "remove the Wave-4 Qwen proposition-selection model call" in changed
    assert contract["deterministic_selector"]["model_calls"] == 0
    assert contract["execution"]["new_privileged_authority_required"] is False
    assert contract["execution"]["gitrunner_substrate_change_required"] is False


def test_wave5_research_gate_remains_unchanged() -> None:
    report = {
        "hard_safety": {"all_safe": True},
        "metrics": {"job_local_completion_fraction": 1.0},
        "teacher_subset_agreement": {
            "qwen_precision_at_f1_0_5": 0.0,
            "teacher_claim_coverage_at_f1_0_5": 0.0,
            "page_agreement_on_passing_matches": 0.0,
            "type_agreement_on_passing_matches": 0.0,
        },
    }
    assert decide_wave5(report) == "PARTIAL"

    report["metrics"]["job_local_completion_fraction"] = 0.49
    assert decide_wave5(report) == "REJECT"
