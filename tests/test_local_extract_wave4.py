from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave4 import (
    decide_wave4,
    exact_propositions,
    guard_normalization_payload,
    guard_proposal_payload,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave4-contract.v1.json"
)


def _region(raw: str) -> dict:
    return {
        "region_id": "pdf:p2:000100-000900",
        "page_number": 2,
        "start_char": 100,
        "end_char": 100 + len(raw),
        "raw_source_form": raw,
        "display_text": raw,
        "section_hint": "Methods",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "rank": 1,
    }


def test_exact_propositions_split_terminal_sentences_with_exact_offsets() -> None:
    raw = "First complete scientific claim. Second complete result!"
    rows = exact_propositions(_region(raw))
    assert len(rows) == 2
    assert rows[0]["raw_source_form"] == "First complete scientific claim."
    assert rows[0]["start_char"] == 100
    assert rows[0]["end_char"] == 132
    assert rows[1]["raw_source_form"] == "Second complete result!"
    assert rows[1]["start_char"] == 133
    assert rows[1]["end_char"] == 156
    assert rows[0]["proposition_id"].endswith(":s01")
    assert rows[1]["proposition_id"].endswith(":s02")


def test_exact_propositions_do_not_split_common_scientific_abbreviation() -> None:
    raw = "Fig. 2 shows a robust result. Another complete claim follows."
    rows = exact_propositions(_region(raw))
    assert len(rows) == 2
    assert rows[0]["raw_source_form"].startswith("Fig. 2 shows")
    assert rows[0]["raw_source_form"].endswith("result.")


def test_exact_propositions_exclude_incomplete_trailing_fragment() -> None:
    raw = (
        "A complete method statement appears here. "
        "Finally we apply the trained encoder"
    )
    rows = exact_propositions(_region(raw))
    assert len(rows) == 1
    assert rows[0]["raw_source_form"].endswith("here.")
    assert "Finally" not in rows[0]["raw_source_form"]


def test_proposal_guard_accepts_only_known_proposition_id() -> None:
    rows = exact_propositions(
        _region("A complete scientific method statement appears here.")
    )
    accepted = guard_proposal_payload(
        {
            "decision": "SELECT",
            "proposition_id": rows[0]["proposition_id"],
        },
        rows,
    )
    assert accepted["status"] == "SELECT_ACCEPTED"
    assert accepted["selected"]["proposition_id"] == rows[0]["proposition_id"]

    rejected = guard_proposal_payload(
        {
            "decision": "SELECT",
            "proposition_id": "invented",
        },
        rows,
    )
    assert rejected["status"] == "HARD_REJECT"
    assert rejected["reasons"] == ["UNKNOWN_PROPOSITION_ID"]


def test_proposal_skip_schema_is_fail_closed() -> None:
    assert guard_proposal_payload(
        {"decision": "SKIP"},
        [],
    )["status"] == "SKIP_ACCEPTED"
    assert guard_proposal_payload(
        {"decision": "SKIP", "extra": True},
        [],
    )["status"] == "HARD_REJECT"


def test_normalization_guard_copies_exact_proposition_provenance() -> None:
    proposition = exact_propositions(
        _region(
            "The receiver sampled IEEE 802.11b frames at 45 MSps."
        )
    )[0]
    guard = guard_normalization_payload(
        {
            "decision": "CLAIM",
            "type": "dataset",
            "normalized_form": (
                "The receiver sampled IEEE 802.11b frames at 45 MSps."
            ),
            "confidence": 0.95,
        },
        proposition,
    )
    assert guard["status"] == "CLAIM_ACCEPTED"
    evidence = guard["evidence"][0]
    assert evidence["candidate_id"] == proposition["proposition_id"]
    assert evidence["parent_region_id"] == proposition["parent_region_id"]
    assert evidence["raw_source_form"] == proposition["raw_source_form"]
    assert evidence["page_number"] == proposition["page_number"]
    assert evidence["source_start_char"] == proposition["start_char"]
    assert evidence["source_end_char"] == proposition["end_char"]


def test_wave4_contract_reverts_wave3_regions_and_preserves_teacher_gate() -> None:
    contract = json.loads(CONTRACT.read_text())
    fixed = set(contract["experimental_control"]["fixed"])
    changed = set(contract["experimental_control"]["changed"])
    assert "Wave-2 exact_source_regions(max_chars=650) region construction" in fixed
    assert "FACET_INTERLEAVED_BM25_V1 ranking algorithm and facet queries" in fixed
    assert "top_k_regions=12" in fixed
    assert (
        "preserved teacher_agreement lexical scorer and F1>=0.5 pass threshold"
        in fixed
    )
    assert any("deterministic code enumerates exact terminal proposition spans" in row for row in changed)
    assert contract["execution"]["new_privileged_authority_required"] is False
    assert contract["execution"]["gitrunner_substrate_change_required"] is False


def test_wave4_research_gate_remains_unchanged() -> None:
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
    assert decide_wave4(report) == "PARTIAL"

    report["metrics"]["job_local_completion_fraction"] = 0.49
    assert decide_wave4(report) == "REJECT"
