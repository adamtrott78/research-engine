from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_diagnostics import (
    _best_bundle_alignment,
    select_under_char_budget,
    sentence_windows,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-post-wave6-diagnostics.v1.json"
)


def _prop(index: int, text: str, *, page: int = 1) -> dict:
    start = index * 100
    return {
        "proposition_id": f"page:p{page}:s{index:02d}",
        "parent_region_id": f"page:p{page}",
        "page_number": page,
        "start_char": start,
        "end_char": start + len(text),
        "raw_source_form": text,
        "display_text": text,
        "section_hint": "",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "terminal_complete": True,
    }


def test_sentence_windows_are_overlapping_and_page_scoped() -> None:
    props = [
        _prop(1, "First statement."),
        _prop(2, "Second statement."),
        _prop(3, "Third statement."),
        _prop(1, "Other page statement.", page=2),
    ]
    rows = sentence_windows(props, window_size=2)
    assert len(rows) == 2
    assert rows[0]["proposition_ids"] == [
        "page:p1:s01",
        "page:p1:s02",
    ]
    assert rows[1]["proposition_ids"] == [
        "page:p1:s02",
        "page:p1:s03",
    ]
    assert all(row["page_number"] == 1 for row in rows)


def test_fixed_budget_selection_never_exceeds_budget() -> None:
    rows = [
        {
            "region_id": f"r{index}",
            "page_number": index,
            "start_char": 0,
            "end_char": len(text),
            "raw_source_form": text,
            "display_text": text,
            "section_hint": "",
            "source_layer": "AUTHORITATIVE_PDF_TEXT",
        }
        for index, text in enumerate(
            [
                "wireless channel robustness evidence one.",
                "unknown rejection calibration evidence two.",
                "dataset receiver transmitter evidence three.",
            ],
            start=1,
        )
    ]
    selected = select_under_char_budget(
        "wireless channel unknown rejection",
        rows,
        budget_chars=85,
    )
    assert selected
    assert sum(len(row["raw_source_form"]) for row in selected) <= 85


def test_bundle_oracle_can_recover_complementary_facts() -> None:
    props = [
        _prop(1, "The framework pre-trains on unlabeled source data."),
        _prop(2, "The classifier is tested on target-domain data."),
    ]
    reference = (
        "The framework pre-trains on unlabeled source data and "
        "tests the classifier on target-domain data."
    )
    single = _best_bundle_alignment(reference, props, max_bundle=1)
    bundle = _best_bundle_alignment(reference, props, max_bundle=2)
    assert bundle["f1"] > single["f1"]
    assert len(bundle["candidate_ids"]) == 2


def test_diagnostic_contract_preserves_legacy_gate_and_atomizes_teacher() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["recovery_inventory"]["historical_evidence_object_count"] == 71
    assert contract["recovery_inventory"]["normalized_teacher_claims_preserved"] == 6
    assert contract["fixed_control"]["no_generation"] is True
    assert contract["fixed_control"]["no_model_download"] is True
    assert "F1>=0.5" in contract["fixed_control"]["legacy_teacher_metric"]
    atoms = contract["teacher_atomization"]
    assert {row["parent_teacher_id"] for row in atoms} == {
        "e1", "e2", "e3", "e4", "e5", "e6"
    }
    assert len(atoms) == 14
    assert next(row for row in atoms if row["id"] == "e5.a")["duplicate_of"] == "e4.b"
    assert contract["boundaries"]["wave7_authorized"] is False
    assert contract["boundaries"]["production_extract_change_authorized"] is False
