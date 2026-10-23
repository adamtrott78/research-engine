from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave1 import _bm25
from research_engine.local_extract_wave4 import exact_propositions
from research_engine.local_extract_wave6 import (
    SELECTOR_ID,
    decide_wave6,
    deterministic_proposition_selections,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave6-contract.v1.json"
)


def _region(region_id: str, page: int, start: int, raw: str) -> dict:
    return {
        "region_id": region_id,
        "page_number": page,
        "start_char": start,
        "end_char": start + len(raw),
        "raw_source_form": raw,
        "display_text": raw,
        "section_hint": "Evaluation",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "rank": 1,
    }


def _with_props(region: dict) -> dict:
    row = dict(region)
    row["propositions"] = exact_propositions(row)
    return row


def test_paper_global_selector_matches_one_global_bm25_ranking() -> None:
    regions = [
        _with_props(
            _region(
                "pdf:p1:000000-000200",
                1,
                0,
                "Receiver diversity improves robustness. "
                "The dataset contains several devices.",
            )
        ),
        _with_props(
            _region(
                "pdf:p2:000000-000220",
                2,
                0,
                "Wireless channel shift reduces classification accuracy. "
                "Calibration improves unknown rejection.",
            )
        ),
    ]
    query = "wireless channel shift robustness unknown rejection"
    selected = deterministic_proposition_selections(query, regions)

    all_props = [
        proposition
        for region in regions
        for proposition in region["propositions"]
    ]
    aliases = []
    for proposition in all_props:
        row = dict(proposition)
        row["region_id"] = str(proposition["proposition_id"])
        aliases.append(row)
    ranked = _bm25(query, aliases)

    expected = {}
    for row in ranked:
        expected.setdefault(
            str(row["parent_region_id"]),
            str(row["proposition_id"]),
        )

    assert {
        parent: row["proposition_id"]
        for parent, row in selected.items()
    } == expected
    assert all(
        row["selector_id"] == SELECTOR_ID
        for row in selected.values()
    )


def test_paper_global_selector_returns_one_per_nonempty_region() -> None:
    regions = [
        _with_props(
            _region(
                "pdf:p1:000000-000200",
                1,
                0,
                "First complete statement. Second complete statement.",
            )
        ),
        _with_props(
            _region(
                "pdf:p2:000000-000200",
                2,
                0,
                "Third complete statement. Fourth complete statement.",
            )
        ),
    ]
    selected = deterministic_proposition_selections("statement", regions)
    assert set(selected) == {
        "pdf:p1:000000-000200",
        "pdf:p2:000000-000200",
    }


def test_wave6_contract_is_narrow_correction() -> None:
    contract = json.loads(CONTRACT.read_text())
    fixed = set(contract["experimental_control"]["fixed"])
    changed = set(contract["experimental_control"]["changed"])

    assert "Wave-2 exact_source_regions(max_chars=650) region construction" in fixed
    assert "Wave-4 deterministic exact proposition enumeration" in fixed
    assert "Wave-4/Wave-5 Qwen normalization prompt" in fixed
    assert (
        "preserved six-claim teacher subset and unchanged lexical F1>=0.5 gate"
        in fixed
    )
    assert any("paper's selected regions together" in item for item in changed)
    assert any("owning paper" in item for item in changed)
    assert contract["deterministic_selector"]["proposal_model_calls"] == 0
    assert contract["provenance_fix"]["scientific_behavior_changed"] is False
    assert contract["execution"]["new_privileged_authority_required"] is False
    assert contract["execution"]["gitrunner_substrate_change_required"] is False


def test_wave6_research_gate_remains_unchanged() -> None:
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
    assert decide_wave6(report) == "PARTIAL"
    report["metrics"]["job_local_completion_fraction"] = 0.49
    assert decide_wave6(report) == "REJECT"
