from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave7a2_v2_intents import (
    _alias_source_context,
    _guard_alias_intents,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7a2-evidence-conditioned.v2.json"


def test_v2_contract_keeps_scientific_gates_and_records_invalid_predecessor() -> None:
    contract=json.loads(CONTRACT.read_text())
    assert contract["invalid_predecessor_run"]["run_id"]==40009297107
    assert contract["invalid_predecessor_run"]["scientific_gate_consumed"] is False
    assert contract["retrieval"]["fusion"]["global_proposition_budget"]==24
    assert contract["predeclared_verdict"]["production_promotion_authorized"] is False
    assert contract["predeclared_verdict"]["wave7b_authorized"] is False


def test_alias_context_uses_short_provenance_handles() -> None:
    regions=[
        {"region_id":"pdf:p2:000100-000200","page_number":2,"display_text":"alpha evidence"},
        {"region_id":"pdf:p3:000300-000400","page_number":3,"display_text":"beta evidence"},
    ]
    context,aliases=_alias_source_context(regions,max_chars=5000)
    assert aliases=={
        "R01":"pdf:p2:000100-000200",
        "R02":"pdf:p3:000300-000400",
    }
    assert "REGION R01" in context
    assert "REGION R02" in context


def test_alias_guard_maps_back_to_exact_source_region_and_ignores_metadata() -> None:
    payload={
        "intents":[
            {
                "region_alias":"R01",
                "question":"How are positive examples constructed for the learning procedure?",
                "note":"ignored metadata",
            }
        ],
        "coverage_note":"ignored top-level metadata",
    }
    result=_guard_alias_intents(
        payload,
        source_context="REGION R01\nTEXT positive examples are described here",
        aliases={"R01":"pdf:p2:000100-000200"},
        min_intents=1,
        max_intents=4,
        min_chars=20,
        max_chars=220,
        max_copy_tokens=8,
    )
    assert result["status"]=="PASS"
    assert result["accepted"][0]["source_region_id"]=="pdf:p2:000100-000200"
