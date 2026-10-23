from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_page_conditioned import (
    SELECTOR_ID,
    page_conditioned_proposition_pool,
    rank_page_conditioned_propositions,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-page-conditioned-retrieval.v1.json"
)


def _prop(pid: str, page: int, text: str) -> dict:
    return {
        "proposition_id": pid,
        "parent_region_id": f"page:p{page}",
        "page_number": page,
        "start_char": 0,
        "end_char": len(text),
        "raw_source_form": text,
        "display_text": text,
        "section_hint": "",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
        "terminal_complete": True,
    }


def _region(rid: str, page: int, text: str) -> dict:
    return {
        "region_id": rid,
        "page_number": page,
        "start_char": 0,
        "end_char": len(text),
        "raw_source_form": text,
        "display_text": text,
        "section_hint": "",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
    }


def test_page_conditioned_pool_uses_selected_pages_not_region_membership() -> None:
    propositions = [
        _prop("p1-a", 1, "First relevant proposition."),
        _prop("p1-b", 1, "Second proposition outside selected region."),
        _prop("p2-a", 2, "Other page proposition."),
    ]
    selected_regions = [
        _region("r1", 1, "Only one coarse region on page one.")
    ]
    pool = page_conditioned_proposition_pool(
        propositions,
        selected_regions,
    )
    assert [row["proposition_id"] for row in pool] == [
        "p1-a",
        "p1-b",
    ]


def test_page_conditioned_ranker_is_bounded_and_uses_existing_facets() -> None:
    propositions = [
        _prop(
            f"p{index}",
            1 + index % 2,
            (
                "wireless channel robustness unknown rejection "
                if index % 3 == 0
                else "dataset receiver transmitter capture accuracy "
            )
            + f"statement {index}.",
        )
        for index in range(20)
    ]
    selected = rank_page_conditioned_propositions(
        "wireless channel unknown rejection dataset evaluation",
        propositions,
        top_k=12,
    )
    assert len(selected) == 12
    assert len({row["proposition_id"] for row in selected}) == 12
    assert all(row["selector_id"] == SELECTOR_ID for row in selected)
    assert all("retrieval_channel" in row for row in selected)


def test_page_conditioned_contract_keeps_wave7_closed() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert "top_k_regions=12" in contract["fixed"]
    assert "maximum 12 selected propositions per paper" in contract["fixed"]
    assert contract["boundaries"]["qwen_model_calls"] == 0
    assert contract["boundaries"]["external_model_downloads"] == 0
    assert contract["boundaries"]["wave7_authorized"] is False
    assert (
        contract["boundaries"]["production_extract_change_authorized"]
        is False
    )
