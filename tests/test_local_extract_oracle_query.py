from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_oracle_query import (
    KNOWN_MISSES,
    _aggregate,
    _bm25_scores,
    _rank,
    _support_rank,
    _verdict,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-oracle-query-diagnostic.v1.json"


def _prop(pid: str, page: int, text: str) -> dict:
    return {
        "proposition_id":pid,
        "page_number":page,
        "start_char":0,
        "end_char":len(text),
        "raw_source_form":text,
        "display_text":text,
        "section_hint":"",
        "source_layer":"AUTHORITATIVE_PDF_TEXT",
    }


def test_contract_freezes_answer_free_intents_for_all_13_atoms() -> None:
    contract=json.loads(CONTRACT.read_text())
    queries=contract["query_modes"]["INTENT_QUERY_ORACLE"]["queries"]
    assert len(queries)==13
    assert set(queries)=={
        "e1.a","e1.b","e1.c","e2.a","e2.b",
        "e3.a","e3.b","e3.c","e4.a","e4.b",
        "e6.a","e6.b","e6.c",
    }
    assert contract["query_modes"]["TARGET_TEXT_ORACLE"]["verdict_eligible"] is False
    assert contract["query_modes"]["INTENT_QUERY_ORACLE"]["verdict_eligible"] is True


def test_support_rank_allows_bounded_composition_but_not_wrong_page() -> None:
    atom={
        "id":"x",
        "page_number":2,
        "required_anchors":["alphaone","betatwo"],
    }
    ranked=_rank(
        [
            _prop("wrong",3,"alpha one beta two"),
            _prop("a",2,"alpha one"),
            _prop("b",2,"beta two"),
        ],
        [3.0,2.0,1.0],
    )
    result=_support_rank(atom,ranked)
    assert result["support_rank"]==3
    assert result["supported_at_3"] is True
    assert result["supported_at_1"] is False


def test_direct_bm25_prefers_exact_query_terms() -> None:
    props=[
        _prop("target",2,"positive pairs use frames from the same transmission"),
        _prop("other",2,"wireless channel fading and mobility"),
    ]
    ranked=_rank(props,_bm25_scores("positive pairs same transmission",props))
    assert ranked[0]["proposition_id"]=="target"


def test_verdict_is_frozen_three_way_gate() -> None:
    base={
        "supported_at_12":2,
        "known_miss_recovery_count":1,
    }
    rows={"a":dict(base)}
    assert _verdict(rows)["verdict"]=="RANKER_LIMIT_SUPPORTED"
    rows={"a":{"supported_at_12":7,"known_miss_recovery_count":1}}
    assert _verdict(rows)["verdict"]=="QUERY_REPRESENTATION_PARTIAL"
    rows={"a":{"supported_at_12":10,"known_miss_recovery_count":3}}
    assert _verdict(rows)["verdict"]=="QUERY_REPRESENTATION_CONFIRMED"


def test_known_miss_set_unchanged() -> None:
    assert KNOWN_MISSES==("e2.a","e2.b","e4.b")


def test_aggregate_reports_delta_from_broad_query() -> None:
    rows=[
        {
            "id":"e2.a",
            "supported_at_1":True,
            "supported_at_3":True,
            "supported_at_12":True,
            "reciprocal_rank":1.0,
            "best_token_f1":{"1":0.5,"3":0.5,"12":0.5},
        },
        {
            "id":"e4.b",
            "supported_at_1":False,
            "supported_at_3":False,
            "supported_at_12":False,
            "reciprocal_rank":0.0,
            "best_token_f1":{"1":0.1,"3":0.2,"12":0.3},
        },
    ]
    result=_aggregate(rows,broad_baseline={"supported_at_12":0})
    assert result["supported_at_12"]==1
    assert result["delta_supported_at_12_vs_broad"]==1
