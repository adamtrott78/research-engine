from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave7_evaluate import (
    _rrf_merge,
    _verdict,
)
from research_engine.local_extract_wave7_intents import (
    _guard_generated_intents,
    _max_contiguous_copy,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7-intent-generation.v1.json"


def _prop(pid: str, page: int=1) -> dict:
    return {
        "proposition_id":pid,
        "page_number":page,
        "raw_source_form":pid,
        "display_text":pid,
    }


def test_wave7_contract_keeps_generator_as_independent_variable() -> None:
    contract=json.loads(CONTRACT.read_text())
    assert [row["id"] for row in contract["generator_arms"]]==[
        "DETERMINISTIC_SCIENCE_SCHEMA_V1",
        "QWEN35_FRONTMATTER_INTENTS_V1",
        "QWEN35_SOURCE_AWARE_INTENTS_V1",
    ]
    assert [row["id"] for row in contract["retrieval"]["retriever_arms"]]==[
        "BM25_DIRECT",
        "BGE_M3_HYBRID_040_020_040",
    ]
    assert contract["retrieval"]["fusion"]=={
        "id":"RRF_V1",
        "k":60,
        "global_proposition_budget":24,
        "deduplicate_by":"proposition_id",
    }
    assert contract["boundaries"]["wave7b"] is False


def test_intent_guard_rejects_numbers_duplicates_and_long_source_copy() -> None:
    source=(
        "the model uses a contrastive learning objective with frames from "
        "the same transmission as positive examples in training"
    )
    payload={
        "intents":[
            {"facet":"method","question":"How are positive examples constructed during training?"},
            {"facet":"evaluation","question":"What improvement of 13 percent is reported?"},
            {"facet":"method","question":"How are positive examples constructed during training?"},
            {"facet":"method","question":"The model uses a contrastive learning objective with frames from the same transmission as positive examples?"},
        ]
    }
    result=_guard_generated_intents(
        payload,
        source_context=source,
        min_intents=1,
        max_intents=12,
        min_chars=20,
        max_chars=220,
        max_copy_tokens=8,
    )
    assert result["status"]=="PASS"
    assert len(result["accepted"])==1
    reasons={row["reason"] for row in result["rejected"]}
    assert {"NUMERIC_ANSWER_LEAK","DUPLICATE","SOURCE_COPY"}<=reasons


def test_copy_detector_counts_contiguous_sequence() -> None:
    assert _max_contiguous_copy(
        "alpha beta gamma delta epsilon",
        "zero alpha beta gamma delta epsilon six",
    )==5


def test_rrf_fusion_rewards_cross_intent_support() -> None:
    a=_prop("a")
    b=_prop("b")
    c=_prop("c")
    merged=_rrf_merge(
        [
            [{**a,"oracle_rank":1},{**b,"oracle_rank":2}],
            [{**c,"oracle_rank":1},{**a,"oracle_rank":2}],
        ],
        rrf_k=60,
        per_intent_depth=12,
        budget=24,
    )
    assert merged[0]["proposition_id"]=="a"
    assert len({row["proposition_id"] for row in merged})==3


def test_wave7_verdict_is_predeclared_three_way_gate() -> None:
    base={
        "generator_id":"QWEN35_SOURCE_AWARE_INTENTS_V1",
        "teacher_leak_reject_fraction":0.0,
        "teacher":{
            "supported_atoms_at_24":7,
            "known_miss_recovery_count":2,
        },
        "nine_paper":{"merged24_page_recall_fraction_of_ceiling":1.0},
    }
    assert _verdict({"x":base})["wave7a_decision"]=="WAVE7A_REJECT"
    partial={**base,"teacher":{"supported_atoms_at_24":8,"known_miss_recovery_count":2}}
    assert _verdict({"x":partial})["wave7a_decision"]=="WAVE7A_PARTIAL"
    promising={
        **base,
        "teacher":{"supported_atoms_at_24":10,"known_miss_recovery_count":3},
        "nine_paper":{"merged24_page_recall_fraction_of_ceiling":0.91},
    }
    assert _verdict({"x":promising})["wave7a_decision"]=="WAVE7A_PROMISING"
