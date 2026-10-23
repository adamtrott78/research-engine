from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave7a2_evaluate import _verdict
from research_engine.local_extract_wave7a2_intents import (
    _guard_evidence_intents,
    _max_contiguous_copy,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7a2-evidence-conditioned.v1.json"


def test_wave7a2_contract_freezes_only_evidence_gap_candidate() -> None:
    contract=json.loads(CONTRACT.read_text())
    arms={row["id"]:row for row in contract["generator_arms"]}
    assert set(arms)=={
        "DETERMINISTIC_SCIENCE_SCHEMA_V1",
        "QWEN35_SOURCE_AWARE_INTENTS_V1",
        "QWEN35_EVIDENCE_GAP_INTENTS_V1",
    }
    assert arms["QWEN35_EVIDENCE_GAP_INTENTS_V1"]["verdict_eligible"] is True
    assert contract["retrieval"]["fusion"]["global_proposition_budget"]==24
    assert contract["execution"]["expected_runs"]==1
    assert contract["boundaries"]["wave7b"] is False


def test_evidence_guard_requires_known_region_and_atomic_question() -> None:
    payload={"intents":[
        {"region_id":"r1","question":"How are positive examples constructed for contrastive training?"},
        {"region_id":"bad","question":"What separate evaluation condition is examined by the authors?"},
    ]}
    result=_guard_evidence_intents(
        payload,
        source_context="[region_id=r1] positive examples are built from source evidence",
        selected_region_ids={"r1"},
        min_intents=1,
        max_intents=4,
        min_chars=20,
        max_chars=220,
        max_copy_tokens=8,
    )
    assert result["status"]=="PASS"
    assert len(result["accepted"])==1
    assert result["accepted"][0]["region_id"]=="r1"
    assert any(x["reason"]=="UNKNOWN_REGION_ID" for x in result["rejected"])


def test_copy_guard_measures_contiguous_source_copy() -> None:
    source="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    assert _max_contiguous_copy(
        "alpha beta gamma delta epsilon zeta eta theta iota?",
        source,
    )>=9


def _pair(gen: str, retr: str, atoms: int, misses: int, leak: float=0.0, frac: float=1.0) -> dict:
    return {
        "generator_id":gen,
        "retriever_id":retr,
        "teacher_leak_reject_fraction":leak,
        "teacher":{
            "supported_atoms_at_24":atoms,
            "known_miss_recovery_count":misses,
        },
        "nine_paper":{
            "merged24_page_recall_fraction_of_ceiling":frac,
        },
    }


def test_verdict_requires_improvement_over_matching_control() -> None:
    pairs={}
    for retr in ("BM25_DIRECT","BGE_M3_HYBRID_040_020_040"):
        pairs[f"QWEN35_SOURCE_AWARE_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_SOURCE_AWARE_INTENTS_V1",retr,7,2
        )
        pairs[f"QWEN35_EVIDENCE_GAP_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_EVIDENCE_GAP_INTENTS_V1",retr,10,3
        )
    result=_verdict(pairs)
    assert result["wave7a2_decision"]=="WAVE7A2_PROMISING"


def test_verdict_partial_for_eight_atoms() -> None:
    pairs={}
    for retr in ("BM25_DIRECT","BGE_M3_HYBRID_040_020_040"):
        pairs[f"QWEN35_SOURCE_AWARE_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_SOURCE_AWARE_INTENTS_V1",retr,5,1
        )
        pairs[f"QWEN35_EVIDENCE_GAP_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_EVIDENCE_GAP_INTENTS_V1",retr,8,2
        )
    assert _verdict(pairs)["wave7a2_decision"]=="WAVE7A2_PARTIAL"
