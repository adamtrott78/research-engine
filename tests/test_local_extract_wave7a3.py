from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_wave7a3_evaluate import _verdict
from research_engine.local_extract_wave7a3_intents import (
    _guard_anchors,
    _guard_obligations,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7a3-author-anchored-obligations.v1.json"


def test_wave7a3_contract_freezes_salience_before_questions() -> None:
    contract=json.loads(CONTRACT.read_text())
    arms={row["id"]:row for row in contract["generator_arms"]}
    assert set(arms)=={
        "DETERMINISTIC_SCIENCE_SCHEMA_V1",
        "QWEN35_SOURCE_AWARE_INTENTS_V1",
        "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1",
    }
    candidate=arms["QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1"]
    assert candidate["verdict_eligible"] is True
    assert candidate["salience_context"]["anchor_roles"]==[
        "problem","contribution","result","limitation_or_future_work"
    ]
    assert contract["evaluation"]["teacher_residual_atoms"]==[
        "e1.a","e1.b","e1.c","e3.c","e6.a","e6.c"
    ]
    assert contract["retrieval"]["fusion"]["global_proposition_budget"]==24
    assert contract["boundaries"]["discourse_classifier_download"] is False
    assert contract["execution"]["expected_runs"]==1


def test_anchor_guard_preserves_rhetorical_roles_and_blocks_long_copy() -> None:
    payload={"anchors":[
        {
            "anchor_id":"A01",
            "role":"contribution",
            "descriptor":"The paper proposes a transfer-oriented representation learning framework for device identification.",
        },
        {
            "anchor_id":"A02",
            "role":"result",
            "descriptor":"one two three four five six seven eight nine ten eleven",
        },
    ]}
    result=_guard_anchors(
        payload,
        source_context="zero one two three four five six seven eight nine ten eleven twelve",
        min_anchors=1,
        max_anchors=7,
        min_chars=20,
        max_chars=260,
        max_copy_tokens=10,
    )
    assert result["status"]=="PASS"
    assert [x["anchor_id"] for x in result["accepted"]]==["A01"]
    assert any(x["reason"]=="SOURCE_COPY" for x in result["rejected"])


def test_obligation_guard_requires_known_anchor_and_answer_free_question() -> None:
    anchors=[
        {
            "anchor_id":"A01",
            "role":"contribution",
            "descriptor":"The paper proposes a domain-transfer framework.",
        }
    ]
    payload={"intents":[
        {
            "anchor_id":"A01",
            "question":"How are supervision signals constructed for the proposed training procedure?",
        },
        {
            "anchor_id":"A99",
            "question":"What separate experimental condition is used during evaluation?",
        },
        {
            "anchor_id":"A01",
            "question":"What happens at 45 MSps during capture?",
        },
    ]}
    result=_guard_obligations(
        payload,
        anchors=anchors,
        source_context="selected source context",
        min_intents=1,
        max_intents=8,
        max_per_anchor=4,
        min_chars=20,
        max_chars=220,
        max_copy_tokens=8,
    )
    assert result["status"]=="PASS"
    assert len(result["accepted"])==1
    assert result["accepted"][0]["anchor_id"]=="A01"
    reasons={x["reason"] for x in result["rejected"]}
    assert "UNKNOWN_ANCHOR_ID" in reasons
    assert "NUMERIC_ANSWER_LEAK" in reasons


def _pair(
    generator_id: str,
    retriever_id: str,
    supported: list[str],
    *,
    leak: float=0.0,
    page_fraction: float=1.0,
) -> dict:
    residual={"e1.a","e1.b","e1.c","e3.c","e6.a","e6.c"}
    control={"e2.a","e2.b","e3.a","e3.b","e4.a","e4.b","e6.b"}
    s=set(supported)
    return {
        "generator_id":generator_id,
        "retriever_id":retriever_id,
        "teacher_leak_reject_fraction":leak,
        "teacher":{
            "supported_atoms":supported,
            "supported_atoms_at_24":len(supported),
            "residual_atoms_recovered_at_24":len(s & residual),
            "control_supported_atoms_retained_at_24":len(s & control),
        },
        "nine_paper":{"merged24_page_recall_fraction_of_ceiling":page_fraction},
    }


def test_wave7a3_promising_requires_residual_recovery_and_retention() -> None:
    control_supported=["e2.a","e2.b","e3.a","e3.b","e4.a","e4.b","e6.b"]
    candidate=control_supported+["e1.a","e1.b","e6.a"]
    pairs={}
    for retr in ("BM25_DIRECT","BGE_M3_HYBRID_040_020_040"):
        pairs[f"QWEN35_SOURCE_AWARE_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_SOURCE_AWARE_INTENTS_V1",retr,control_supported
        )
        pairs[f"QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1__{retr}"]=_pair(
            "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1",retr,candidate
        )
    result=_verdict(pairs)
    assert result["wave7a3_decision"]=="WAVE7A3_PROMISING"


def test_wave7a3_partial_requires_actual_improvement() -> None:
    control_supported=["e2.a","e2.b","e3.a","e3.b","e4.a","e4.b","e6.b"]
    candidate=control_supported+["e1.a"]
    pairs={}
    for retr in ("BM25_DIRECT","BGE_M3_HYBRID_040_020_040"):
        pairs[f"QWEN35_SOURCE_AWARE_INTENTS_V1__{retr}"]=_pair(
            "QWEN35_SOURCE_AWARE_INTENTS_V1",retr,control_supported
        )
        pairs[f"QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1__{retr}"]=_pair(
            "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1",retr,candidate
        )
    assert _verdict(pairs)["wave7a3_decision"]=="WAVE7A3_REJECT"
