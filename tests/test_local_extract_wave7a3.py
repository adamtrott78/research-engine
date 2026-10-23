from __future__ import annotations

import json
import pathlib

import pytest

from research_engine.local_extract_wave7a3_evaluate import (
    _implementation_validity,
    _verdict,
)
from research_engine.local_extract_wave7a3_intents import (
    _anchor_json_schema,
    _guard_anchors,
    _guard_obligations,
    _obligation_json_schema,
    _parse_structured_json,
)


ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7a3-author-anchored-obligations.v1.json"
WAVE7A2_CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"local-extract-wave7a2-evidence-conditioned.v2.json"


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



def test_wave7a3_structured_parser_rejects_multiple_objects() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_structured_json(
            '{"anchors":[]}\n{"anchors":[]}'
        )


def test_wave7a3_structured_schemas_freeze_only_interface_shape() -> None:
    anchors=_anchor_json_schema(min_anchors=3,max_anchors=7)
    assert anchors["additionalProperties"] is False
    assert anchors["properties"]["anchors"]["minItems"]==3
    assert anchors["properties"]["anchors"]["maxItems"]==7
    item=anchors["properties"]["anchors"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"])=={"anchor_id","role","descriptor"}
    assert item["properties"]["role"]["enum"]==[
        "problem","contribution","result","limitation_or_future_work"
    ]

    obligations=_obligation_json_schema(
        anchors=[{"anchor_id":"A01"},{"anchor_id":"A02"}],
        min_intents=8,
        max_intents=18,
    )
    assert obligations["additionalProperties"] is False
    assert obligations["properties"]["intents"]["minItems"]==8
    assert obligations["properties"]["intents"]["maxItems"]==18
    oitem=obligations["properties"]["intents"]["items"]
    assert set(oitem["required"])=={"anchor_id","question"}
    assert oitem["properties"]["anchor_id"]["enum"]==["A01","A02"]


def test_wave7a3_control_operational_contract_matches_valid_wave7a2() -> None:
    wave7a2=json.loads(WAVE7A2_CONTRACT.read_text())
    wave7a3=json.loads(CONTRACT.read_text())
    a72=next(
        row for row in wave7a2["generator_arms"]
        if row["id"]=="QWEN35_SOURCE_AWARE_INTENTS_V1"
    )
    a73=next(
        row for row in wave7a3["generator_arms"]
        if row["id"]=="QWEN35_SOURCE_AWARE_INTENTS_V1"
    )
    operational_keys={
        "id","model_id","model_revision","input_scope","max_input_chars",
        "min_intents","max_intents","max_output_tokens","temperature",
        "top_p","seed",
    }
    assert {k:a73[k] for k in operational_keys}=={
        k:a72[k] for k in operational_keys
    }


def test_wave7a3_implementation_validity_blocks_complete_fail_close() -> None:
    generated={"papers":[
        {
            "item_id":"paper-1",
            "generator_arms":{
                "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1":{
                    "status":"FAIL_CLOSED",
                    "anchor_interface_status":"PASS",
                    "obligation_interface_status":"PASS",
                }
            },
        },
        {
            "item_id":"paper-2",
            "generator_arms":{
                "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1":{
                    "status":"FAIL_CLOSED",
                    "anchor_interface_status":"PASS",
                    "obligation_interface_status":"PASS",
                }
            },
        },
    ]}
    result=_implementation_validity(generated)
    assert result["status"]=="FAIL_CLOSED"
    assert result["completely_fail_closed_candidate"] is True
    assert any(
        row["reason"]=="COMPLETELY_FAIL_CLOSED_CANDIDATE"
        for row in result["failures"]
    )


def test_wave7a3_implementation_validity_blocks_interface_failure() -> None:
    generated={"papers":[
        {
            "item_id":"paper-1",
            "generator_arms":{
                "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1":{
                    "status":"PASS",
                    "anchor_interface_status":"PASS",
                    "obligation_interface_status":"PASS",
                }
            },
        },
        {
            "item_id":"paper-2",
            "generator_arms":{
                "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1":{
                    "status":"FAIL_CLOSED",
                    "anchor_interface_status":"FAIL_CLOSED",
                    "anchor_interface_error":"JSONDecodeError:Extra data",
                    "obligation_interface_status":"PASS",
                }
            },
        },
    ]}
    result=_implementation_validity(generated)
    assert result["status"]=="FAIL_CLOSED"
    assert result["usable_candidate_papers"]==1
    assert any(
        row["stage"]=="anchor" and row["item_id"]=="paper-2"
        for row in result["failures"]
    )
