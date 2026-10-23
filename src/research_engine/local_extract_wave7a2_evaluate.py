from __future__ import annotations

import gc
import pathlib
from typing import Any

from research_engine.local_extract_reranker_bakeoff import (
    BGEM3Scorer,
    snapshot_download_exact,
)
from research_engine.local_extract_wave7_evaluate import (
    _evaluate_pair,
    _prepare_papers,
)


CANDIDATE_ID="QWEN35_EVIDENCE_GAP_INTENTS_V1"
CONTROL_ID="QWEN35_SOURCE_AWARE_INTENTS_V1"


def _verdict(pairs: dict[str,dict[str,Any]]) -> dict[str,Any]:
    promising=[]
    partial=[]
    details=[]
    for retriever_id in ("BM25_DIRECT","BGE_M3_HYBRID_040_020_040"):
        candidate_key=f"{CANDIDATE_ID}__{retriever_id}"
        control_key=f"{CONTROL_ID}__{retriever_id}"
        candidate=pairs[candidate_key]
        control=pairs[control_key]
        teacher=candidate["teacher"]
        nine=candidate["nine_paper"]
        delta=(
            int(teacher["supported_atoms_at_24"])
            - int(control["teacher"]["supported_atoms_at_24"])
        )
        details.append({
            "candidate_pair":candidate_key,
            "control_pair":control_key,
            "delta_supported_atoms_vs_control":delta,
        })
        if (
            float(candidate["teacher_leak_reject_fraction"])<=0.20
            and int(teacher["supported_atoms_at_24"])>=10
            and int(teacher["known_miss_recovery_count"])==3
            and float(nine["merged24_page_recall_fraction_of_ceiling"])>=0.90
            and delta>=3
        ):
            promising.append(candidate_key)
        elif (
            int(teacher["supported_atoms_at_24"])>=8
            or int(teacher["known_miss_recovery_count"])==3
        ):
            partial.append(candidate_key)
    if promising:
        return {
            "wave7a2_decision":"WAVE7A2_PROMISING",
            "qualifying_pairs":promising,
            "candidate_vs_control":details,
            "rationale":"Evidence-conditioned gap discovery meets the frozen semantic, hard-miss, leak, page-coverage, and improvement-over-one-shot gates.",
        }
    if partial:
        return {
            "wave7a2_decision":"WAVE7A2_PARTIAL",
            "qualifying_pairs":partial,
            "candidate_vs_control":details,
            "rationale":"Evidence-conditioned gap discovery materially improves atomic evidence recovery but does not satisfy every frozen promotion gate.",
        }
    return {
        "wave7a2_decision":"WAVE7A2_REJECT",
        "qualifying_pairs":[],
        "candidate_vs_control":details,
        "rationale":"Evidence-conditioned gap discovery does not meet the frozen promising or partial retrieval gate.",
    }


def evaluate_wave7a2(
    *,
    fixture: dict[str,Any],
    contract: dict[str,Any],
    support_fixture: dict[str,Any],
    generated: dict[str,Any],
    workdir: pathlib.Path,
) -> dict[str,Any]:
    if generated["scientific_state_materialized"] or generated["production_routing_changed"]:
        raise RuntimeError("generation report violated scientific boundary")
    papers=_prepare_papers(
        fixture=fixture,
        generated=generated,
        workdir=workdir,
    )
    atoms=[dict(x) for x in support_fixture["atoms"]]
    generator_ids=[row["id"] for row in contract["generator_arms"]]
    pairs={}

    for generator_id in generator_ids:
        key=f"{generator_id}__BM25_DIRECT"
        pairs[key]=_evaluate_pair(
            papers=papers,
            generator_id=generator_id,
            retriever_id="BM25_DIRECT",
            contract=contract,
            atoms=atoms,
            bge=None,
        )

    bge_cfg=next(
        row for row in contract["retrieval"]["retriever_arms"]
        if row["id"]=="BGE_M3_HYBRID_040_020_040"
    )
    model_dir=workdir/"models"/"bge-m3"
    download=snapshot_download_exact(
        repository=str(bge_cfg["repository"]),
        revision=str(bge_cfg["revision"]),
        destination=model_dir,
        allow_patterns=[
            "config.json","pytorch_model.bin","sentencepiece.bpe.model",
            "special_tokens_map.json","tokenizer.json","tokenizer_config.json",
            "colbert_linear.pt","sparse_linear.pt",
        ],
    )
    bge=BGEM3Scorer(
        model_dir,
        max_passage_length=int(bge_cfg["max_passage_length"]),
        hybrid_weights=[float(v) for v in bge_cfg["hybrid_weights"]],
        batch_size=int(bge_cfg["batch_size"]),
    )
    try:
        for generator_id in generator_ids:
            key=f"{generator_id}__BGE_M3_HYBRID_040_020_040"
            pairs[key]=_evaluate_pair(
                papers=papers,
                generator_id=generator_id,
                retriever_id="BGE_M3_HYBRID_040_020_040",
                contract=contract,
                atoms=atoms,
                bge=bge,
            )
        bge_metrics=bge.resource_metrics()
    finally:
        bge.close()
        gc.collect()

    decision=_verdict(pairs)
    return {
        "schema_version":1,
        "operation":"LOCAL_EXTRACT_WAVE7A2_EVIDENCE_CONDITIONED_EVAL",
        "benchmark_id":contract["benchmark_id"],
        "baseline_sha":contract["baseline_sha"],
        **decision,
        "pairs":pairs,
        "generation_metrics":generated["metrics"],
        "bge_resource_metrics":{
            **bge_metrics,
            "model_download_bytes":int(download["bytes"]),
            "model_download_seconds":float(download["download_seconds"]),
            "model_snapshot_tree_sha256":str(download["tree_sha256"]),
        },
        "hard_safety":{
            "scientific_state_materialized_false":True,
            "production_routing_changed_false":True,
            "qwen_verify_changed_false":True,
            "wave7b_started_false":True,
        },
        "scientific_state_materialized":False,
        "production_routing_changed":False,
        "qwen_verify_changed":False,
        "wave7b_started":False,
    }
