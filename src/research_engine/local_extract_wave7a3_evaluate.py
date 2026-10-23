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


CANDIDATE_ID="QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1"
CONTROL_ID="QWEN35_SOURCE_AWARE_INTENTS_V1"


def _augment_pair(
    row: dict[str,Any],
    *,
    residual_atoms: set[str],
    control_supported_atoms: set[str],
) -> dict[str,Any]:
    supported=set(str(x) for x in row["teacher"]["supported_atoms"])
    row["teacher"]["residual_atoms_recovered"]=[
        atom for atom in sorted(residual_atoms) if atom in supported
    ]
    row["teacher"]["residual_atoms_recovered_at_24"]=len(
        row["teacher"]["residual_atoms_recovered"]
    )
    row["teacher"]["control_supported_atoms_retained"]=[
        atom for atom in sorted(control_supported_atoms) if atom in supported
    ]
    row["teacher"]["control_supported_atoms_retained_at_24"]=len(
        row["teacher"]["control_supported_atoms_retained"]
    )
    return row


def _verdict(
    pairs: dict[str,dict[str,Any]],
) -> dict[str,Any]:
    promising=[]
    partial=[]
    comparisons=[]
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
        comparisons.append({
            "candidate_pair":candidate_key,
            "control_pair":control_key,
            "delta_supported_atoms_vs_control":delta,
            "residual_atoms_recovered_at_24":int(
                teacher["residual_atoms_recovered_at_24"]
            ),
            "control_supported_atoms_retained_at_24":int(
                teacher["control_supported_atoms_retained_at_24"]
            ),
        })
        if (
            float(candidate["teacher_leak_reject_fraction"])<=0.20
            and int(teacher["supported_atoms_at_24"])>=10
            and int(teacher["residual_atoms_recovered_at_24"])>=3
            and int(teacher["control_supported_atoms_retained_at_24"])>=6
            and float(nine["merged24_page_recall_fraction_of_ceiling"])>=0.90
            and delta>=3
        ):
            promising.append(candidate_key)
        elif (
            int(teacher["supported_atoms_at_24"])>=8
            and int(teacher["residual_atoms_recovered_at_24"])>=2
            and delta>=1
            and float(candidate["teacher_leak_reject_fraction"])<=0.20
        ):
            partial.append(candidate_key)
    if promising:
        return {
            "wave7a3_decision":"WAVE7A3_PROMISING",
            "qualifying_pairs":promising,
            "candidate_vs_control":comparisons,
            "rationale":"Author-anchored evidence obligations meet the frozen total-recall, residual-recovery, retention, leak, coverage, and improvement gates.",
        }
    if partial:
        return {
            "wave7a3_decision":"WAVE7A3_PARTIAL",
            "qualifying_pairs":partial,
            "candidate_vs_control":comparisons,
            "rationale":"Author-anchored evidence obligations improve over the frozen one-shot control and recover part of the salience residual, but do not satisfy every promotion gate.",
        }
    return {
        "wave7a3_decision":"WAVE7A3_REJECT",
        "qualifying_pairs":[],
        "candidate_vs_control":comparisons,
        "rationale":"Author-anchored evidence obligations do not meet the frozen promising or partial salience-recovery gate.",
    }


def _teacher_generation_diagnostic(
    generated: dict[str,Any],
    *,
    teacher_item_id: str,
) -> dict[str,Any]:
    paper=next(
        row for row in generated["papers"]
        if str(row["item_id"])==teacher_item_id
    )
    candidate=dict(paper["generator_arms"][CANDIDATE_ID])
    control=dict(paper["generator_arms"][CONTROL_ID])
    return {
        "item_id":teacher_item_id,
        "candidate_status":candidate.get("status"),
        "candidate_anchor_count":int(candidate.get("anchor_count") or 0),
        "candidate_anchors":list(candidate.get("anchors") or []),
        "candidate_intent_count":len(candidate.get("accepted") or []),
        "candidate_intents":list(candidate.get("accepted") or []),
        "candidate_rejected":list(candidate.get("rejected") or []),
        "control_status":control.get("status"),
        "control_intent_count":len(control.get("accepted") or []),
        "control_intents":list(control.get("accepted") or []),
    }


def evaluate_wave7a3(
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
    residual=set(str(x) for x in contract["evaluation"]["teacher_residual_atoms"])
    control_supported=set(
        str(x) for x in contract["evaluation"]["control_supported_atoms"]
    )
    pairs={}

    for generator_id in generator_ids:
        key=f"{generator_id}__BM25_DIRECT"
        pairs[key]=_augment_pair(
            _evaluate_pair(
                papers=papers,
                generator_id=generator_id,
                retriever_id="BM25_DIRECT",
                contract=contract,
                atoms=atoms,
                bge=None,
            ),
            residual_atoms=residual,
            control_supported_atoms=control_supported,
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
            pairs[key]=_augment_pair(
                _evaluate_pair(
                    papers=papers,
                    generator_id=generator_id,
                    retriever_id="BGE_M3_HYBRID_040_020_040",
                    contract=contract,
                    atoms=atoms,
                    bge=bge,
                ),
                residual_atoms=residual,
                control_supported_atoms=control_supported,
            )
        bge_metrics=bge.resource_metrics()
    finally:
        bge.close()
        gc.collect()

    decision=_verdict(pairs)
    return {
        "schema_version":1,
        "operation":"LOCAL_EXTRACT_WAVE7A3_AUTHOR_ANCHORED_EVAL",
        "benchmark_id":contract["benchmark_id"],
        "baseline_sha":contract["baseline_sha"],
        **decision,
        "pairs":pairs,
        "teacher_generation_diagnostic":_teacher_generation_diagnostic(
            generated,
            teacher_item_id=str(support_fixture["source_item_id"]),
        ),
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
