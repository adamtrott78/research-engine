from __future__ import annotations

import gc
import hashlib
import pathlib
import re
from typing import Any

from research_engine.local_extract_diagnostics import (
    _best_alignment,
    full_pdf_propositions,
)
from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _rank,
)
from research_engine.local_extract_page_conditioned import (
    page_conditioned_proposition_pool,
)
from research_engine.local_extract_reranker_bakeoff import (
    BGEM3Scorer,
    snapshot_download_exact,
)
from research_engine.local_extract_wave1 import (
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    rank_regions,
)


QWEN_GENERATORS={
    "QWEN35_FRONTMATTER_INTENTS_V1",
    "QWEN35_SOURCE_AWARE_INTENTS_V1",
}
RETRIEVERS=("BM25_DIRECT","BGE_M3_HYBRID_040_020_040")


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+","",str(value).lower())


def _ensure_pdf(
    paper: dict[str,Any],
    path: pathlib.Path,
) -> dict[str,Any]:
    expected=str(paper["sha256"])
    if path.exists():
        payload=path.read_bytes()
        actual=hashlib.sha256(payload).hexdigest()
        if actual==expected:
            return {"url":"REUSED_EXACT_LOCAL","sha256":actual,"bytes":len(payload)}
        path.unlink()
    return download_exact_pdf(paper,path)


def _teacher_leak_filter(
    intents: list[dict[str,Any]],
    *,
    forbidden: list[str],
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    accepted=[]
    rejected=[]
    for row in intents:
        question=str(row["question"])
        normalized=_normalized(question)
        matched=[value for value in forbidden if value in normalized]
        if any(ch.isdigit() for ch in question) or matched:
            rejected.append({
                **row,
                "reason":"TEACHER_ANSWER_LEAK",
                "matched_forbidden":matched,
            })
        else:
            accepted.append(dict(row))
    return accepted,rejected


def _rrf_merge(
    rankings: list[list[dict[str,Any]]],
    *,
    rrf_k: int,
    per_intent_depth: int,
    budget: int,
) -> list[dict[str,Any]]:
    scores: dict[str,float]={}
    rows: dict[str,dict[str,Any]]={}
    contributing: dict[str,list[dict[str,Any]]]={}
    for intent_index,ranking in enumerate(rankings):
        for rank,row in enumerate(ranking[:per_intent_depth],start=1):
            pid=str(row["proposition_id"])
            rows[pid]=dict(row)
            scores[pid]=scores.get(pid,0.0)+(1.0/(rrf_k+rank))
            contributing.setdefault(pid,[]).append({
                "intent_index":intent_index,
                "rank":rank,
            })
    ordered=sorted(
        scores,
        key=lambda pid:(-scores[pid],pid),
    )[:budget]
    result=[]
    for rank,pid in enumerate(ordered,start=1):
        row=dict(rows[pid])
        row["fusion_rank"]=rank
        row["fusion_score"]=float(scores[pid])
        row["fusion_contributors"]=contributing[pid]
        result.append(row)
    return result


def _teacher_support(
    atoms: list[dict[str,Any]],
    merged: list[dict[str,Any]],
) -> dict[str,Any]:
    supported=[]
    atom_rows=[]
    for atom in atoms:
        expected_page=int(atom["page_number"])
        matched=[]
        for anchor in atom["required_anchors"]:
            hit=False
            for row in merged:
                if int(row["page_number"])!=expected_page:
                    continue
                source=_normalized(str(row.get("raw_source_form") or row.get("display_text") or ""))
                if str(anchor) in source:
                    hit=True
                    break
            matched.append(hit)
        ok=all(matched)
        if ok:
            supported.append(str(atom["id"]))
        alignment=_best_alignment(str(atom["normalized_form"]),merged)
        atom_rows.append({
            "id":str(atom["id"]),
            "supported":ok,
            "anchor_hits":matched,
            "best_token_f1":float(alignment["f1"]),
        })
    known=[x for x in ("e2.a","e2.b","e4.b") if x in supported]
    return {
        "supported_atoms":supported,
        "supported_atoms_at_24":len(supported),
        "support_recall_at_24":len(supported)/len(atoms) if atoms else 0.0,
        "known_miss_recovered":known,
        "known_miss_recovery_count":len(known),
        "mean_best_token_f1_at_24":(
            sum(row["best_token_f1"] for row in atom_rows)/len(atom_rows)
            if atom_rows else 0.0
        ),
        "atoms":atom_rows,
    }


def _prepare_papers(
    *,
    fixture: dict[str,Any],
    generated: dict[str,Any],
    workdir: pathlib.Path,
) -> list[dict[str,Any]]:
    generated_by_id={str(row["item_id"]):row for row in generated["papers"]}
    papers=[]
    for source in fixture["papers"]:
        paper=dict(source)
        item_id=str(paper["item_id"])
        gen=generated_by_id[item_id]
        pdf_path=workdir/"pdfs"/f"{paper['arxiv_id']}.pdf"
        acquisition=_ensure_pdf(paper,pdf_path)
        pages,title=extract_pdf_pages(pdf_path)
        regions=exact_source_regions(
            pages,
            max_chars=int(fixture["qwen_config"]["max_region_chars"]),
        )
        broad=(str(fixture["topic"])+" "+title+" "+item_id).strip()
        selected=rank_regions(broad,regions,top_k=32)[:12]
        props=page_conditioned_proposition_pool(
            full_pdf_propositions(pages),
            selected,
        )
        papers.append({
            **paper,
            "title":title,
            "acquisition":acquisition,
            "selected_page_numbers":sorted({int(row["page_number"]) for row in selected}),
            "propositions":props,
            "generator_arms":gen["generator_arms"],
        })
    return papers


def _evaluate_pair(
    *,
    papers: list[dict[str,Any]],
    generator_id: str,
    retriever_id: str,
    contract: dict[str,Any],
    atoms: list[dict[str,Any]],
    bge: BGEM3Scorer | None,
) -> dict[str,Any]:
    fusion=dict(contract["retrieval"]["fusion"])
    forbidden=list(contract["teacher_anti_leak_audit"]["forbidden_normalized_substrings"])
    total_expected_pages=0
    total_coarse_hits=0
    total_merged_hits=0
    teacher_detail=None
    paper_rows=[]
    total_leak_reject=0
    total_raw_accepted=0
    for paper in papers:
        arm=dict(paper["generator_arms"][generator_id])
        intents=[dict(x) for x in arm.get("accepted") or []]
        leak_rejected=[]
        if paper.get("teacher_claims"):
            intents,leak_rejected=_teacher_leak_filter(intents,forbidden=forbidden)
            total_leak_reject+=len(leak_rejected)
            total_raw_accepted+=len(arm.get("accepted") or [])
        rankings=[]
        for intent in intents:
            question=str(intent["question"])
            if retriever_id=="BM25_DIRECT":
                ranking=_rank(
                    paper["propositions"],
                    _bm25_scores(question,paper["propositions"]),
                )
            elif retriever_id=="BGE_M3_HYBRID_040_020_040":
                if bge is None:
                    raise RuntimeError("BGE scorer missing")
                scores=bge.score(question,paper["propositions"])[
                    "BGE_M3_HYBRID_040_020_040"
                ]
                ranking=_rank(paper["propositions"],scores)
            else:
                raise RuntimeError(f"unknown retriever {retriever_id}")
            rankings.append(ranking)
        merged=_rrf_merge(
            rankings,
            rrf_k=int(fusion["k"]),
            per_intent_depth=int(contract["retrieval"]["per_intent_depth"]),
            budget=int(fusion["global_proposition_budget"]),
        ) if rankings else []

        expected={int(x) for x in paper["historical_evidence_pages"]}
        selected_pages=set(int(x) for x in paper["selected_page_numbers"])
        merged_pages={int(x["page_number"]) for x in merged}
        total_expected_pages+=len(expected)
        total_coarse_hits+=len(expected & selected_pages)
        total_merged_hits+=len(expected & merged_pages)

        row={
            "item_id":str(paper["item_id"]),
            "intent_count_raw":int(arm.get("raw_count") or 0),
            "intent_count_guard_accepted":len(arm.get("accepted") or []),
            "intent_count_after_teacher_leak_audit":len(intents),
            "teacher_leak_rejected":leak_rejected,
            "merged_proposition_count":len(merged),
            "merged_page_numbers":sorted(merged_pages),
            "historical_page_hits":len(expected & merged_pages),
            "historical_page_count":len(expected),
            "merged_propositions":[
                {
                    "proposition_id":str(x["proposition_id"]),
                    "page_number":int(x["page_number"]),
                    "fusion_rank":int(x["fusion_rank"]),
                    "fusion_score":float(x["fusion_score"]),
                    "raw_source_form":str(x["raw_source_form"]),
                }
                for x in merged
            ],
        }
        if paper.get("teacher_claims"):
            teacher_detail=_teacher_support(atoms,merged)
            row["teacher_support"]=teacher_detail
        paper_rows.append(row)

    coarse_ceiling=(
        total_coarse_hits/total_expected_pages if total_expected_pages else 1.0
    )
    page_recall=(
        total_merged_hits/total_expected_pages if total_expected_pages else 1.0
    )
    fraction=(page_recall/coarse_ceiling) if coarse_ceiling else 1.0
    teacher_detail=teacher_detail or {
        "supported_atoms_at_24":0,
        "support_recall_at_24":0.0,
        "known_miss_recovery_count":0,
        "known_miss_recovered":[],
        "mean_best_token_f1_at_24":0.0,
        "atoms":[],
    }
    leak_fraction=(
        total_leak_reject/total_raw_accepted if total_raw_accepted else 0.0
    )
    return {
        "generator_id":generator_id,
        "retriever_id":retriever_id,
        "teacher_leak_reject_count":total_leak_reject,
        "teacher_leak_reject_fraction":leak_fraction,
        "teacher":teacher_detail,
        "nine_paper":{
            "historical_evidence_bearing_page_count":total_expected_pages,
            "coarse_selected_page_hits":total_coarse_hits,
            "coarse_selected_page_recall_ceiling":coarse_ceiling,
            "merged24_page_hits":total_merged_hits,
            "historical_evidence_bearing_page_recall_in_merged24":page_recall,
            "merged24_page_recall_fraction_of_ceiling":fraction,
        },
        "papers":paper_rows,
    }


def _verdict(pairs: dict[str,dict[str,Any]]) -> dict[str,Any]:
    promising=[]
    partial=[]
    for key,row in pairs.items():
        if row["generator_id"] not in QWEN_GENERATORS:
            continue
        teacher=row["teacher"]
        nine=row["nine_paper"]
        if (
            float(row["teacher_leak_reject_fraction"])<=0.20
            and int(teacher["supported_atoms_at_24"])>=10
            and int(teacher["known_miss_recovery_count"])==3
            and float(nine["merged24_page_recall_fraction_of_ceiling"])>=0.90
        ):
            promising.append(key)
        elif (
            int(teacher["supported_atoms_at_24"])>=8
            or int(teacher["known_miss_recovery_count"])==3
        ):
            partial.append(key)
    if promising:
        return {
            "wave7a_decision":"WAVE7A_PROMISING",
            "qualifying_pairs":promising,
            "rationale":"At least one generated-intent arm meets the frozen semantic, hard-miss, leak, and nine-paper coverage gates.",
        }
    if partial:
        return {
            "wave7a_decision":"WAVE7A_PARTIAL",
            "qualifying_pairs":partial,
            "rationale":"Generated intents materially improve retrieval but do not meet every frozen promotion gate.",
        }
    return {
        "wave7a_decision":"WAVE7A_REJECT",
        "qualifying_pairs":[],
        "rationale":"No generated-intent arm meets the frozen promising or partial retrieval gate.",
    }


def evaluate_wave7_intents(
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
            "config.json",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "colbert_linear.pt",
            "sparse_linear.pt",
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
        "operation":"LOCAL_EXTRACT_WAVE7_INTENT_GENERATION_EVAL",
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
