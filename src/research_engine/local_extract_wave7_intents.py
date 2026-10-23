from __future__ import annotations

import hashlib
import json
import pathlib
import re
import time
from typing import Any

from research_engine.local_extract_wave1 import (
    GPUMemorySampler,
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    initialize_llm,
    model_claims,
    rank_regions,
)


FACETS={"method","dataset","evaluation","robustness","limitation"}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(value).lower())


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+","",str(value).lower())


def _max_contiguous_copy(question: str, source: str) -> int:
    q=_tokens(question)
    s=_tokens(source)
    if not q or not s:
        return 0
    best=0
    positions: dict[str,list[int]]={}
    for i,token in enumerate(s):
        positions.setdefault(token,[]).append(i)
    for qi,token in enumerate(q):
        for si in positions.get(token,[]):
            length=0
            while qi+length<len(q) and si+length<len(s) and q[qi+length]==s[si+length]:
                length+=1
            best=max(best,length)
    return best


def _source_context(
    selected_regions: list[dict[str,Any]],
    *,
    max_chars: int,
) -> str:
    parts=[]
    used=0
    for row in selected_regions:
        piece=f"[page {int(row['page_number'])}] {str(row['display_text'])}".strip()
        if used+len(piece)>max_chars:
            remaining=max_chars-used
            if remaining>0:
                parts.append(piece[:remaining])
            break
        parts.append(piece)
        used+=len(piece)+1
    return "\n".join(parts)


def _frontmatter(pages: list[dict[str,Any]], *, max_chars: int) -> str:
    if not pages:
        return ""
    text=str(pages[0].get("text") or "")
    return text[:max_chars]


def _generator_prompt(
    *,
    topic: str,
    title: str,
    context_label: str,
    context: str,
    min_intents: int,
    max_intents: int,
) -> str:
    return (
        "You are an evidence-intent planner for scientific document extraction. "
        "Your only job is to write specific QUESTIONS that a retrieval system "
        "should answer from this paper. Do not answer the questions. Do not "
        "include any numeric values, measurements, counts, percentages, exact "
        "result values, or copied answer phrases. Avoid quoting the paper. "
        "Questions should be specific enough to retrieve one concrete scientific "
        "fact, not vague prompts like 'what are the results?'. Cover useful facts "
        "across method, dataset, evaluation, robustness, and limitations when "
        "the supplied context supports those categories. Prefer questions about "
        "how something is done, what is compared or measured, what setup is used, "
        "what result relationship is reported, and what limitation/future work "
        "is identified. Do not invent facts absent from the supplied context. "
        f"Return between {min_intents} and {max_intents} distinct intents. "
        "Return JSON only with top-level key intents. Each intents item must have "
        "exactly facet and question. facet must be one of method,dataset,"
        "evaluation,robustness,limitation.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_TITLE:\n{title}\n\n"
        f"{context_label}:\n{context}\n"
    )


def _guard_generated_intents(
    payload: dict[str,Any],
    *,
    source_context: str,
    min_intents: int,
    max_intents: int,
    min_chars: int,
    max_chars: int,
    max_copy_tokens: int,
) -> dict[str,Any]:
    raw=payload.get("intents")
    if set(payload)!={"intents"} or not isinstance(raw,list):
        return {"status":"FAIL_CLOSED","accepted":[],"rejected":[{"reason":"INVALID_TOP_LEVEL_SCHEMA"}]}
    accepted=[]
    rejected=[]
    seen=set()
    for idx,item in enumerate(raw):
        if not isinstance(item,dict) or set(item)!={"facet","question"}:
            rejected.append({"index":idx,"reason":"INVALID_ITEM_SCHEMA"})
            continue
        facet=str(item["facet"]).strip().lower()
        question=" ".join(str(item["question"]).split()).strip()
        reason=None
        if facet not in FACETS:
            reason="INVALID_FACET"
        elif not (min_chars<=len(question)<=max_chars):
            reason="QUESTION_LENGTH"
        elif any(ch.isdigit() for ch in question):
            reason="NUMERIC_ANSWER_LEAK"
        elif _max_contiguous_copy(question,source_context)>max_copy_tokens:
            reason="SOURCE_COPY"
        elif question.casefold() in seen:
            reason="DUPLICATE"
        if reason:
            rejected.append({"index":idx,"facet":facet,"question":question,"reason":reason})
            continue
        seen.add(question.casefold())
        accepted.append({"facet":facet,"question":question})
        if len(accepted)>=max_intents:
            break
    status="PASS" if min_intents<=len(accepted)<=max_intents else "FAIL_CLOSED"
    return {
        "status":status,
        "accepted":accepted,
        "rejected":rejected,
        "accepted_count":len(accepted),
        "raw_count":len(raw),
    }


def generate_wave7_intents(
    *,
    fixture: dict[str,Any],
    contract: dict[str,Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str,Any]:
    workdir.mkdir(parents=True,exist_ok=True)
    qwen_arms={
        row["id"]:row
        for row in contract["generator_arms"]
        if row.get("model_id")=="qwen3.5-4b"
    }
    papers=[]
    for source_paper in fixture["papers"]:
        paper=dict(source_paper)
        pdf_path=workdir/"pdfs"/f"{paper['arxiv_id']}.pdf"
        acquisition=download_exact_pdf(paper,pdf_path)
        pages,title=extract_pdf_pages(pdf_path)
        if len(pages)!=int(paper["total_pdf_pages"]):
            raise RuntimeError(f"page-count mismatch for {paper['item_id']}")
        regions=exact_source_regions(pages,max_chars=int(fixture["qwen_config"]["max_region_chars"]))
        broad_query=(str(fixture["topic"])+" "+title+" "+str(paper["item_id"])).strip()
        selected=rank_regions(broad_query,regions,top_k=32)[:12]
        source_context=_source_context(
            selected,
            max_chars=int(qwen_arms["QWEN35_SOURCE_AWARE_INTENTS_V1"]["max_input_chars"]),
        )
        frontmatter=_frontmatter(
            pages,
            max_chars=int(qwen_arms["QWEN35_FRONTMATTER_INTENTS_V1"]["max_input_chars"]),
        )
        papers.append({
            **paper,
            "title":title,
            "acquisition":acquisition,
            "broad_query":broad_query,
            "selected_page_numbers":sorted({int(row["page_number"]) for row in selected}),
            "selected_region_ids":[str(row["region_id"]) for row in selected],
            "frontmatter_context":frontmatter,
            "source_context":source_context,
            "frontmatter_sha256":hashlib.sha256(frontmatter.encode()).hexdigest(),
            "source_context_sha256":hashlib.sha256(source_context.encode()).hexdigest(),
            "generator_arms":{
                "DETERMINISTIC_SCIENCE_SCHEMA_V1":{
                    "status":"PASS",
                    "accepted":[dict(row) for row in contract["deterministic_intents"]],
                    "rejected":[],
                    "accepted_count":len(contract["deterministic_intents"]),
                    "raw_count":len(contract["deterministic_intents"]),
                    "usage":{"prompt_tokens":0,"output_tokens":0,"wall_time_seconds":0.0},
                }
            },
        })

    config=dict(fixture["qwen_config"])
    init_started=time.perf_counter()
    with GPUMemorySampler(expected_gpu_uuids) as gpu:
        llm=initialize_llm(
            model_path,
            tensor_parallel_size=int(config["tensor_parallel_size"]),
        )
        init_seconds=time.perf_counter()-init_started
        model_calls=0
        prompt_tokens=0
        output_tokens=0
        inference_seconds=0.0
        for paper in papers:
            for arm_id,arm in qwen_arms.items():
                if arm_id=="QWEN35_FRONTMATTER_INTENTS_V1":
                    label="PAPER_FRONT_MATTER"
                    context=str(paper["frontmatter_context"])
                elif arm_id=="QWEN35_SOURCE_AWARE_INTENTS_V1":
                    label="COARSE_SELECTED_SOURCE_CONTEXT"
                    context=str(paper["source_context"])
                else:
                    raise RuntimeError(f"unknown generator arm {arm_id}")
                try:
                    payload,usage=model_claims(
                        llm,
                        _generator_prompt(
                            topic=str(fixture["topic"]),
                            title=str(paper["title"]),
                            context_label=label,
                            context=context,
                            min_intents=int(arm["min_intents"]),
                            max_intents=int(arm["max_intents"]),
                        ),
                        temperature=float(arm["temperature"]),
                        top_p=float(arm["top_p"]),
                        seed=int(arm["seed"]),
                        max_tokens=int(arm["max_output_tokens"]),
                    )
                    model_calls+=1
                    prompt_tokens+=int(usage["prompt_tokens"])
                    output_tokens+=int(usage["output_tokens"])
                    inference_seconds+=float(usage["wall_time_seconds"])
                    guard=_guard_generated_intents(
                        payload,
                        source_context=context,
                        min_intents=int(arm["min_intents"]),
                        max_intents=int(arm["max_intents"]),
                        min_chars=int(contract["generated_intent_schema"]["question_min_chars"]),
                        max_chars=int(contract["generated_intent_schema"]["question_max_chars"]),
                        max_copy_tokens=int(contract["generated_intent_schema"]["no_verbatim_source_quote_longer_than_tokens"]),
                    )
                    paper["generator_arms"][arm_id]={**guard,"usage":usage}
                except Exception as exc:
                    paper["generator_arms"][arm_id]={
                        "status":"FAIL_CLOSED",
                        "accepted":[],
                        "rejected":[{"reason":f"{type(exc).__name__}:{exc}"}],
                        "accepted_count":0,
                        "raw_count":0,
                        "usage":{"prompt_tokens":0,"output_tokens":0,"wall_time_seconds":0.0},
                    }

    for paper in papers:
        paper.pop("frontmatter_context",None)
        paper.pop("source_context",None)

    return {
        "schema_version":1,
        "operation":"LOCAL_EXTRACT_WAVE7_INTENT_GENERATION",
        "benchmark_id":contract["benchmark_id"],
        "baseline_sha":contract["baseline_sha"],
        "model_id":"qwen3.5-4b",
        "model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "model_path":model_path,
        "tensor_parallel_size":int(config["tensor_parallel_size"]),
        "metrics":{
            "paper_count":len(papers),
            "model_calls":model_calls,
            "model_init_seconds":init_seconds,
            "prompt_tokens":prompt_tokens,
            "output_tokens":output_tokens,
            "inference_seconds":inference_seconds,
            "peak_gpu_memory_mib_by_uuid":dict(gpu.peak),
        },
        "papers":papers,
        "scientific_state_materialized":False,
        "production_routing_changed":False,
        "qwen_verify_changed":False,
        "wave7b_started":False,
    }
