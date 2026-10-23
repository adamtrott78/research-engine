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
from research_engine.local_extract_wave7_intents import (
    _generator_prompt,
    _guard_generated_intents,
    _max_contiguous_copy,
    _source_context,
)


ANCHOR_ROLES={"problem","contribution","result","limitation_or_future_work"}


def _parse_structured_json(text: str) -> dict[str,Any]:
    parsed=json.loads(str(text).strip())
    if not isinstance(parsed,dict):
        raise RuntimeError("structured model output root must be an object")
    return parsed


def _anchor_json_schema(
    *,
    min_anchors: int,
    max_anchors: int,
) -> dict[str,Any]:
    return {
        "type":"object",
        "additionalProperties":False,
        "required":["anchors"],
        "properties":{
            "anchors":{
                "type":"array",
                "minItems":min_anchors,
                "maxItems":max_anchors,
                "items":{
                    "type":"object",
                    "additionalProperties":False,
                    "required":["anchor_id","role","descriptor"],
                    "properties":{
                        "anchor_id":{"type":"string","pattern":r"^A[0-9]{2}$"},
                        "role":{
                            "type":"string",
                            "enum":[
                                "problem",
                                "contribution",
                                "result",
                                "limitation_or_future_work",
                            ],
                        },
                        "descriptor":{"type":"string"},
                    },
                },
            }
        },
    }


def _obligation_json_schema(
    *,
    anchors: list[dict[str,Any]],
    min_intents: int,
    max_intents: int,
) -> dict[str,Any]:
    anchor_ids=[
        str(row["anchor_id"])
        for row in anchors
        if str(row.get("anchor_id") or "")
    ]
    anchor_id_schema: dict[str,Any]={"type":"string"}
    if anchor_ids:
        anchor_id_schema["enum"]=anchor_ids
    else:
        anchor_id_schema["pattern"]=r"^A[0-9]{2}$"
    return {
        "type":"object",
        "additionalProperties":False,
        "required":["intents"],
        "properties":{
            "intents":{
                "type":"array",
                "minItems":min_intents,
                "maxItems":max_intents,
                "items":{
                    "type":"object",
                    "additionalProperties":False,
                    "required":["anchor_id","question"],
                    "properties":{
                        "anchor_id":anchor_id_schema,
                        "question":{"type":"string"},
                    },
                },
            }
        },
    }


def _model_json_schema(
    llm: Any,
    prompt: str,
    *,
    json_schema: dict[str,Any],
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str,Any],dict[str,Any]]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params=SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(
            json=json_schema,
            disable_additional_properties=True,
        ),
    )
    started=time.perf_counter()
    outputs=llm.chat(
        [{"role":"user","content":prompt}],
        sampling_params=params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking":False},
    )
    seconds=time.perf_counter()-started
    output=outputs[0]
    generated=output.outputs[0]
    parsed=_parse_structured_json(generated.text)
    return parsed,{
        "wall_time_seconds":seconds,
        "prompt_tokens":len(getattr(output,"prompt_token_ids",[]) or []),
        "output_tokens":len(generated.token_ids),
        "generated_text_sha256":hashlib.sha256(
            generated.text.encode("utf-8")
        ).hexdigest(),
    }


def _author_salience_context(
    pages: list[dict[str,Any]],
    *,
    frontmatter_max_chars: int,
    tail_max_chars: int,
) -> str:
    first=str(pages[0].get("text") or "")[:frontmatter_max_chars] if pages else ""
    last=str(pages[-1].get("text") or "")[-tail_max_chars:] if pages else ""
    if len(pages)==1:
        last=""
    return (
        "FRONT_MATTER\n"
        + first
        + ("\n\nTAIL_CONTEXT\n"+last if last else "")
    )


def _anchor_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    min_anchors: int,
    max_anchors: int,
) -> str:
    return (
        "You are identifying AUTHOR-FRAMED SCIENTIFIC SALIENCE in one research "
        "paper. Read the title plus front matter and tail context. Identify only "
        "the small set of claims/problems/results/limitations that the authors "
        "themselves foreground as central to what the paper contributes, proves, "
        "evaluates, or leaves unresolved. Do NOT enumerate every answerable fact. "
        "Do NOT reward background definitions, cited prior work, generic field "
        "motivation, or incidental implementation detail unless the authors frame "
        "it as part of their own contribution.\n\n"
        "Use these rhetorical roles, which describe discourse function rather "
        "than domain ontology: problem, contribution, result, "
        "limitation_or_future_work. Each descriptor should be a concise paraphrase "
        "of what matters scientifically. It may summarize author-stated facts, "
        "but should not copy long spans verbatim. Prefer multiple distinct anchors "
        "when the paper makes multiple contributions or distinct result/limitation "
        "claims. Return JSON only with top-level key anchors. Every item must have "
        "exactly anchor_id, role, descriptor. anchor_id must be A01, A02, ... in "
        "order. "
        f"Return between {min_anchors} and {max_anchors} anchors.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"AUTHOR_SALIENCE_CONTEXT:\n{context}\n"
    )


def _obligation_prompt(
    *,
    topic: str,
    title: str,
    anchors: list[dict[str,Any]],
    selected_context: str,
    min_intents: int,
    max_intents: int,
    max_per_anchor: int,
) -> str:
    rendered="\n".join(
        f"- {row['anchor_id']} [{row['role']}]: {row['descriptor']}"
        for row in anchors
    )
    return (
        "You are turning author-framed scientific contribution anchors into "
        "ATOMIC EVIDENCE OBLIGATIONS for retrieval. The anchors define what is "
        "scientifically salient. The selected source context is available only "
        "to ensure each question corresponds to evidence actually present in the "
        "paper.\n\n"
        "For each anchor, ask the concrete answer-free questions that must be "
        "answered to substantiate, operationalize, or precisely characterize that "
        "anchor. Questions may target mechanism, training/supervision, data or "
        "test conditions, experimental setup, comparison, result relationship, "
        "failure mode, limitation, or future-work cause ONLY when that information "
        "is relevant to the anchor. Do not mechanically fill a global checklist. "
        "Let each paper-specific anchor determine its own evidence needs.\n\n"
        "Every question must target exactly one fact. Split independent facts "
        "even when they occur in one passage. Do not answer the questions. Do not "
        "include digits, measurements, counts, percentages, numeric result values, "
        "or copied answer phrases. Do not create questions for background facts "
        "that do not substantiate an anchor. Do not merely paraphrase the anchor "
        "as a question; drill down to the concrete evidence needed to support it. "
        "Every item must reference an existing anchor_id. Return JSON only with "
        "top-level key intents. Every item must contain exactly anchor_id and "
        "question. "
        f"Return between {min_intents} and {max_intents} total intents and no more "
        f"than {max_per_anchor} intents for any one anchor.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"AUTHOR_ANCHORS:\n{rendered}\n\n"
        f"COARSE_SELECTED_SOURCE_CONTEXT:\n{selected_context}\n"
    )


def _guard_anchors(
    payload: dict[str,Any],
    *,
    source_context: str,
    min_anchors: int,
    max_anchors: int,
    min_chars: int,
    max_chars: int,
    max_copy_tokens: int,
) -> dict[str,Any]:
    raw=payload.get("anchors")
    if set(payload)!={"anchors"} or not isinstance(raw,list):
        return {
            "status":"FAIL_CLOSED",
            "accepted":[],
            "rejected":[{"reason":"INVALID_TOP_LEVEL_SCHEMA"}],
            "raw_count":0,
            "accepted_count":0,
        }
    accepted=[]
    rejected=[]
    seen_ids=set()
    seen_desc=set()
    for idx,item in enumerate(raw):
        if not isinstance(item,dict) or set(item)!={"anchor_id","role","descriptor"}:
            rejected.append({"index":idx,"reason":"INVALID_ITEM_SCHEMA"})
            continue
        anchor_id=str(item["anchor_id"]).strip().upper()
        role=str(item["role"]).strip().lower()
        descriptor=" ".join(str(item["descriptor"]).split()).strip()
        reason=None
        if not re.fullmatch(r"A\d{2}",anchor_id):
            reason="INVALID_ANCHOR_ID"
        elif anchor_id in seen_ids:
            reason="DUPLICATE_ANCHOR_ID"
        elif role not in ANCHOR_ROLES:
            reason="INVALID_ROLE"
        elif not (min_chars<=len(descriptor)<=max_chars):
            reason="DESCRIPTOR_LENGTH"
        elif _max_contiguous_copy(descriptor,source_context)>max_copy_tokens:
            reason="SOURCE_COPY"
        elif descriptor.casefold() in seen_desc:
            reason="DUPLICATE_DESCRIPTOR"
        if reason:
            rejected.append({
                "index":idx,
                "anchor_id":anchor_id,
                "role":role,
                "descriptor":descriptor,
                "reason":reason,
            })
            continue
        seen_ids.add(anchor_id)
        seen_desc.add(descriptor.casefold())
        accepted.append({
            "anchor_id":anchor_id,
            "role":role,
            "descriptor":descriptor,
        })
        if len(accepted)>=max_anchors:
            break
    status="PASS" if min_anchors<=len(accepted)<=max_anchors else "FAIL_CLOSED"
    return {
        "status":status,
        "accepted":accepted,
        "rejected":rejected,
        "raw_count":len(raw),
        "accepted_count":len(accepted),
    }


def _guard_obligations(
    payload: dict[str,Any],
    *,
    anchors: list[dict[str,Any]],
    source_context: str,
    min_intents: int,
    max_intents: int,
    max_per_anchor: int,
    min_chars: int,
    max_chars: int,
    max_copy_tokens: int,
) -> dict[str,Any]:
    raw=payload.get("intents")
    if set(payload)!={"intents"} or not isinstance(raw,list):
        return {
            "status":"FAIL_CLOSED",
            "accepted":[],
            "rejected":[{"reason":"INVALID_TOP_LEVEL_SCHEMA"}],
            "raw_count":0,
            "accepted_count":0,
        }
    anchor_ids={str(x["anchor_id"]) for x in anchors}
    accepted=[]
    rejected=[]
    seen=set()
    per_anchor={anchor_id:0 for anchor_id in anchor_ids}
    for idx,item in enumerate(raw):
        if not isinstance(item,dict) or set(item)!={"anchor_id","question"}:
            rejected.append({"index":idx,"reason":"INVALID_ITEM_SCHEMA"})
            continue
        anchor_id=str(item["anchor_id"]).strip().upper()
        question=" ".join(str(item["question"]).split()).strip()
        reason=None
        if anchor_id not in anchor_ids:
            reason="UNKNOWN_ANCHOR_ID"
        elif not (min_chars<=len(question)<=max_chars):
            reason="QUESTION_LENGTH"
        elif any(ch.isdigit() for ch in question):
            reason="NUMERIC_ANSWER_LEAK"
        elif _max_contiguous_copy(question,source_context)>max_copy_tokens:
            reason="SOURCE_COPY"
        elif question.casefold() in seen:
            reason="DUPLICATE"
        elif per_anchor[anchor_id]>=max_per_anchor:
            reason="ANCHOR_INTENT_LIMIT"
        if reason:
            rejected.append({
                "index":idx,
                "anchor_id":anchor_id,
                "question":question,
                "reason":reason,
            })
            continue
        seen.add(question.casefold())
        per_anchor[anchor_id]+=1
        accepted.append({"anchor_id":anchor_id,"question":question})
        if len(accepted)>=max_intents:
            break
    status="PASS" if min_intents<=len(accepted)<=max_intents else "FAIL_CLOSED"
    return {
        "status":status,
        "accepted":accepted,
        "rejected":rejected,
        "raw_count":len(raw),
        "accepted_count":len(accepted),
        "accepted_per_anchor":per_anchor,
    }


def generate_wave7a3_intents(
    *,
    fixture: dict[str,Any],
    contract: dict[str,Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str,Any]:
    workdir.mkdir(parents=True,exist_ok=True)
    arms={row["id"]:row for row in contract["generator_arms"]}
    control=arms["QWEN35_SOURCE_AWARE_INTENTS_V1"]
    candidate=arms["QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1"]
    anchor_schema=contract["anchor_schema"]
    obligation_schema=contract["obligation_schema"]

    papers=[]
    for source_paper in fixture["papers"]:
        paper=dict(source_paper)
        pdf_path=workdir/"pdfs"/f"{paper['arxiv_id']}.pdf"
        acquisition=download_exact_pdf(paper,pdf_path)
        pages,title=extract_pdf_pages(pdf_path)
        if len(pages)!=int(paper["total_pdf_pages"]):
            raise RuntimeError(f"page-count mismatch for {paper['item_id']}")
        regions=exact_source_regions(
            pages,
            max_chars=int(fixture["qwen_config"]["max_region_chars"]),
        )
        broad=(str(fixture["topic"])+" "+title+" "+str(paper["item_id"])).strip()
        selected=rank_regions(broad,regions,top_k=32)[:12]
        control_context=_source_context(
            selected,
            max_chars=int(control["max_input_chars"]),
        )
        selected_context=_source_context(
            selected,
            max_chars=int(candidate["obligation_context"]["selected_source_max_chars"]),
        )
        author_context=_author_salience_context(
            pages,
            frontmatter_max_chars=int(candidate["salience_context"]["frontmatter_max_chars"]),
            tail_max_chars=int(candidate["salience_context"]["tail_max_chars"]),
        )
        papers.append({
            **paper,
            "title":title,
            "acquisition":acquisition,
            "broad_query":broad,
            "selected_page_numbers":sorted({int(row["page_number"]) for row in selected}),
            "selected_region_ids":[str(row["region_id"]) for row in selected],
            "control_context":control_context,
            "selected_context":selected_context,
            "author_context":author_context,
            "control_context_sha256":hashlib.sha256(control_context.encode()).hexdigest(),
            "selected_context_sha256":hashlib.sha256(selected_context.encode()).hexdigest(),
            "author_context_sha256":hashlib.sha256(author_context.encode()).hexdigest(),
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
        llm=initialize_llm(model_path,tensor_parallel_size=int(config["tensor_parallel_size"]))
        init_seconds=time.perf_counter()-init_started
        model_calls=0
        prompt_tokens=0
        output_tokens=0
        inference_seconds=0.0

        for paper in papers:
            control_context=str(paper["control_context"])
            selected_context=str(paper["selected_context"])
            author_context=str(paper["author_context"])

            # Exact one-shot source-aware control.
            try:
                payload,usage=model_claims(
                    llm,
                    _generator_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context_label="COARSE_SELECTED_SOURCE_CONTEXT",
                        context=control_context,
                        min_intents=int(control["min_intents"]),
                        max_intents=int(control["max_intents"]),
                    ),
                    temperature=float(control["temperature"]),
                    top_p=float(control["top_p"]),
                    seed=int(control["seed"]),
                    max_tokens=int(control["max_output_tokens"]),
                )
                model_calls+=1
                prompt_tokens+=int(usage["prompt_tokens"])
                output_tokens+=int(usage["output_tokens"])
                inference_seconds+=float(usage["wall_time_seconds"])
                guard=_guard_generated_intents(
                    payload,
                    source_context=control_context,
                    min_intents=int(control["min_intents"]),
                    max_intents=int(control["max_intents"]),
                    min_chars=int(obligation_schema["question_min_chars"]),
                    max_chars=int(obligation_schema["question_max_chars"]),
                    max_copy_tokens=int(obligation_schema["no_verbatim_source_quote_longer_than_tokens"]),
                )
                paper["generator_arms"]["QWEN35_SOURCE_AWARE_INTENTS_V1"]={**guard,"usage":usage}
            except Exception as exc:
                paper["generator_arms"]["QWEN35_SOURCE_AWARE_INTENTS_V1"]={
                    "status":"FAIL_CLOSED","accepted":[],
                    "rejected":[{"reason":f"{type(exc).__name__}:{exc}"}],
                    "accepted_count":0,"raw_count":0,
                    "usage":{"prompt_tokens":0,"output_tokens":0,"wall_time_seconds":0.0},
                }

            anchor_cfg=candidate["salience_context"]
            anchor_usage={"prompt_tokens":0,"output_tokens":0,"wall_time_seconds":0.0}
            obligation_usage=dict(anchor_usage)
            anchor_interface_status="NOT_ATTEMPTED"
            anchor_interface_error=None
            obligation_interface_status="NOT_ATTEMPTED"
            obligation_interface_error=None
            try:
                anchor_payload,anchor_usage=_model_json_schema(
                    llm,
                    _anchor_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=author_context,
                        min_anchors=int(anchor_cfg["anchor_min"]),
                        max_anchors=int(anchor_cfg["anchor_max"]),
                    ),
                    json_schema=_anchor_json_schema(
                        min_anchors=int(anchor_cfg["anchor_min"]),
                        max_anchors=int(anchor_cfg["anchor_max"]),
                    ),
                    temperature=float(candidate["temperature"]),
                    top_p=float(candidate["top_p"]),
                    seed=int(candidate["seed"]),
                    max_tokens=int(anchor_cfg["max_output_tokens"]),
                )
                anchor_interface_status="PASS"
                model_calls+=1
                prompt_tokens+=int(anchor_usage["prompt_tokens"])
                output_tokens+=int(anchor_usage["output_tokens"])
                inference_seconds+=float(anchor_usage["wall_time_seconds"])
                anchor_guard=_guard_anchors(
                    anchor_payload,
                    source_context=author_context,
                    min_anchors=int(anchor_cfg["anchor_min"]),
                    max_anchors=int(anchor_cfg["anchor_max"]),
                    min_chars=int(anchor_schema["descriptor_min_chars"]),
                    max_chars=int(anchor_schema["descriptor_max_chars"]),
                    max_copy_tokens=int(anchor_schema["no_verbatim_source_quote_longer_than_tokens"]),
                )
            except Exception as exc:
                anchor_interface_status="FAIL_CLOSED"
                anchor_interface_error=f"{type(exc).__name__}:{exc}"
                anchor_guard={
                    "status":"FAIL_CLOSED","accepted":[],
                    "rejected":[{"reason":f"{type(exc).__name__}:{exc}"}],
                    "accepted_count":0,"raw_count":0,
                }

            obligation_cfg=candidate["obligation_context"]
            try:
                anchors=list(anchor_guard.get("accepted") or [])
                obligation_payload,obligation_usage=_model_json_schema(
                    llm,
                    _obligation_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        anchors=anchors,
                        selected_context=selected_context,
                        min_intents=int(obligation_cfg["min_intents"]),
                        max_intents=int(obligation_cfg["max_intents"]),
                        max_per_anchor=int(obligation_cfg["max_intents_per_anchor"]),
                    ),
                    json_schema=_obligation_json_schema(
                        anchors=anchors,
                        min_intents=int(obligation_cfg["min_intents"]),
                        max_intents=int(obligation_cfg["max_intents"]),
                    ),
                    temperature=float(candidate["temperature"]),
                    top_p=float(candidate["top_p"]),
                    seed=int(candidate["seed"]),
                    max_tokens=int(obligation_cfg["max_output_tokens"]),
                )
                obligation_interface_status="PASS"
                model_calls+=1
                prompt_tokens+=int(obligation_usage["prompt_tokens"])
                output_tokens+=int(obligation_usage["output_tokens"])
                inference_seconds+=float(obligation_usage["wall_time_seconds"])
                combined_context=author_context+"\n\n"+selected_context
                obligation_guard=_guard_obligations(
                    obligation_payload,
                    anchors=anchors,
                    source_context=combined_context,
                    min_intents=int(obligation_cfg["min_intents"]),
                    max_intents=int(obligation_cfg["max_intents"]),
                    max_per_anchor=int(obligation_cfg["max_intents_per_anchor"]),
                    min_chars=int(obligation_schema["question_min_chars"]),
                    max_chars=int(obligation_schema["question_max_chars"]),
                    max_copy_tokens=int(obligation_schema["no_verbatim_source_quote_longer_than_tokens"]),
                )
            except Exception as exc:
                obligation_interface_status="FAIL_CLOSED"
                obligation_interface_error=f"{type(exc).__name__}:{exc}"
                obligation_guard={
                    "status":"FAIL_CLOSED","accepted":[],
                    "rejected":[{"reason":f"{type(exc).__name__}:{exc}"}],
                    "accepted_count":0,"raw_count":0,
                }

            status="PASS" if anchor_guard["status"]=="PASS" and obligation_guard["status"]=="PASS" else "FAIL_CLOSED"
            paper["generator_arms"]["QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1"]={
                "status":status,
                "accepted":list(obligation_guard.get("accepted") or []),
                "accepted_count":int(obligation_guard.get("accepted_count") or 0),
                "raw_count":int(obligation_guard.get("raw_count") or 0),
                "rejected":list(obligation_guard.get("rejected") or []),
                "anchors":list(anchor_guard.get("accepted") or []),
                "anchor_count":int(anchor_guard.get("accepted_count") or 0),
                "anchor_rejected":list(anchor_guard.get("rejected") or []),
                "anchor_guard_status":anchor_guard["status"],
                "anchor_interface_status":anchor_interface_status,
                "anchor_interface_error":anchor_interface_error,
                "obligation_guard_status":obligation_guard["status"],
                "obligation_interface_status":obligation_interface_status,
                "obligation_interface_error":obligation_interface_error,
                "accepted_per_anchor":dict(obligation_guard.get("accepted_per_anchor") or {}),
                "usage":{
                    "prompt_tokens":int(anchor_usage["prompt_tokens"])+int(obligation_usage["prompt_tokens"]),
                    "output_tokens":int(anchor_usage["output_tokens"])+int(obligation_usage["output_tokens"]),
                    "wall_time_seconds":float(anchor_usage["wall_time_seconds"])+float(obligation_usage["wall_time_seconds"]),
                },
            }

    for paper in papers:
        paper.pop("control_context",None)
        paper.pop("selected_context",None)
        paper.pop("author_context",None)

    return {
        "schema_version":1,
        "operation":"LOCAL_EXTRACT_WAVE7A3_AUTHOR_ANCHORED_GENERATION",
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
