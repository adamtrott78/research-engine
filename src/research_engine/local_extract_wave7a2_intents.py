from __future__ import annotations

import hashlib
import pathlib
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
    _guard_generated_intents,
    _max_contiguous_copy,
)


def _source_context_with_region_ids(
    selected_regions: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    parts: list[str] = []
    used = 0
    for row in selected_regions:
        piece = (
            f"[region_id={row['region_id']} page={int(row['page_number'])}] "
            f"{str(row['display_text']).strip()}"
        )
        if used + len(piece) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                parts.append(piece[:remaining])
            break
        parts.append(piece)
        used += len(piece) + 1
    return "\n".join(parts)


def _source_aware_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    min_intents: int,
    max_intents: int,
) -> str:
    return (
        "You are an evidence-intent planner for scientific document extraction. "
        "Write specific answer-free QUESTIONS a retrieval system should answer "
        "from this paper. Do not answer them. Do not include numeric values, "
        "measurements, percentages, counts, exact result values, or copied answer "
        "phrases. Each question should target one concrete scientific fact. "
        "Return JSON only with top-level key intents. Each item must contain "
        "exactly facet and question; facet must be one of method,dataset,"
        "evaluation,robustness,limitation. "
        f"Return between {min_intents} and {max_intents} distinct intents.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"COARSE_SELECTED_SOURCE_CONTEXT:\n{context}\n"
    )


def _evidence_pass1_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    min_intents: int,
    max_intents: int,
) -> str:
    return (
        "You are converting scientific evidence into atomic retrieval questions. "
        "The source below is already a coarse-selected set of evidence-bearing "
        "regions. Inspect the evidence itself rather than filling a predefined "
        "scientific checklist. For each distinct scientifically meaningful fact "
        "that the evidence directly supports, emit an answer-free question that "
        "would retrieve that fact. One question must target exactly one fact. "
        "If one region contains multiple independent facts, split them into "
        "separate questions. Do not merge related facts merely because they share "
        "a topic. Do not answer the questions. Do not include digits, numeric "
        "values, measurements, percentages, counts, or copied answer phrases. "
        "Do not invent facts absent from the supplied evidence. Each item must "
        "cite exactly one supplied region_id that directly supports the intended "
        "answer. Return JSON only with top-level key intents; each item must have "
        "exactly region_id and question. "
        f"Return between {min_intents} and {max_intents} distinct atomic intents.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"COARSE_SELECTED_SOURCE_CONTEXT:\n{context}\n"
    )


def _evidence_gap_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    existing: list[dict[str, Any]],
    max_additional: int,
) -> str:
    rendered = "\n".join(
        f"- region_id={row['region_id']}: {row['question']}"
        for row in existing
    )
    return (
        "You are performing a coverage-gap audit for scientific evidence "
        "extraction. The source evidence and the questions already generated are "
        "shown below. Find scientifically meaningful atomic facts that are "
        "directly present in the evidence but are NOT yet targeted by any existing "
        "question. Pay special attention to compound passages: if an existing "
        "question covers one fact in a passage but another independent fact "
        "remains, generate a separate question for the uncovered fact. Do not use "
        "a fixed ontology or checklist; let the evidence determine what facts "
        "exist. Do not paraphrase or duplicate existing questions. Do not answer "
        "questions. Do not include digits, numeric values, measurements, "
        "percentages, counts, or copied answer phrases. Each new item must cite "
        "one supplied region_id that directly supports its intended answer. "
        "Return JSON only with top-level key intents; each item must have exactly "
        "region_id and question. It is valid to return an empty intents list if "
        "nothing substantive is uncovered. "
        f"Return at most {max_additional} additional intents.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"COARSE_SELECTED_SOURCE_CONTEXT:\n{context}\n\n"
        f"EXISTING_QUESTIONS:\n{rendered}\n"
    )


def _guard_evidence_intents(
    payload: dict[str, Any],
    *,
    source_context: str,
    selected_region_ids: set[str],
    min_intents: int,
    max_intents: int,
    min_chars: int,
    max_chars: int,
    max_copy_tokens: int,
    existing_questions: set[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    raw = payload.get("intents")
    if set(payload) != {"intents"} or not isinstance(raw, list):
        return {
            "status": "FAIL_CLOSED",
            "accepted": [],
            "rejected": [{"reason": "INVALID_TOP_LEVEL_SCHEMA"}],
            "raw_count": 0,
            "accepted_count": 0,
        }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen = {value.casefold() for value in (existing_questions or set())}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"region_id", "question"}:
            rejected.append({"index": idx, "reason": "INVALID_ITEM_SCHEMA"})
            continue
        region_id = str(item["region_id"]).strip()
        question = " ".join(str(item["question"]).split()).strip()
        reason = None
        if region_id not in selected_region_ids:
            reason = "UNKNOWN_REGION_ID"
        elif not (min_chars <= len(question) <= max_chars):
            reason = "QUESTION_LENGTH"
        elif any(ch.isdigit() for ch in question):
            reason = "NUMERIC_ANSWER_LEAK"
        elif _max_contiguous_copy(question, source_context) > max_copy_tokens:
            reason = "SOURCE_COPY"
        elif question.casefold() in seen:
            reason = "DUPLICATE"
        if reason:
            rejected.append(
                {
                    "index": idx,
                    "region_id": region_id,
                    "question": question,
                    "reason": reason,
                }
            )
            continue
        seen.add(question.casefold())
        accepted.append({"region_id": region_id, "question": question})
        if len(accepted) >= max_intents:
            break
    lower_ok = allow_empty or len(accepted) >= min_intents
    status = "PASS" if lower_ok and len(accepted) <= max_intents else "FAIL_CLOSED"
    return {
        "status": status,
        "accepted": accepted,
        "rejected": rejected,
        "raw_count": len(raw),
        "accepted_count": len(accepted),
    }


def generate_wave7a2_intents(
    *,
    fixture: dict[str, Any],
    contract: dict[str, Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    arms = {row["id"]: row for row in contract["generator_arms"]}
    source_arm = arms["QWEN35_SOURCE_AWARE_INTENTS_V1"]
    gap_arm = arms["QWEN35_EVIDENCE_GAP_INTENTS_V1"]
    schema = contract["evidence_gap_schema"]

    papers: list[dict[str, Any]] = []
    for source_paper in fixture["papers"]:
        paper = dict(source_paper)
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise RuntimeError(f"page-count mismatch for {paper['item_id']}")
        regions = exact_source_regions(
            pages,
            max_chars=int(fixture["qwen_config"]["max_region_chars"]),
        )
        broad = (
            str(fixture["topic"]) + " " + title + " " + str(paper["item_id"])
        ).strip()
        selected = rank_regions(broad, regions, top_k=32)[:12]
        context = _source_context_with_region_ids(
            selected,
            max_chars=int(gap_arm["max_input_chars"]),
        )
        papers.append(
            {
                **paper,
                "title": title,
                "acquisition": acquisition,
                "broad_query": broad,
                "selected_page_numbers": sorted(
                    {int(row["page_number"]) for row in selected}
                ),
                "selected_region_ids": [
                    str(row["region_id"]) for row in selected
                ],
                "source_context": context,
                "source_context_sha256": hashlib.sha256(
                    context.encode()
                ).hexdigest(),
                "generator_arms": {
                    "DETERMINISTIC_SCIENCE_SCHEMA_V1": {
                        "status": "PASS",
                        "accepted": [
                            dict(row) for row in contract["deterministic_intents"]
                        ],
                        "rejected": [],
                        "accepted_count": len(contract["deterministic_intents"]),
                        "raw_count": len(contract["deterministic_intents"]),
                        "usage": {
                            "prompt_tokens": 0,
                            "output_tokens": 0,
                            "wall_time_seconds": 0.0,
                        },
                    }
                },
            }
        )

    config = dict(fixture["qwen_config"])
    init_started = time.perf_counter()
    with GPUMemorySampler(expected_gpu_uuids) as gpu:
        llm = initialize_llm(
            model_path,
            tensor_parallel_size=int(config["tensor_parallel_size"]),
        )
        init_seconds = time.perf_counter() - init_started
        model_calls = 0
        prompt_tokens = 0
        output_tokens = 0
        inference_seconds = 0.0

        for paper in papers:
            context = str(paper["source_context"])
            selected_ids = set(paper["selected_region_ids"])

            # Frozen Wave-7 one-shot source-aware control.
            try:
                payload, usage = model_claims(
                    llm,
                    _source_aware_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=context,
                        min_intents=int(source_arm["min_intents"]),
                        max_intents=int(source_arm["max_intents"]),
                    ),
                    temperature=float(source_arm["temperature"]),
                    top_p=float(source_arm["top_p"]),
                    seed=int(source_arm["seed"]),
                    max_tokens=int(source_arm["max_output_tokens"]),
                )
                model_calls += 1
                prompt_tokens += int(usage["prompt_tokens"])
                output_tokens += int(usage["output_tokens"])
                inference_seconds += float(usage["wall_time_seconds"])
                guard = _guard_generated_intents(
                    payload,
                    source_context=context,
                    min_intents=int(source_arm["min_intents"]),
                    max_intents=int(source_arm["max_intents"]),
                    min_chars=int(schema["question_min_chars"]),
                    max_chars=int(schema["question_max_chars"]),
                    max_copy_tokens=int(
                        schema["no_verbatim_source_quote_longer_than_tokens"]
                    ),
                )
                paper["generator_arms"]["QWEN35_SOURCE_AWARE_INTENTS_V1"] = {
                    **guard,
                    "usage": usage,
                }
            except Exception as exc:
                paper["generator_arms"]["QWEN35_SOURCE_AWARE_INTENTS_V1"] = {
                    "status": "FAIL_CLOSED",
                    "accepted": [],
                    "rejected": [{"reason": f"{type(exc).__name__}:{exc}"}],
                    "accepted_count": 0,
                    "raw_count": 0,
                    "usage": {
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "wall_time_seconds": 0.0,
                    },
                }

            # Candidate pass 1: evidence -> atomic answerable questions.
            pass1_cfg = gap_arm["pass1"]
            pass2_cfg = gap_arm["pass2"]
            pass1_guard: dict[str, Any]
            pass2_guard: dict[str, Any]
            pass1_usage = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "wall_time_seconds": 0.0,
            }
            pass2_usage = dict(pass1_usage)
            try:
                payload1, pass1_usage = model_claims(
                    llm,
                    _evidence_pass1_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=context,
                        min_intents=int(pass1_cfg["min_intents"]),
                        max_intents=int(pass1_cfg["max_intents"]),
                    ),
                    temperature=float(gap_arm["temperature"]),
                    top_p=float(gap_arm["top_p"]),
                    seed=int(gap_arm["seed"]),
                    max_tokens=int(pass1_cfg["max_output_tokens"]),
                )
                model_calls += 1
                prompt_tokens += int(pass1_usage["prompt_tokens"])
                output_tokens += int(pass1_usage["output_tokens"])
                inference_seconds += float(pass1_usage["wall_time_seconds"])
                pass1_guard = _guard_evidence_intents(
                    payload1,
                    source_context=context,
                    selected_region_ids=selected_ids,
                    min_intents=int(pass1_cfg["min_intents"]),
                    max_intents=int(pass1_cfg["max_intents"]),
                    min_chars=int(schema["question_min_chars"]),
                    max_chars=int(schema["question_max_chars"]),
                    max_copy_tokens=int(
                        schema["no_verbatim_source_quote_longer_than_tokens"]
                    ),
                )
            except Exception as exc:
                pass1_guard = {
                    "status": "FAIL_CLOSED",
                    "accepted": [],
                    "rejected": [{"reason": f"{type(exc).__name__}:{exc}"}],
                    "accepted_count": 0,
                    "raw_count": 0,
                }

            # Candidate pass 2: evidence + pass-1 questions -> uncovered facts.
            try:
                existing = list(pass1_guard.get("accepted") or [])
                payload2, pass2_usage = model_claims(
                    llm,
                    _evidence_gap_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=context,
                        existing=existing,
                        max_additional=int(
                            pass2_cfg["max_additional_intents"]
                        ),
                    ),
                    temperature=float(gap_arm["temperature"]),
                    top_p=float(gap_arm["top_p"]),
                    seed=int(gap_arm["seed"]),
                    max_tokens=int(pass2_cfg["max_output_tokens"]),
                )
                model_calls += 1
                prompt_tokens += int(pass2_usage["prompt_tokens"])
                output_tokens += int(pass2_usage["output_tokens"])
                inference_seconds += float(pass2_usage["wall_time_seconds"])
                pass2_guard = _guard_evidence_intents(
                    payload2,
                    source_context=context,
                    selected_region_ids=selected_ids,
                    min_intents=0,
                    max_intents=int(
                        pass2_cfg["max_additional_intents"]
                    ),
                    min_chars=int(schema["question_min_chars"]),
                    max_chars=int(schema["question_max_chars"]),
                    max_copy_tokens=int(
                        schema["no_verbatim_source_quote_longer_than_tokens"]
                    ),
                    existing_questions={
                        str(row["question"]) for row in existing
                    },
                    allow_empty=True,
                )
            except Exception as exc:
                pass2_guard = {
                    "status": "FAIL_CLOSED",
                    "accepted": [],
                    "rejected": [{"reason": f"{type(exc).__name__}:{exc}"}],
                    "accepted_count": 0,
                    "raw_count": 0,
                }

            combined = (
                list(pass1_guard.get("accepted") or [])
                + list(pass2_guard.get("accepted") or [])
            )[: int(gap_arm["max_total_intents"])]
            candidate_status = (
                "PASS"
                if pass1_guard["status"] == "PASS"
                and pass2_guard["status"] == "PASS"
                and len(combined) >= int(pass1_cfg["min_intents"])
                else "FAIL_CLOSED"
            )
            paper["generator_arms"]["QWEN35_EVIDENCE_GAP_INTENTS_V1"] = {
                "status": candidate_status,
                "accepted": combined,
                "accepted_count": len(combined),
                "raw_count": int(pass1_guard.get("raw_count") or 0)
                + int(pass2_guard.get("raw_count") or 0),
                "rejected": (
                    list(pass1_guard.get("rejected") or [])
                    + list(pass2_guard.get("rejected") or [])
                ),
                "pass1": pass1_guard,
                "pass2": pass2_guard,
                "usage": {
                    "prompt_tokens": int(pass1_usage["prompt_tokens"])
                    + int(pass2_usage["prompt_tokens"]),
                    "output_tokens": int(pass1_usage["output_tokens"])
                    + int(pass2_usage["output_tokens"]),
                    "wall_time_seconds": float(pass1_usage["wall_time_seconds"])
                    + float(pass2_usage["wall_time_seconds"]),
                },
            }

    for paper in papers:
        paper.pop("source_context", None)

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_WAVE7A2_EVIDENCE_CONDITIONED_GENERATION",
        "benchmark_id": contract["benchmark_id"],
        "baseline_sha": contract["baseline_sha"],
        "model_id": "qwen3.5-4b",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "model_path": model_path,
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "metrics": {
            "paper_count": len(papers),
            "model_calls": model_calls,
            "model_init_seconds": init_seconds,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "inference_seconds": inference_seconds,
            "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
        },
        "papers": papers,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
    }
