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
    _generator_prompt,
    _guard_generated_intents,
    _max_contiguous_copy,
    _source_context,
)


def _alias_source_context(
    selected_regions: list[dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    aliases: dict[str, str] = {}
    used = 0
    for idx, row in enumerate(selected_regions, start=1):
        alias = f"R{idx:02d}"
        aliases[alias] = str(row["region_id"])
        piece = (
            f"REGION {alias}\n"
            f"PAGE {int(row['page_number'])}\n"
            f"TEXT {str(row['display_text']).strip()}\n"
            f"END REGION {alias}"
        )
        if used + len(piece) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                parts.append(piece[:remaining])
            break
        parts.append(piece)
        used += len(piece) + 2
    return "\n\n".join(parts), aliases


def _pass1_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    aliases: list[str],
    min_intents: int,
    max_intents: int,
) -> str:
    alias_text = ", ".join(aliases)
    return (
        "You are converting selected scientific evidence into atomic retrieval "
        "questions. Let the supplied evidence determine the facts; do not fill a "
        "predefined ontology or checklist. Identify distinct scientifically "
        "meaningful facts directly supported by the regions. For every fact you "
        "choose, write ONE answer-free question that targets exactly that fact. "
        "If a passage states multiple independent facts, split them into separate "
        "questions. Do not merge related facts merely because they share a topic. "
        "Do not answer the questions. Do not include digits, measurements, counts, "
        "percentages, numeric results, or copied answer phrases in question text. "
        "Do not invent facts absent from the evidence. Each item must contain a "
        "region_alias chosen from the supplied aliases and the question. "
        f"Allowed aliases: {alias_text}. "
        "Return JSON with key intents whose value is a list. Every list item must "
        "contain region_alias and question. Do not add prose outside JSON. "
        f"Return between {min_intents} and {max_intents} distinct atomic intents.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"SELECTED_EVIDENCE:\n{context}\n"
    )


def _gap_prompt(
    *,
    topic: str,
    title: str,
    context: str,
    aliases: list[str],
    existing: list[dict[str, Any]],
    max_additional: int,
) -> str:
    alias_text = ", ".join(aliases)
    rendered = "\n".join(
        f"- {row['region_alias']}: {row['question']}" for row in existing
    ) or "(none)"
    return (
        "Perform a second-pass coverage audit over the SAME selected scientific "
        "evidence. Compare the evidence against the existing questions. Generate "
        "new atomic answer-free questions ONLY for scientifically meaningful facts "
        "that are directly stated in the evidence but are not yet targeted by any "
        "existing question. Compound passages are important: an existing question "
        "may cover one fact while another independent fact in the same passage is "
        "still uncovered. Do not use a fixed ontology or checklist. Do not "
        "duplicate or merely paraphrase existing questions. Do not answer the "
        "questions. Do not include digits, measurements, counts, percentages, "
        "numeric results, or copied answer phrases in question text. Each item "
        "must contain a region_alias from the allowed aliases and question. "
        f"Allowed aliases: {alias_text}. "
        "Return JSON with key intents whose value is a list; it is valid for the "
        "list to be empty if no substantive uncovered fact remains. Do not add "
        "prose outside JSON. "
        f"Return at most {max_additional} additional intents.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\nPAPER_TITLE:\n{title}\n\n"
        f"SELECTED_EVIDENCE:\n{context}\n\n"
        f"EXISTING_QUESTIONS:\n{rendered}\n"
    )


def _guard_alias_intents(
    payload: dict[str, Any],
    *,
    source_context: str,
    aliases: dict[str, str],
    min_intents: int,
    max_intents: int,
    min_chars: int,
    max_chars: int,
    max_copy_tokens: int,
    existing_questions: set[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    raw = payload.get("intents")
    if not isinstance(raw, list):
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
        if not isinstance(item, dict):
            rejected.append({"index": idx, "reason": "INVALID_ITEM_SCHEMA"})
            continue
        if "region_alias" not in item or "question" not in item:
            rejected.append({"index": idx, "reason": "INVALID_ITEM_SCHEMA"})
            continue
        alias = str(item["region_alias"]).strip().upper()
        question = " ".join(str(item["question"]).split()).strip()
        reason = None
        if alias not in aliases:
            reason = "UNKNOWN_REGION_ALIAS"
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
                    "region_alias": alias,
                    "question": question,
                    "reason": reason,
                }
            )
            continue
        seen.add(question.casefold())
        accepted.append(
            {
                "region_alias": alias,
                "source_region_id": aliases[alias],
                "question": question,
            }
        )
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


def generate_wave7a2_v2_intents(
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

        # Exact Wave-7A source-aware control representation.
        control_context = _source_context(
            selected,
            max_chars=int(source_arm["max_input_chars"]),
        )

        # Candidate representation uses short aliases only.
        candidate_context, aliases = _alias_source_context(
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
                "control_context": control_context,
                "candidate_context": candidate_context,
                "region_aliases": aliases,
                "control_context_sha256": hashlib.sha256(
                    control_context.encode()
                ).hexdigest(),
                "candidate_context_sha256": hashlib.sha256(
                    candidate_context.encode()
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
            control_context = str(paper["control_context"])
            candidate_context = str(paper["candidate_context"])
            aliases = {
                str(k): str(v)
                for k, v in dict(paper["region_aliases"]).items()
            }

            # Exact prior Wave-7A one-shot control.
            try:
                payload, usage = model_claims(
                    llm,
                    _generator_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context_label="COARSE_SELECTED_SOURCE_CONTEXT",
                        context=control_context,
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
                    source_context=control_context,
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

            pass1_cfg = gap_arm["pass1"]
            pass2_cfg = gap_arm["pass2"]
            pass1_usage = {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "wall_time_seconds": 0.0,
            }
            pass2_usage = dict(pass1_usage)

            try:
                payload1, pass1_usage = model_claims(
                    llm,
                    _pass1_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=candidate_context,
                        aliases=sorted(aliases),
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
                pass1_guard = _guard_alias_intents(
                    payload1,
                    source_context=candidate_context,
                    aliases=aliases,
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

            try:
                existing = list(pass1_guard.get("accepted") or [])
                payload2, pass2_usage = model_claims(
                    llm,
                    _gap_prompt(
                        topic=str(fixture["topic"]),
                        title=str(paper["title"]),
                        context=candidate_context,
                        aliases=sorted(aliases),
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
                pass2_guard = _guard_alias_intents(
                    payload2,
                    source_context=candidate_context,
                    aliases=aliases,
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
        paper.pop("control_context", None)
        paper.pop("candidate_context", None)

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_WAVE7A2_EVIDENCE_CONDITIONED_GENERATION_V2",
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
