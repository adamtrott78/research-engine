from __future__ import annotations

import pathlib
import re
import time
from typing import Any

from research_engine.local_extract_wave1 import (
    ALLOWED_TYPES,
    GPUMemorySampler,
    display_text,
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    initialize_llm,
    model_claims,
    rank_regions,
    retrieval_frontier,
    teacher_agreement,
    token_f1,
)
from research_engine.local_extract_wave2 import (
    WAVE1_FAILED_PAPER_IDS,
    aggregate_region_units,
)


MAX_PROPOSAL_TOKENS = 128
MAX_NORMALIZATION_TOKENS = 320
MIN_PROPOSITION_DISPLAY_CHARS = 24
ABBREVIATIONS = {
    "al",
    "dr",
    "eq",
    "fig",
    "i.e",
    "mr",
    "mrs",
    "no",
    "prof",
    "sec",
    "vs",
}
TERMINAL_RE = re.compile(r"""[.!?]["')\]]?(?=\s|$)""")


def _period_is_abbreviation(raw: str, punctuation_index: int) -> bool:
    prefix = raw[:punctuation_index]
    match = re.search(r"([A-Za-z]+)$", prefix)
    if match is None:
        return False
    token = match.group(1).casefold()
    return token in ABBREVIATIONS or len(token) == 1


def exact_propositions(
    region: dict[str, Any],
    *,
    min_display_chars: int = MIN_PROPOSITION_DISPLAY_CHARS,
) -> list[dict[str, Any]]:
    raw = str(region["raw_source_form"])
    if min_display_chars < 8:
        raise ValueError("min_display_chars must be at least 8")

    boundaries: list[int] = []
    for match in TERMINAL_RE.finditer(raw):
        if raw[match.start()] == "." and _period_is_abbreviation(
            raw, match.start()
        ):
            continue
        boundaries.append(match.end())

    result: list[dict[str, Any]] = []
    cursor = 0
    for end in boundaries:
        chunk = raw[cursor:end]
        left_trim = len(chunk) - len(chunk.lstrip())
        right_len = len(chunk.rstrip())
        local_start = cursor + left_trim
        local_end = cursor + right_len
        cursor = end
        if local_end <= local_start:
            continue
        source = raw[local_start:local_end]
        rendered = display_text(source)
        if len(rendered) < min_display_chars:
            continue

        index = len(result) + 1
        absolute_start = int(region["start_char"]) + local_start
        absolute_end = int(region["start_char"]) + local_end
        result.append(
            {
                "proposition_id": (
                    f"{region['region_id']}:s{index:02d}"
                ),
                "parent_region_id": str(region["region_id"]),
                "page_number": int(region["page_number"]),
                "start_char": absolute_start,
                "end_char": absolute_end,
                "raw_source_form": source,
                "display_text": rendered,
                "section_hint": str(region.get("section_hint") or ""),
                "source_layer": "AUTHORITATIVE_PDF_TEXT",
                "terminal_complete": True,
            }
        )
    return result


def build_proposal_prompt(
    *,
    topic: str,
    paper: dict[str, Any],
    title: str,
    region: dict[str, Any],
    propositions: list[dict[str, Any]],
) -> str:
    candidates = [
        {
            "proposition_id": str(row["proposition_id"]),
            "source_text": str(row["display_text"]),
        }
        for row in propositions
    ]
    return (
        "You are the exact proposition proposal stage of a scientific "
        "evidence pipeline. You receive one retrieved source region already "
        "split deterministically into exact proposition candidates. Choose "
        "at most ONE proposition that contains the most scientifically useful "
        "claim relevant to the research topic. You are NOT normalizing or "
        "paraphrasing any claim. Return JSON only. Return exactly "
        '{"decision":"SKIP"} if none is useful. Otherwise return exactly '
        '{"decision":"SELECT","proposition_id":"<known id>"}. '
        "Never invent an ID and never emit source text, claim type, confidence, "
        "page number, or normalized prose.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_ID: {paper['item_id']}\n"
        f"PAPER_TITLE: {title}\n"
        f"REGION_ID: {region['region_id']}\n\n"
        f"PROPOSITIONS_JSON:\n{candidates}"
    )


def guard_proposal_payload(
    payload: dict[str, Any],
    propositions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "HARD_REJECT",
            "reasons": ["INVALID_TOP_LEVEL_SCHEMA"],
            "selected": None,
        }

    decision = payload.get("decision")
    if decision == "SKIP":
        if set(payload) != {"decision"}:
            return {
                "status": "HARD_REJECT",
                "reasons": ["SKIP_SCHEMA"],
                "selected": None,
            }
        return {
            "status": "SKIP_ACCEPTED",
            "reasons": [],
            "selected": None,
        }

    if decision != "SELECT":
        return {
            "status": "HARD_REJECT",
            "reasons": ["INVALID_DECISION"],
            "selected": None,
        }
    if set(payload) != {"decision", "proposition_id"}:
        return {
            "status": "HARD_REJECT",
            "reasons": ["SELECT_SCHEMA"],
            "selected": None,
        }

    proposition_id = str(payload["proposition_id"])
    by_id = {
        str(row["proposition_id"]): row
        for row in propositions
    }
    selected = by_id.get(proposition_id)
    if selected is None:
        return {
            "status": "HARD_REJECT",
            "reasons": ["UNKNOWN_PROPOSITION_ID"],
            "selected": None,
        }
    return {
        "status": "SELECT_ACCEPTED",
        "reasons": [],
        "selected": selected,
    }


def build_normalization_prompt(
    *,
    topic: str,
    paper: dict[str, Any],
    title: str,
    proposition: dict[str, Any],
) -> str:
    candidate = {
        "source_text": str(proposition["display_text"]),
    }
    return (
        "You are the atomic-claim normalization stage of a scientific "
        "evidence pipeline. A previous guarded stage has already selected "
        "exactly one authoritative source proposition. You may normalize ONLY "
        "that proposition; you may not choose another source. Return JSON only. "
        "If the proposition cannot safely support one scientifically useful "
        "atomic claim relevant to the research topic, return exactly "
        '{"decision":"SKIP"}. Otherwise return exactly one object with '
        '"decision":"CLAIM","type","normalized_form","confidence". '
        "type must be one of method,dataset,result,limitation,research_gap,other. "
        "normalized_form must be one atomic scientifically useful sentence "
        "fully supported by the proposition alone. Do not add numbers, causes, "
        "comparisons, entities, or conclusions absent from the proposition. "
        "Do not emit source identity, page number, offsets, or source text.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_ID: {paper['item_id']}\n"
        f"PAPER_TITLE: {title}\n\n"
        f"SELECTED_PROPOSITION_JSON:\n{candidate}"
    )


def guard_normalization_payload(
    payload: dict[str, Any],
    proposition: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "HARD_REJECT",
            "reasons": ["INVALID_TOP_LEVEL_SCHEMA"],
            "evidence": [],
        }

    decision = payload.get("decision")
    if decision == "SKIP":
        if set(payload) != {"decision"}:
            return {
                "status": "HARD_REJECT",
                "reasons": ["SKIP_SCHEMA"],
                "evidence": [],
            }
        return {
            "status": "SKIP_ACCEPTED",
            "reasons": [],
            "evidence": [],
        }

    if decision != "CLAIM":
        return {
            "status": "HARD_REJECT",
            "reasons": ["INVALID_DECISION"],
            "evidence": [],
        }
    expected = {"decision", "type", "normalized_form", "confidence"}
    if set(payload) != expected:
        return {
            "status": "HARD_REJECT",
            "reasons": ["CLAIM_SCHEMA"],
            "evidence": [],
        }

    kind = str(payload["type"]).strip()
    normalized = display_text(str(payload["normalized_form"]))
    confidence = payload["confidence"]
    reasons: list[str] = []
    if kind not in ALLOWED_TYPES:
        reasons.append("INVALID_TYPE")
    if not 8 <= len(normalized) <= 420:
        reasons.append("INVALID_NORMALIZED_LENGTH")
    if not isinstance(confidence, (int, float)):
        reasons.append("INVALID_CONFIDENCE")
    else:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            reasons.append("INVALID_CONFIDENCE")

    if reasons:
        return {
            "status": "HARD_REJECT",
            "reasons": reasons,
            "evidence": [],
        }

    return {
        "status": "CLAIM_ACCEPTED",
        "reasons": [],
        "evidence": [
            {
                "candidate_id": str(proposition["proposition_id"]),
                "parent_region_id": str(
                    proposition["parent_region_id"]
                ),
                "type": kind,
                "normalized_form": normalized,
                "raw_source_form": str(
                    proposition["raw_source_form"]
                ),
                "page_number": int(proposition["page_number"]),
                "confidence": float(confidence),
                "source_layer": "AUTHORITATIVE_PDF_TEXT",
                "source_start_char": int(proposition["start_char"]),
                "source_end_char": int(proposition["end_char"]),
            }
        ],
    }


def teacher_proposition_alignment(
    teacher: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for teacher_index, expected in enumerate(teacher):
        scored = [
            (
                token_f1(
                    str(expected["normalized_form"]),
                    str(row["display_text"]),
                ),
                row,
            )
            for row in propositions
        ]
        scored.sort(
            key=lambda pair: (
                -pair[0],
                int(pair[1]["page_number"]),
                int(pair[1]["start_char"]),
                str(pair[1]["proposition_id"]),
            )
        )
        if scored:
            score, row = scored[0]
            rows.append(
                {
                    "teacher_index": teacher_index,
                    "teacher_id": expected.get("id"),
                    "best_normalized_f1": score,
                    "page_equal": (
                        int(expected["page_number"])
                        == int(row["page_number"])
                    ),
                    "proposition_id": str(row["proposition_id"]),
                    "parent_region_id": str(
                        row["parent_region_id"]
                    ),
                }
            )
    return {
        "teacher_claim_count": len(teacher),
        "best_proposition_alignment": rows,
        "mean_best_normalized_f1": (
            sum(
                row["best_normalized_f1"]
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
    }


def decide_wave4(report: dict[str, Any]) -> str:
    if not all(bool(value) for value in report["hard_safety"].values()):
        return "REJECT"

    metrics = report["metrics"]
    completion = float(metrics["job_local_completion_fraction"])
    if completion < 0.50:
        return "REJECT"

    control = report.get("teacher_subset_agreement") or {}
    research_positive = (
        float(control.get("qwen_precision_at_f1_0_5", 0.0)) >= 0.95
        and float(
            control.get("teacher_claim_coverage_at_f1_0_5", 0.0)
        )
        >= 0.80
        and float(
            control.get("page_agreement_on_passing_matches", 0.0)
        )
        >= 0.95
        and float(
            control.get("type_agreement_on_passing_matches", 0.0)
        )
        >= 0.95
        and completion >= 0.50
    )
    return "PROMISING" if research_positive else "PARTIAL"


def run_wave4(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_region_chars = int(config["max_region_chars"])
    if top_k != 12 or max_region_chars != 650:
        raise ValueError(
            "Wave 4 requires Wave-2 top_k_regions=12 and max_region_chars=650"
        )

    papers: list[dict[str, Any]] = []
    total_expected_pages = 0
    total_hit_pages = 0
    total_selected_chars = 0
    total_candidate_regions = 0
    total_available_propositions = 0
    acquisition_bytes = 0

    for source_paper in fixture["papers"]:
        paper = dict(source_paper)
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        acquisition_bytes += int(acquisition["bytes"])
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise RuntimeError(
                f"page-count mismatch for {paper['item_id']}: "
                f"expected {paper['total_pdf_pages']} got {len(pages)}"
            )

        regions = exact_source_regions(
            pages,
            max_chars=max_region_chars,
        )
        query = (
            str(fixture["topic"])
            + " "
            + title
            + " "
            + str(paper["item_id"])
        ).strip()
        ranked = rank_regions(query, regions, top_k=32)
        selected = ranked[:top_k]
        for rank, region in enumerate(selected, start=1):
            region["rank"] = rank
            region["propositions"] = exact_propositions(region)
            total_available_propositions += len(
                region["propositions"]
            )

        expected_pages = {
            int(value)
            for value in paper["historical_evidence_pages"]
        }
        hit_pages = expected_pages & {
            int(row["page_number"]) for row in selected
        }
        total_expected_pages += len(expected_pages)
        total_hit_pages += len(hit_pages)
        selected_chars = sum(
            len(str(row["raw_source_form"])) for row in selected
        )
        total_selected_chars += selected_chars
        total_candidate_regions += len(regions)

        papers.append(
            {
                **paper,
                "title": title,
                "acquisition": acquisition,
                "page_count": len(pages),
                "candidate_region_count": len(regions),
                "query": query,
                "retrieval_frontier": retrieval_frontier(
                    paper["historical_evidence_pages"],
                    ranked,
                ),
                "historical_evidence_page_recall_at_12": (
                    len(hit_pages) / len(expected_pages)
                    if expected_pages
                    else 1.0
                ),
                "selected_source_chars": selected_chars,
                "selected_regions": selected,
            }
        )

    init_started = time.perf_counter()
    with GPUMemorySampler(expected_gpu_uuids) as gpu:
        llm = initialize_llm(
            model_path,
            tensor_parallel_size=int(config["tensor_parallel_size"]),
        )
        model_init_seconds = time.perf_counter() - init_started

        prompt_tokens = 0
        output_tokens = 0
        inference_seconds = 0.0
        proposal_model_calls = 0
        normalization_model_calls = 0
        deterministic_no_complete = 0
        proposal_select_count = 0
        proposal_skip_count = 0
        proposal_hard_reject_count = 0
        normalization_skip_count = 0
        normalization_hard_reject_count = 0

        for paper in papers:
            units: list[dict[str, Any]] = []
            for region in paper["selected_regions"]:
                propositions = list(region["propositions"])
                unit: dict[str, Any] = {
                    "candidate_id": str(region["region_id"]),
                    "rank": int(region["rank"]),
                    "page_number": int(region["page_number"]),
                    "available_proposition_count": len(propositions),
                    "proposal_payload": None,
                    "proposal_usage": {
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "wall_time_seconds": 0.0,
                    },
                    "normalization_payload": None,
                    "normalization_usage": {
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "wall_time_seconds": 0.0,
                    },
                }

                if not propositions:
                    deterministic_no_complete += 1
                    unit["proposal_guard"] = {
                        "status": "SKIP_ACCEPTED",
                        "reasons": ["NO_COMPLETE_PROPOSITION"],
                        "selected": None,
                    }
                    unit["guard"] = {
                        "status": "SKIP_ACCEPTED",
                        "reasons": ["NO_COMPLETE_PROPOSITION"],
                        "evidence": [],
                    }
                    units.append(unit)
                    continue

                try:
                    proposal_payload, proposal_usage = model_claims(
                        llm,
                        build_proposal_prompt(
                            topic=str(fixture["topic"]),
                            paper=paper,
                            title=str(paper["title"]),
                            region=region,
                            propositions=propositions,
                        ),
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        seed=int(config["seed"]),
                        max_tokens=MAX_PROPOSAL_TOKENS,
                    )
                    proposal_model_calls += 1
                    prompt_tokens += int(
                        proposal_usage["prompt_tokens"]
                    )
                    output_tokens += int(
                        proposal_usage["output_tokens"]
                    )
                    inference_seconds += float(
                        proposal_usage["wall_time_seconds"]
                    )
                    proposal_guard = guard_proposal_payload(
                        proposal_payload,
                        propositions,
                    )
                    unit["proposal_payload"] = proposal_payload
                    unit["proposal_usage"] = proposal_usage
                    unit["proposal_guard"] = proposal_guard
                except Exception as exc:
                    proposal_guard = {
                        "status": "HARD_REJECT",
                        "reasons": [
                            f"{type(exc).__name__}:{exc}"
                        ],
                        "selected": None,
                    }
                    unit["proposal_guard"] = proposal_guard

                proposal_status = str(proposal_guard["status"])
                if proposal_status == "HARD_REJECT":
                    proposal_hard_reject_count += 1
                    unit["guard"] = {
                        "status": "HARD_REJECT",
                        "reasons": list(
                            proposal_guard.get("reasons") or []
                        ),
                        "evidence": [],
                    }
                    units.append(unit)
                    continue
                if proposal_status == "SKIP_ACCEPTED":
                    proposal_skip_count += 1
                    unit["guard"] = {
                        "status": "SKIP_ACCEPTED",
                        "reasons": [],
                        "evidence": [],
                    }
                    units.append(unit)
                    continue

                proposal_select_count += 1
                selected_proposition = dict(
                    proposal_guard["selected"]
                )
                unit["selected_proposition"] = selected_proposition

                try:
                    norm_payload, norm_usage = model_claims(
                        llm,
                        build_normalization_prompt(
                            topic=str(fixture["topic"]),
                            paper=paper,
                            title=str(paper["title"]),
                            proposition=selected_proposition,
                        ),
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        seed=int(config["seed"]),
                        max_tokens=MAX_NORMALIZATION_TOKENS,
                    )
                    normalization_model_calls += 1
                    prompt_tokens += int(norm_usage["prompt_tokens"])
                    output_tokens += int(norm_usage["output_tokens"])
                    inference_seconds += float(
                        norm_usage["wall_time_seconds"]
                    )
                    norm_guard = guard_normalization_payload(
                        norm_payload,
                        selected_proposition,
                    )
                    unit["normalization_payload"] = norm_payload
                    unit["normalization_usage"] = norm_usage
                    unit["normalization_guard"] = norm_guard
                    unit["guard"] = norm_guard
                except Exception as exc:
                    norm_guard = {
                        "status": "HARD_REJECT",
                        "reasons": [
                            f"{type(exc).__name__}:{exc}"
                        ],
                        "evidence": [],
                    }
                    unit["normalization_guard"] = norm_guard
                    unit["guard"] = norm_guard

                norm_status = str(unit["guard"]["status"])
                if norm_status == "SKIP_ACCEPTED":
                    normalization_skip_count += 1
                elif norm_status == "HARD_REJECT":
                    normalization_hard_reject_count += 1
                units.append(unit)

            paper["region_units"] = units
            paper["aggregation"] = aggregate_region_units(units)

    teacher_paper = next(
        paper for paper in papers if paper.get("teacher_claims")
    )
    teacher = list(teacher_paper["teacher_claims"])
    teacher_control = teacher_agreement(
        teacher,
        list(teacher_paper["aggregation"]["evidence"]),
    )
    available_teacher_propositions = [
        proposition
        for region in teacher_paper["selected_regions"]
        for proposition in region["propositions"]
    ]
    selected_teacher_propositions = [
        dict(unit["selected_proposition"])
        for unit in teacher_paper["region_units"]
        if unit.get("selected_proposition") is not None
    ]
    available_alignment = teacher_proposition_alignment(
        teacher,
        available_teacher_propositions,
    )
    selected_alignment = teacher_proposition_alignment(
        teacher,
        selected_teacher_propositions,
    )

    completed = sum(
        1
        for paper in papers
        if paper["aggregation"]["status"] == "PASS_LOCAL"
    )
    accepted_claims = sum(
        int(paper["aggregation"]["accepted_claim_count"])
        for paper in papers
    )
    unit_claim_count = sum(
        int(paper["aggregation"]["claim_units"])
        for paper in papers
    )
    unit_skip_count = sum(
        int(paper["aggregation"]["skip_units"])
        for paper in papers
    )
    unit_hard_reject_count = sum(
        int(paper["aggregation"]["hard_reject_units"])
        for paper in papers
    )
    exact_duplicates = sum(
        int(
            paper["aggregation"][
                "exact_duplicate_claims_dropped"
            ]
        )
        for paper in papers
    )
    accepted_on_wave1_failed = sum(
        int(paper["aggregation"]["accepted_claim_count"])
        for paper in papers
        if str(paper["item_id"]) in WAVE1_FAILED_PAPER_IDS
    )

    historical = dict(fixture["historical"])
    metrics = {
        "paper_count": len(papers),
        "historical_evidence_count": int(
            historical["evidence_object_count"]
        ),
        "historical_evidence_bearing_page_count": (
            total_expected_pages
        ),
        "historical_evidence_page_hits_at_12": total_hit_pages,
        "historical_evidence_page_recall_at_12": (
            total_hit_pages / total_expected_pages
            if total_expected_pages
            else 1.0
        ),
        "candidate_region_count_total": total_candidate_regions,
        "candidate_region_count_per_paper": (
            total_candidate_regions / len(papers)
            if papers
            else 0.0
        ),
        "available_exact_proposition_count": (
            total_available_propositions
        ),
        "selected_source_chars_total": total_selected_chars,
        "historical_extraction_context_chars": int(
            historical["extraction_context_chars_total"]
        ),
        "selected_source_char_reduction_fraction": (
            1.0
            - total_selected_chars
            / int(historical["extraction_context_chars_total"])
        ),
        "proposal_model_calls": proposal_model_calls,
        "normalization_model_calls": normalization_model_calls,
        "model_calls": (
            proposal_model_calls + normalization_model_calls
        ),
        "qwen_prompt_tokens": prompt_tokens,
        "qwen_output_tokens": output_tokens,
        "historical_remote_extract_input_tokens": int(
            historical["extract_input_tokens"]
        ),
        "projected_remote_input_token_reduction": (
            1.0
            - prompt_tokens
            / int(historical["extract_input_tokens"])
        ),
        "job_local_completion_fraction": (
            completed / len(papers) if papers else 0.0
        ),
        "locally_completed_jobs": completed,
        "accepted_claim_count": accepted_claims,
        "unit_claim_count": unit_claim_count,
        "unit_skip_count": unit_skip_count,
        "unit_hard_reject_count": unit_hard_reject_count,
        "residual_unit_count": unit_hard_reject_count,
        "accepted_claims_on_wave1_failed_papers": (
            accepted_on_wave1_failed
        ),
        "exact_duplicate_claims_dropped": exact_duplicates,
        "deterministic_no_complete_proposition_count": (
            deterministic_no_complete
        ),
        "proposal_select_count": proposal_select_count,
        "proposal_skip_count": proposal_skip_count,
        "proposal_hard_reject_count": (
            proposal_hard_reject_count
        ),
        "normalization_skip_count": normalization_skip_count,
        "normalization_hard_reject_count": (
            normalization_hard_reject_count
        ),
        "model_init_seconds": model_init_seconds,
        "inference_seconds_total": inference_seconds,
        "downloaded_source_bytes": acquisition_bytes,
        "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
    }

    proposition_by_id = {
        str(proposition["proposition_id"]): proposition
        for paper in papers
        for region in paper["selected_regions"]
        for proposition in region["propositions"]
    }
    hard_safety = {
        "fabricated_or_unlocatable_quote_count_zero": all(
            evidence["candidate_id"] in proposition_by_id
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "exact_quote_locatability_one": all(
            evidence["raw_source_form"]
            == proposition_by_id[
                evidence["candidate_id"]
            ]["raw_source_form"]
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "page_provenance_coverage_one": all(
            isinstance(evidence["page_number"], int)
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "structural_claim_budget_respected": all(
            paper["aggregation"]["accepted_claim_count"]
            <= top_k
            and all(
                len(unit["guard"].get("evidence") or []) <= 1
                for unit in paper["region_units"]
            )
            for paper in papers
        ),
        "scientific_state_materialized_false": True,
        "production_routing_changed_false": True,
    }

    report = {
        "schema_version": 1,
        "operation": (
            "LOCAL_EXTRACT_WAVE4_EXACT_PROPOSAL_THEN_NORMALIZATION"
        ),
        "source_run_id": fixture["source_run_id"],
        "proposal_model_role": (
            "ONE_REGION_EXACT_PROPOSITION_SELECTOR"
        ),
        "normalization_model_role": (
            "ONE_EXACT_PROPOSITION_TO_ZERO_OR_ONE_ATOMIC_CLAIM"
        ),
        "model_path": model_path,
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "retrieval_strategy": "FACET_INTERLEAVED_BM25_V1",
        "region_construction": {
            "id": "WAVE2_EXACT650_V1",
            "max_raw_chars": max_region_chars,
        },
        "top_k_regions": top_k,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "max_normalization_tokens": MAX_NORMALIZATION_TOKENS,
        "qwen_config_inherited": config,
        "historical": historical,
        "metrics": metrics,
        "teacher_subset_agreement": teacher_control,
        "teacher_available_proposition_alignment": (
            available_alignment
        ),
        "teacher_selected_proposition_alignment": (
            selected_alignment
        ),
        "teacher_control_scope": {
            "full_historical_teacher_claim_count": int(
                historical["evidence_object_count"]
            ),
            "normalized_teacher_claims_available": int(
                teacher_control["teacher_claim_count"]
            ),
            "normalized_teacher_control_fraction": (
                teacher_control["teacher_claim_count"]
                / int(historical["evidence_object_count"])
            ),
            "limitation": (
                "The preserved normalized teacher subset contains only six "
                "claims from one paper and includes multi-fact historical "
                "claims. The legacy lexical scorer remains unchanged for "
                "continuity; proposal-stage source alignment is diagnostic."
            ),
        },
        "hard_safety": hard_safety,
        "papers": papers,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "public_model_acquisition": False,
        "source_acquisition": "ARXIV_EXACT_SHA_EPHEMERAL",
        "wave2_baseline": {
            "decision": "PARTIAL",
            "job_local_completion_fraction": 1.0,
            "accepted_claim_count": 70,
            "teacher_matched_normalized_f1": (
                0.365763503150073
            ),
            "teacher_claim_coverage_at_f1_0_5": 0.0,
            "mean_best_teacher_source_region_f1": (
                0.27380476969124573
            ),
            "artifact_id": 12464958088,
            "report_sha256": (
                "e830160adc794322f30924c89ffc9fc632fc5ccba61f739bb8bcf015e854762f"
            ),
        },
        "wave3_disposition": {
            "decision": "PARTIAL",
            "hypothesis_decision": "NEGATIVE_DO_NOT_RETAIN",
            "artifact_id": 12468740757,
            "report_sha256": (
                "8596b613b1644234808523ed90d7ac3bae90c5152ba49a633c796d60b437b45f"
            ),
        },
    }
    report["wave4_decision"] = decide_wave4(report)
    return report
