from __future__ import annotations

import pathlib
import re
import time
from collections import Counter
from typing import Any

from research_engine.local_extract_wave1 import (
    ALLOWED_TYPES,
    GPUMemorySampler,
    display_text,
    download_exact_pdf,
    extract_pdf_pages,
    initialize_llm,
    looks_like_heading,
    model_claims,
    rank_regions,
    retrieval_frontier,
    teacher_agreement,
    token_f1,
)


WAVE1_FAILED_PAPER_IDS = {
    "arxiv:2411.03636",
    "arxiv:2209.03142",
    "arxiv:2403.04036",
    "arxiv:2609.02007",
    "arxiv:2609.20765",
    "arxiv:2108.13099",
}
MAX_OUTPUT_TOKENS_PER_REGION = 320
SOFT_REGION_CHARS = 650
HARD_REGION_CHARS = 1200


def exact_source_regions_soft_terminal(
    pages: list[dict[str, Any]],
    *,
    soft_chars: int = SOFT_REGION_CHARS,
    hard_chars: int = HARD_REGION_CHARS,
) -> list[dict[str, Any]]:
    if soft_chars < 64:
        raise ValueError("soft_chars must be at least 64")
    if hard_chars < soft_chars:
        raise ValueError("hard_chars must be >= soft_chars")

    regions: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text") or "")
        section_hint = ""
        start: int | None = None
        end = 0
        region_section = ""
        extended_past_soft = False

        for match in re.finditer(r".*(?:\\n|$)", text):
            raw_line = match.group(0)
            if not raw_line:
                continue
            line = display_text(raw_line)
            if looks_like_heading(line):
                section_hint = line

            if start is None:
                start = match.start()
                region_section = section_hint
                extended_past_soft = False
            end = match.end()
            current = text[start:end]
            terminal = bool(
                re.search(r'[.!?][\"\\')\\]]?
def build_region_prompt(
    *,
    topic: str,
    paper: dict[str, Any],
    title: str,
    region: dict[str, Any],
) -> str:
    candidate = {
        "page_number": int(region["page_number"]),
        "section_hint": str(region.get("section_hint") or ""),
        "source_text": str(region["display_text"]),
    }
    return (
        "You are the atomic-claim normalization stage of a scientific "
        "evidence pipeline. You receive exactly one already-located exact "
        "source region. You are NOT discovering source quotes and you must "
        "not emit candidate_id, raw_source_form, page_number, or offsets. "
        "Return JSON only. If this region does not contain one scientifically "
        "useful claim relevant to the research topic, return exactly "
        '{"decision":"SKIP"}. If it does, return exactly one object with '
        '"decision":"CLAIM","type","normalized_form","confidence". '
        "type must be one of method,dataset,result,limitation,research_gap,other. "
        "normalized_form must be one atomic scientifically useful sentence "
        "fully supported by this region alone. Do not add numbers, causes, "
        "comparisons, entities, or conclusions absent from the region. "
        "Prefer SKIP to speculation. Emit at most one claim. Deterministic "
        "code owns the source identity and quote.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_ID: {paper['item_id']}\n"
        f"PAPER_TITLE: {title}\n\n"
        f"CANDIDATE_JSON:\n{candidate}"
    )


def guard_region_payload(
    payload: dict[str, Any],
    region: dict[str, Any],
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

    expected = {
        "decision",
        "type",
        "normalized_form",
        "confidence",
    }
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
                "candidate_id": str(region["region_id"]),
                "type": kind,
                "normalized_form": normalized,
                "raw_source_form": str(region["raw_source_form"]),
                "page_number": int(region["page_number"]),
                "confidence": float(confidence),
                "source_layer": "AUTHORITATIVE_PDF_TEXT",
                "source_start_char": int(region["start_char"]),
                "source_end_char": int(region["end_char"]),
            }
        ],
    }


def deduplicate_evidence(
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for row in evidence:
        key = (
            str(row["type"]),
            display_text(str(row["normalized_form"])).casefold(),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def aggregate_region_units(
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    hard_rejects: list[dict[str, Any]] = []
    skips = 0

    for unit in units:
        guard = dict(unit["guard"])
        status = str(guard["status"])
        if status == "CLAIM_ACCEPTED":
            accepted.extend(list(guard["evidence"]))
        elif status == "SKIP_ACCEPTED":
            skips += 1
        else:
            hard_rejects.append(
                {
                    "candidate_id": str(unit["candidate_id"]),
                    "rank": int(unit["rank"]),
                    "reasons": list(guard.get("reasons") or []),
                }
            )

    deduped, duplicate_count = deduplicate_evidence(accepted)
    residual_required = bool(hard_rejects) or not deduped
    return {
        "status": (
            "PASS_LOCAL"
            if not residual_required
            else "RESIDUAL_REQUIRED"
        ),
        "evidence": deduped,
        "hard_rejects": hard_rejects,
        "claim_units": len(accepted),
        "skip_units": skips,
        "hard_reject_units": len(hard_rejects),
        "exact_duplicate_claims_dropped": duplicate_count,
        "residual_required": residual_required,
        "accepted_claim_count": len(deduped),
    }


def decide_wave3(report: dict[str, Any]) -> str:
    if not all(bool(value) for value in report["hard_safety"].values()):
        return "REJECT"

    metrics = report["metrics"]
    completion = float(metrics["job_local_completion_fraction"])
    if completion < 0.50:
        return "REJECT"

    control = report.get("teacher_subset_agreement") or {}
    research_positive = (
        float(control.get("qwen_precision_at_f1_0_5", 0.0)) >= 0.95
        and float(control.get("teacher_claim_coverage_at_f1_0_5", 0.0))
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


def run_wave3(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_region_chars = int(config["max_region_chars"])

    if top_k != 12:
        raise ValueError("Wave 3 experimental control requires top_k_regions=12")

    papers: list[dict[str, Any]] = []
    total_expected_pages = 0
    total_hit_pages = 0
    total_selected_chars = 0
    total_candidate_regions = 0
    total_candidate_soft_extended = 0
    total_candidate_forced_hard = 0
    total_selected_soft_extended = 0
    total_selected_forced_hard = 0
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

        regions = exact_source_regions_soft_terminal(
            pages,
            soft_chars=max_region_chars,
            hard_chars=HARD_REGION_CHARS,
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

        expected_pages = {
            int(value) for value in paper["historical_evidence_pages"]
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
        total_candidate_soft_extended += sum(
            int(bool(row.get("soft_extended"))) for row in regions
        )
        total_candidate_forced_hard += sum(
            int(bool(row.get("forced_hard_split"))) for row in regions
        )
        total_selected_soft_extended += sum(
            int(bool(row.get("soft_extended"))) for row in selected
        )
        total_selected_forced_hard += sum(
            int(bool(row.get("forced_hard_split"))) for row in selected
        )

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
                "selected_soft_extended_region_count": sum(
                    int(bool(row.get("soft_extended"))) for row in selected
                ),
                "selected_forced_hard_split_count": sum(
                    int(bool(row.get("forced_hard_split"))) for row in selected
                ),
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

        for paper in papers:
            units: list[dict[str, Any]] = []
            for region in paper["selected_regions"]:
                prompt = build_region_prompt(
                    topic=str(fixture["topic"]),
                    paper=paper,
                    title=str(paper["title"]),
                    region=region,
                )
                try:
                    payload, usage = model_claims(
                        llm,
                        prompt,
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        seed=int(config["seed"]),
                        max_tokens=MAX_OUTPUT_TOKENS_PER_REGION,
                    )
                    guard = guard_region_payload(payload, region)
                    unit = {
                        "candidate_id": str(region["region_id"]),
                        "rank": int(region["rank"]),
                        "page_number": int(region["page_number"]),
                        "model_payload": payload,
                        "qwen_usage": usage,
                        "guard": guard,
                    }
                    prompt_tokens += int(usage["prompt_tokens"])
                    output_tokens += int(usage["output_tokens"])
                    inference_seconds += float(usage["wall_time_seconds"])
                except Exception as exc:
                    unit = {
                        "candidate_id": str(region["region_id"]),
                        "rank": int(region["rank"]),
                        "page_number": int(region["page_number"]),
                        "model_payload": None,
                        "qwen_usage": {
                            "prompt_tokens": 0,
                            "output_tokens": 0,
                            "wall_time_seconds": 0.0,
                        },
                        "guard": {
                            "status": "HARD_REJECT",
                            "reasons": [f"{type(exc).__name__}:{exc}"],
                            "evidence": [],
                        },
                    }
                units.append(unit)

            paper["region_units"] = units
            paper["aggregation"] = aggregate_region_units(units)

    teacher_paper = next(
        paper for paper in papers if paper.get("teacher_claims")
    )
    teacher_control = teacher_agreement(
        list(teacher_paper["teacher_claims"]),
        list(teacher_paper["aggregation"]["evidence"]),
    )

    teacher_source_control = teacher_source_region_alignment(
        list(teacher_paper["teacher_claims"]),
        list(teacher_paper["selected_regions"]),
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
        int(paper["aggregation"]["exact_duplicate_claims_dropped"])
        for paper in papers
    )
    preserved_despite_residuals = sum(
        1
        for paper in papers
        if paper["aggregation"]["status"] == "RESIDUAL_REQUIRED"
        and paper["aggregation"]["accepted_claim_count"] > 0
    )
    accepted_on_wave1_failed = sum(
        int(paper["aggregation"]["accepted_claim_count"])
        for paper in papers
        if str(paper["item_id"]) in WAVE1_FAILED_PAPER_IDS
    )
    paper_wide_collateral_loss_count = sum(
        1
        for paper in papers
        if paper["aggregation"]["hard_reject_units"] > 0
        and paper["aggregation"]["claim_units"] > 0
        and paper["aggregation"]["accepted_claim_count"] == 0
    )

    historical = dict(fixture["historical"])
    metrics = {
        "paper_count": len(papers),
        "historical_evidence_count": int(
            historical["evidence_object_count"]
        ),
        "historical_evidence_bearing_page_count": total_expected_pages,
        "historical_evidence_page_hits_at_12": total_hit_pages,
        "historical_evidence_page_recall_at_12": (
            total_hit_pages / total_expected_pages
            if total_expected_pages
            else 1.0
        ),
        "candidate_region_count_total": total_candidate_regions,
        "candidate_region_count_per_paper": (
            total_candidate_regions / len(papers) if papers else 0.0
        ),
        "candidate_soft_extended_region_count": total_candidate_soft_extended,
        "candidate_forced_hard_split_count": total_candidate_forced_hard,
        "selected_soft_extended_region_count": total_selected_soft_extended,
        "selected_forced_hard_split_count": total_selected_forced_hard,
        "selected_source_chars_total": total_selected_chars,
        "historical_extraction_context_chars": int(
            historical["extraction_context_chars_total"]
        ),
        "selected_source_char_reduction_fraction": (
            1.0
            - total_selected_chars
            / int(historical["extraction_context_chars_total"])
        ),
        "model_calls": sum(len(paper["region_units"]) for paper in papers),
        "qwen_prompt_tokens": prompt_tokens,
        "qwen_output_tokens": output_tokens,
        "historical_remote_extract_input_tokens": int(
            historical["extract_input_tokens"]
        ),
        "projected_remote_input_token_reduction": (
            1.0
            - prompt_tokens / int(historical["extract_input_tokens"])
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
        "papers_with_preserved_claims_despite_residuals": (
            preserved_despite_residuals
        ),
        "accepted_claims_on_wave1_failed_papers": accepted_on_wave1_failed,
        "exact_duplicate_claims_dropped": exact_duplicates,
        "paper_wide_collateral_loss_count": (
            paper_wide_collateral_loss_count
        ),
        "model_init_seconds": model_init_seconds,
        "inference_seconds_total": inference_seconds,
        "downloaded_source_bytes": acquisition_bytes,
        "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
    }

    hard_safety = {
        "fabricated_or_unlocatable_quote_count_zero": all(
            evidence["candidate_id"]
            in {
                str(region["region_id"])
                for region in paper["selected_regions"]
            }
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "exact_quote_locatability_one": all(
            evidence["raw_source_form"]
            == next(
                str(region["raw_source_form"])
                for region in paper["selected_regions"]
                if str(region["region_id"]) == evidence["candidate_id"]
            )
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "page_provenance_coverage_one": all(
            isinstance(evidence["page_number"], int)
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "structural_claim_budget_respected": all(
            paper["aggregation"]["accepted_claim_count"] <= top_k
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
        "operation": "LOCAL_EXTRACT_WAVE3_SOFT_TERMINAL_REGIONS",
        "source_run_id": fixture["source_run_id"],
        "model_role": "ONE_REGION_TO_ZERO_OR_ONE_ATOMIC_CLAIM",
        "model_path": model_path,
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "retrieval_strategy": "FACET_INTERLEAVED_BM25_V1",
        "top_k_regions": top_k,
        "max_output_tokens_per_region": MAX_OUTPUT_TOKENS_PER_REGION,
        "region_construction": {
            "id": "SOFT650_TERMINAL_OR_HARD1200_V1",
            "soft_target_raw_chars": max_region_chars,
            "hard_max_raw_chars": HARD_REGION_CHARS,
        },
        "qwen_config_inherited": config,
        "historical": historical,
        "metrics": metrics,
        "teacher_subset_agreement": teacher_control,
        "teacher_source_region_alignment": teacher_source_control,
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
                "Full normalized Codex teacher text is not staged into the "
                "isolated benchmark runner. Page/count controls cover all nine "
                "papers; semantic F1/type/page control uses the preserved "
                "teacher subset only."
            ),
        },
        "hard_safety": hard_safety,
        "papers": papers,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "public_model_acquisition": False,
        "source_acquisition": "ARXIV_EXACT_SHA_EPHEMERAL",
        "wave1_baseline": {
            "decision": "REJECT",
            "job_local_completion_fraction": 0.3333333333333333,
            "accepted_claim_count": 30,
            "historical_evidence_page_recall_at_12": 0.7441860465116279,
            "projected_remote_input_token_reduction": 0.9175094731750141,
            "artifact_id": 12231293963,
            "report_sha256": (
                "5954e2de2fd1d23c770f7f687c5282b8e2f43e68c9b29ece6374fed5442f98dc"
            ),
        },
        "wave2_baseline": {
            "decision": "PARTIAL",
            "job_local_completion_fraction": 1.0,
            "accepted_claim_count": 70,
            "teacher_claim_coverage_at_f1_0_5": 0.0,
            "projected_remote_input_token_reduction": 0.784898460105168,
            "artifact_id": 12464958088,
            "report_sha256": (
                "e830160adc794322f30924c89ffc9fc632fc5ccba61f739bb8bcf015e854762f"
            ),
        },
    }
    report["wave3_decision"] = decide_wave3(report)
    return report
, current.rstrip())
            )

            forced_hard_split = False
            if not terminal:
                if len(current) < soft_chars:
                    continue
                if len(current) < hard_chars:
                    extended_past_soft = True
                    continue
                forced_hard_split = True

            raw = text[start:end]
            rendered = display_text(raw)
            if len(rendered) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": rendered,
                        "section_hint": region_section,
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "soft_extended": extended_past_soft,
                        "forced_hard_split": forced_hard_split,
                    }
                )
            start = None
            region_section = ""
            extended_past_soft = False

        if start is not None and end > start:
            raw = text[start:end]
            rendered = display_text(raw)
            if len(rendered) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": rendered,
                        "section_hint": region_section,
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "soft_extended": extended_past_soft,
                        "forced_hard_split": False,
                    }
                )
    return regions


def teacher_source_region_alignment(
    teacher: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for teacher_index, expected in enumerate(teacher):
        scored = [
            (
                token_f1(
                    str(expected["normalized_form"]),
                    str(region["display_text"]),
                ),
                region,
            )
            for region in regions
        ]
        scored.sort(
            key=lambda pair: (
                -pair[0],
                int(pair[1]["page_number"]),
                int(pair[1]["start_char"]),
            )
        )
        if scored:
            score, region = scored[0]
            rows.append(
                {
                    "teacher_index": teacher_index,
                    "teacher_id": expected.get("id"),
                    "best_normalized_f1": score,
                    "page_equal": (
                        int(expected["page_number"])
                        == int(region["page_number"])
                    ),
                    "region_id": str(region["region_id"]),
                    "region_rank": int(region["rank"]),
                    "soft_extended": bool(
                        region.get("soft_extended", False)
                    ),
                    "forced_hard_split": bool(
                        region.get("forced_hard_split", False)
                    ),
                }
            )
    return {
        "teacher_claim_count": len(teacher),
        "best_region_alignment": rows,
        "mean_best_normalized_f1": (
            sum(row["best_normalized_f1"] for row in rows) / len(rows)
            if rows else 0.0
        ),
    }



def build_region_prompt(
    *,
    topic: str,
    paper: dict[str, Any],
    title: str,
    region: dict[str, Any],
) -> str:
    candidate = {
        "page_number": int(region["page_number"]),
        "section_hint": str(region.get("section_hint") or ""),
        "source_text": str(region["display_text"]),
    }
    return (
        "You are the atomic-claim normalization stage of a scientific "
        "evidence pipeline. You receive exactly one already-located exact "
        "source region. You are NOT discovering source quotes and you must "
        "not emit candidate_id, raw_source_form, page_number, or offsets. "
        "Return JSON only. If this region does not contain one scientifically "
        "useful claim relevant to the research topic, return exactly "
        '{"decision":"SKIP"}. If it does, return exactly one object with '
        '"decision":"CLAIM","type","normalized_form","confidence". '
        "type must be one of method,dataset,result,limitation,research_gap,other. "
        "normalized_form must be one atomic scientifically useful sentence "
        "fully supported by this region alone. Do not add numbers, causes, "
        "comparisons, entities, or conclusions absent from the region. "
        "Prefer SKIP to speculation. Emit at most one claim. Deterministic "
        "code owns the source identity and quote.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_ID: {paper['item_id']}\n"
        f"PAPER_TITLE: {title}\n\n"
        f"CANDIDATE_JSON:\n{candidate}"
    )


def guard_region_payload(
    payload: dict[str, Any],
    region: dict[str, Any],
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

    expected = {
        "decision",
        "type",
        "normalized_form",
        "confidence",
    }
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
                "candidate_id": str(region["region_id"]),
                "type": kind,
                "normalized_form": normalized,
                "raw_source_form": str(region["raw_source_form"]),
                "page_number": int(region["page_number"]),
                "confidence": float(confidence),
                "source_layer": "AUTHORITATIVE_PDF_TEXT",
                "source_start_char": int(region["start_char"]),
                "source_end_char": int(region["end_char"]),
            }
        ],
    }


def deduplicate_evidence(
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for row in evidence:
        key = (
            str(row["type"]),
            display_text(str(row["normalized_form"])).casefold(),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def aggregate_region_units(
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    hard_rejects: list[dict[str, Any]] = []
    skips = 0

    for unit in units:
        guard = dict(unit["guard"])
        status = str(guard["status"])
        if status == "CLAIM_ACCEPTED":
            accepted.extend(list(guard["evidence"]))
        elif status == "SKIP_ACCEPTED":
            skips += 1
        else:
            hard_rejects.append(
                {
                    "candidate_id": str(unit["candidate_id"]),
                    "rank": int(unit["rank"]),
                    "reasons": list(guard.get("reasons") or []),
                }
            )

    deduped, duplicate_count = deduplicate_evidence(accepted)
    residual_required = bool(hard_rejects) or not deduped
    return {
        "status": (
            "PASS_LOCAL"
            if not residual_required
            else "RESIDUAL_REQUIRED"
        ),
        "evidence": deduped,
        "hard_rejects": hard_rejects,
        "claim_units": len(accepted),
        "skip_units": skips,
        "hard_reject_units": len(hard_rejects),
        "exact_duplicate_claims_dropped": duplicate_count,
        "residual_required": residual_required,
        "accepted_claim_count": len(deduped),
    }


def decide_wave3(report: dict[str, Any]) -> str:
    if not all(bool(value) for value in report["hard_safety"].values()):
        return "REJECT"

    metrics = report["metrics"]
    completion = float(metrics["job_local_completion_fraction"])
    if completion < 0.50:
        return "REJECT"

    control = report.get("teacher_subset_agreement") or {}
    research_positive = (
        float(control.get("qwen_precision_at_f1_0_5", 0.0)) >= 0.95
        and float(control.get("teacher_claim_coverage_at_f1_0_5", 0.0))
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


def run_wave3(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_region_chars = int(config["max_region_chars"])

    if top_k != 12:
        raise ValueError("Wave 3 experimental control requires top_k_regions=12")

    papers: list[dict[str, Any]] = []
    total_expected_pages = 0
    total_hit_pages = 0
    total_selected_chars = 0
    total_candidate_regions = 0
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

        regions = exact_source_regions(pages, max_chars=max_region_chars)
        query = (
            str(fixture["topic"])
            + " "
            + title
            + " "
            + str(paper["item_id"])
        ).strip()
        ranked = rank_regions(query, regions, top_k=32)
        selected = ranked[:top_k]

        expected_pages = {
            int(value) for value in paper["historical_evidence_pages"]
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

        for paper in papers:
            units: list[dict[str, Any]] = []
            for region in paper["selected_regions"]:
                prompt = build_region_prompt(
                    topic=str(fixture["topic"]),
                    paper=paper,
                    title=str(paper["title"]),
                    region=region,
                )
                try:
                    payload, usage = model_claims(
                        llm,
                        prompt,
                        temperature=float(config["temperature"]),
                        top_p=float(config["top_p"]),
                        seed=int(config["seed"]),
                        max_tokens=MAX_OUTPUT_TOKENS_PER_REGION,
                    )
                    guard = guard_region_payload(payload, region)
                    unit = {
                        "candidate_id": str(region["region_id"]),
                        "rank": int(region["rank"]),
                        "page_number": int(region["page_number"]),
                        "model_payload": payload,
                        "qwen_usage": usage,
                        "guard": guard,
                    }
                    prompt_tokens += int(usage["prompt_tokens"])
                    output_tokens += int(usage["output_tokens"])
                    inference_seconds += float(usage["wall_time_seconds"])
                except Exception as exc:
                    unit = {
                        "candidate_id": str(region["region_id"]),
                        "rank": int(region["rank"]),
                        "page_number": int(region["page_number"]),
                        "model_payload": None,
                        "qwen_usage": {
                            "prompt_tokens": 0,
                            "output_tokens": 0,
                            "wall_time_seconds": 0.0,
                        },
                        "guard": {
                            "status": "HARD_REJECT",
                            "reasons": [f"{type(exc).__name__}:{exc}"],
                            "evidence": [],
                        },
                    }
                units.append(unit)

            paper["region_units"] = units
            paper["aggregation"] = aggregate_region_units(units)

    teacher_paper = next(
        paper for paper in papers if paper.get("teacher_claims")
    )
    teacher_control = teacher_agreement(
        list(teacher_paper["teacher_claims"]),
        list(teacher_paper["aggregation"]["evidence"]),
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
        int(paper["aggregation"]["exact_duplicate_claims_dropped"])
        for paper in papers
    )
    preserved_despite_residuals = sum(
        1
        for paper in papers
        if paper["aggregation"]["status"] == "RESIDUAL_REQUIRED"
        and paper["aggregation"]["accepted_claim_count"] > 0
    )
    accepted_on_wave1_failed = sum(
        int(paper["aggregation"]["accepted_claim_count"])
        for paper in papers
        if str(paper["item_id"]) in WAVE1_FAILED_PAPER_IDS
    )
    paper_wide_collateral_loss_count = sum(
        1
        for paper in papers
        if paper["aggregation"]["hard_reject_units"] > 0
        and paper["aggregation"]["claim_units"] > 0
        and paper["aggregation"]["accepted_claim_count"] == 0
    )

    historical = dict(fixture["historical"])
    metrics = {
        "paper_count": len(papers),
        "historical_evidence_count": int(
            historical["evidence_object_count"]
        ),
        "historical_evidence_bearing_page_count": total_expected_pages,
        "historical_evidence_page_hits_at_12": total_hit_pages,
        "historical_evidence_page_recall_at_12": (
            total_hit_pages / total_expected_pages
            if total_expected_pages
            else 1.0
        ),
        "candidate_region_count_total": total_candidate_regions,
        "candidate_region_count_per_paper": (
            total_candidate_regions / len(papers) if papers else 0.0
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
        "model_calls": sum(len(paper["region_units"]) for paper in papers),
        "qwen_prompt_tokens": prompt_tokens,
        "qwen_output_tokens": output_tokens,
        "historical_remote_extract_input_tokens": int(
            historical["extract_input_tokens"]
        ),
        "projected_remote_input_token_reduction": (
            1.0
            - prompt_tokens / int(historical["extract_input_tokens"])
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
        "papers_with_preserved_claims_despite_residuals": (
            preserved_despite_residuals
        ),
        "accepted_claims_on_wave1_failed_papers": accepted_on_wave1_failed,
        "exact_duplicate_claims_dropped": exact_duplicates,
        "paper_wide_collateral_loss_count": (
            paper_wide_collateral_loss_count
        ),
        "model_init_seconds": model_init_seconds,
        "inference_seconds_total": inference_seconds,
        "downloaded_source_bytes": acquisition_bytes,
        "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
    }

    hard_safety = {
        "fabricated_or_unlocatable_quote_count_zero": all(
            evidence["candidate_id"]
            in {
                str(region["region_id"])
                for region in paper["selected_regions"]
            }
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "exact_quote_locatability_one": all(
            evidence["raw_source_form"]
            == next(
                str(region["raw_source_form"])
                for region in paper["selected_regions"]
                if str(region["region_id"]) == evidence["candidate_id"]
            )
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "page_provenance_coverage_one": all(
            isinstance(evidence["page_number"], int)
            for paper in papers
            for evidence in paper["aggregation"]["evidence"]
        ),
        "structural_claim_budget_respected": all(
            paper["aggregation"]["accepted_claim_count"] <= top_k
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
        "operation": "LOCAL_EXTRACT_WAVE2_REGION_ISOLATED_NORMALIZATION",
        "source_run_id": fixture["source_run_id"],
        "model_role": "ONE_REGION_TO_ZERO_OR_ONE_ATOMIC_CLAIM",
        "model_path": model_path,
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "retrieval_strategy": "FACET_INTERLEAVED_BM25_V1",
        "top_k_regions": top_k,
        "max_output_tokens_per_region": MAX_OUTPUT_TOKENS_PER_REGION,
        "qwen_config_inherited": config,
        "historical": historical,
        "metrics": metrics,
        "teacher_subset_agreement": teacher_control,
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
                "Full normalized Codex teacher text is not staged into the "
                "isolated benchmark runner. Page/count controls cover all nine "
                "papers; semantic F1/type/page control uses the preserved "
                "teacher subset only."
            ),
        },
        "hard_safety": hard_safety,
        "papers": papers,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "public_model_acquisition": False,
        "source_acquisition": "ARXIV_EXACT_SHA_EPHEMERAL",
        "wave1_baseline": {
            "decision": "REJECT",
            "job_local_completion_fraction": 0.3333333333333333,
            "accepted_claim_count": 30,
            "historical_evidence_page_recall_at_12": 0.7441860465116279,
            "projected_remote_input_token_reduction": 0.9175094731750141,
            "artifact_id": 12231293963,
            "report_sha256": (
                "5954e2de2fd1d23c770f7f687c5282b8e2f43e68c9b29ece6374fed5442f98dc"
            ),
        },
    }
    report["wave2_decision"] = decide_wave3(report)
    return report
