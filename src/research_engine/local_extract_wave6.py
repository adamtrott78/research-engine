from __future__ import annotations

import pathlib
import time
from typing import Any

from research_engine.local_extract_wave1 import (
    GPUMemorySampler,
    _bm25,
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    initialize_llm,
    model_claims,
    rank_regions,
    retrieval_frontier,
    teacher_agreement,
)
from research_engine.local_extract_wave2 import (
    WAVE1_FAILED_PAPER_IDS,
    aggregate_region_units,
)
from research_engine.local_extract_wave4 import (
    MAX_NORMALIZATION_TOKENS,
    build_normalization_prompt,
    exact_propositions,
    guard_normalization_payload,
    teacher_proposition_alignment,
)


SELECTOR_ID = "PAPER_GLOBAL_BM25_THEN_PER_REGION_TOP_V1"


def deterministic_proposition_selections(
    query: str,
    regions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    propositions = [
        proposition
        for region in regions
        for proposition in region["propositions"]
    ]
    if not propositions:
        return {}

    aliases: list[dict[str, Any]] = []
    for proposition in propositions:
        row = dict(proposition)
        row["region_id"] = str(proposition["proposition_id"])
        aliases.append(row)

    ranked = _bm25(query, aliases)
    selected: dict[str, dict[str, Any]] = {}
    for ranked_row in ranked:
        parent_region_id = str(ranked_row["parent_region_id"])
        if parent_region_id in selected:
            continue
        proposition = dict(ranked_row)
        proposition.pop("region_id", None)
        proposition["selector_score"] = float(
            proposition.pop("retrieval_score")
        )
        proposition["selector_id"] = SELECTOR_ID
        selected[parent_region_id] = proposition
    return selected


def decide_wave6(report: dict[str, Any]) -> str:
    if not all(bool(value) for value in report["hard_safety"].values()):
        return "REJECT"

    completion = float(
        report["metrics"]["job_local_completion_fraction"]
    )
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


def run_wave6(
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
            "Wave 6 requires Wave-2 top_k_regions=12 and max_region_chars=650"
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

        selected_by_region = deterministic_proposition_selections(
            query,
            selected,
        )
        for region in selected:
            region["selected_proposition"] = selected_by_region.get(
                str(region["region_id"])
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
        normalization_model_calls = 0
        deterministic_no_complete = 0
        deterministic_select_count = 0
        normalization_skip_count = 0
        normalization_hard_reject_count = 0

        for paper in papers:
            units: list[dict[str, Any]] = []
            for region in paper["selected_regions"]:
                propositions = list(region["propositions"])
                selected_proposition = region.get(
                    "selected_proposition"
                )
                unit: dict[str, Any] = {
                    "candidate_id": str(region["region_id"]),
                    "rank": int(region["rank"]),
                    "page_number": int(region["page_number"]),
                    "available_proposition_count": len(propositions),
                    "selector_id": SELECTOR_ID,
                    "selected_proposition": selected_proposition,
                    "normalization_payload": None,
                    "normalization_usage": {
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "wall_time_seconds": 0.0,
                    },
                }

                if selected_proposition is None:
                    deterministic_no_complete += 1
                    unit["guard"] = {
                        "status": "SKIP_ACCEPTED",
                        "reasons": ["NO_COMPLETE_PROPOSITION"],
                        "evidence": [],
                    }
                    units.append(unit)
                    continue

                deterministic_select_count += 1
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
        "historical_evidence_bearing_page_count": total_expected_pages,
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
        "proposal_model_calls": 0,
        "deterministic_selected_proposition_count": (
            deterministic_select_count
        ),
        "normalization_model_calls": normalization_model_calls,
        "model_calls": normalization_model_calls,
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
        "normalization_skip_count": normalization_skip_count,
        "normalization_hard_reject_count": (
            normalization_hard_reject_count
        ),
        "model_init_seconds": model_init_seconds,
        "inference_seconds_total": inference_seconds,
        "downloaded_source_bytes": acquisition_bytes,
        "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
    }

    provenance_membership_ok = True
    exact_quote_locatability = True
    for paper in papers:
        proposition_by_id = {
            str(proposition["proposition_id"]): proposition
            for region in paper["selected_regions"]
            for proposition in region["propositions"]
        }
        for evidence in paper["aggregation"]["evidence"]:
            proposition = proposition_by_id.get(
                str(evidence["candidate_id"])
            )
            if proposition is None:
                provenance_membership_ok = False
                exact_quote_locatability = False
                continue
            if (
                evidence["raw_source_form"]
                != proposition["raw_source_form"]
                or int(evidence["page_number"])
                != int(proposition["page_number"])
            ):
                exact_quote_locatability = False

    hard_safety = {
        "fabricated_or_unlocatable_quote_count_zero": (
            provenance_membership_ok
        ),
        "exact_quote_locatability_one": exact_quote_locatability,
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
            "LOCAL_EXTRACT_WAVE6_PAPER_GLOBAL_PROPOSITION_RANKING"
        ),
        "source_run_id": fixture["source_run_id"],
        "proposal_selector": {
            "id": SELECTOR_ID,
            "model_calls": 0,
            "scope": "PAPER_SELECTED_PROPOSITIONS_THEN_ONE_PER_REGION",
        },
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
                "continuity; selector-stage source alignment is diagnostic."
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
        "wave5_baseline": {
            "raw_decision": "REJECT",
            "scientific_disposition": (
                "MIS_SPECIFIED_SELECTOR_EXPERIMENT_DO_NOT_PROMOTE"
            ),
            "job_local_completion_fraction": 0.7777777777777778,
            "accepted_claim_count": 70,
            "selected_proposition_mean_f1": 0.2767904408888015,
            "final_normalized_mean_f1": 0.28556888629352395,
            "projected_remote_input_token_reduction": (
                0.8361297489048741
            ),
            "artifact_id": 12469624778,
            "report_sha256": (
                "2f633729c0d1948248e3f30d17e0b1fd8459519a7bf656e5d4feb3886b286b15"
            ),
            "paper_global_offline_selector_mean_f1": (
                0.30305743036333815
            ),
        },
        "wave4_baseline": {
            "decision": "PARTIAL",
            "job_local_completion_fraction": 0.7777777777777778,
            "accepted_claim_count": 50,
            "available_exact_proposition_mean_f1": (
                0.31908516604168774
            ),
            "qwen_selected_proposition_mean_f1": (
                0.21498668482680108
            ),
            "final_normalized_mean_f1": 0.17839877907935986,
            "projected_remote_input_token_reduction": (
                0.6980722200822352
            ),
            "artifact_id": 12469556387,
            "report_sha256": (
                "f65f3cb55ded5a221d5691b14d9a804efa67192fb9c22dd2647d6a7486bc1e7d"
            ),
        },
    }
    report["wave6_decision"] = decide_wave6(report)
    return report
