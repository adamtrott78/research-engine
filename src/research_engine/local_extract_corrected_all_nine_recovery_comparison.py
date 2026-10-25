from __future__ import annotations

from typing import Any

from research_engine.local_extract_cross_paper_recovery_comparison import (
    PAPER_COUNT,
    RECOVERY_DEPTH,
    RESERVATION_COUNT,
    TOTAL_BUDGET,
    _generated_paper,
    _stress_case,
    _tier1_rankings,
)
from research_engine.local_extract_diagnostics import full_pdf_propositions
from research_engine.local_extract_wave1 import (
    download_exact_pdf,
    extract_pdf_pages,
)


def _baseline_report_paper(
    report: dict[str, Any],
    item_id: str,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in report.get("papers") or []
        if str(row.get("item_id")) == item_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one all-nine baseline row {item_id}, got {len(rows)}"
        )
    return rows[0]


def _materialize_selected(
    *,
    propositions: list[dict[str, Any]],
    baseline_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(baseline_row.get("final_complete")):
        raise RuntimeError("reviewed baseline row is not final_complete")
    if int(baseline_row.get("final_selected_count", -1)) != TOTAL_BUDGET:
        raise RuntimeError("reviewed baseline row is not exact budget28")
    if int(baseline_row.get("final_event_count", -1)) != RESERVATION_COUNT:
        raise RuntimeError("reviewed baseline row does not have four events")

    selected_ids = [
        str(value)
        for value in baseline_row.get("final_selected_proposition_ids") or []
    ]
    events = [
        dict(row)
        for row in baseline_row.get("final_reservation_events") or []
    ]
    if len(selected_ids) != TOTAL_BUDGET:
        raise RuntimeError("reviewed baseline selected-id count mismatch")
    if len(set(selected_ids)) != TOTAL_BUDGET:
        raise RuntimeError("reviewed baseline selected ids are not unique")
    if len(events) != RESERVATION_COUNT:
        raise RuntimeError("reviewed baseline event count mismatch")

    source = {
        str(row["proposition_id"]): dict(row)
        for row in propositions
    }
    missing = [pid for pid in selected_ids if pid not in source]
    if missing:
        raise RuntimeError(
            "reviewed baseline propositions are not locatable: "
            + ",".join(missing)
        )
    selected = [dict(source[pid]) for pid in selected_ids]

    event_ids = [str(row["proposition_id"]) for row in events]
    if len(set(event_ids)) != RESERVATION_COUNT:
        raise RuntimeError("reviewed reservation event ids are not unique")
    if not all(pid in selected_ids for pid in event_ids):
        raise RuntimeError("reviewed reservation event missing from selected set")
    if not all(int(row.get("tier", -1)) == 1 for row in events):
        raise RuntimeError("reviewed reservation event is not tier1")
    if not all(int(row.get("frontier_rank", -1)) >= 25 for row in events):
        raise RuntimeError("reviewed reservation event frontier rank is invalid")
    return selected, events


def _materialize_baseline(
    *,
    propositions: list[dict[str, Any]],
    generated_paper: dict[str, Any],
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    selected, events = _materialize_selected(
        propositions=propositions,
        baseline_row=baseline_row,
    )
    _, tier1_source, tier1_author = _tier1_rankings(
        propositions=propositions,
        generated_paper=generated_paper,
    )
    return {
        "complete": True,
        "reason": None,
        "selected": selected,
        "events": events,
        "tier1_source": tier1_source,
        "tier1_author": tier1_author,
    }


def _verdict(
    *,
    safe: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    if not safe:
        return {
            "cross_paper_recovery_decision": "INVALID_EXECUTION",
            "interpretation": (
                "All-nine baseline artifact identity/materialization, generated "
                "intent identity, corpus/PDF provenance, stress count, or a frozen "
                "hard boundary failed; no scientific verdict."
            ),
        }

    local_only = counts["local_only_exact_recovery_count"]
    allocation_only = counts["allocation_only_exact_recovery_count"]
    both = counts["both_exact_recovery_count"]
    local_total = counts[
        "local_exact_removed_proposition_recovery_count"
    ]
    allocation_total = counts[
        "allocation_exact_removed_proposition_recovery_count"
    ]

    if local_only > 0 and allocation_only > 0:
        return {
            "cross_paper_recovery_decision": "COMPLEMENTARY_STRUCTURAL_RECOVERY",
            "interpretation": (
                "Local context and surviving-intent allocation recover different "
                "origin-dropped selections across the complete 36-case corpus."
            ),
        }
    if local_only > 0 and allocation_only == 0:
        return {
            "cross_paper_recovery_decision": "LOCAL_STRUCTURAL_ADVANTAGE",
            "interpretation": (
                "Radius-1 rescue adds structural recovery unavailable to global "
                "surviving-intent allocation under the fixed all-nine baseline."
            ),
        }
    if allocation_only > 0 and local_only == 0:
        return {
            "cross_paper_recovery_decision": "SURVIVING_ALLOCATION_ADVANTAGE",
            "interpretation": (
                "Global surviving-intent allocation adds structural recovery "
                "unavailable to radius-1 rescue under the fixed all-nine baseline."
            ),
        }
    if local_total == 0 and allocation_total == 0:
        return {
            "cross_paper_recovery_decision": "BOTH_STRUCTURALLY_WEAK",
            "interpretation": (
                "Neither frozen static recovery family reconstructs any "
                "origin-dropped selection."
            ),
        }
    if local_only == 0 and allocation_only == 0 and both > 0:
        return {
            "cross_paper_recovery_decision": "EQUIVALENT_STRUCTURAL_RECOVERY",
            "interpretation": (
                "Both frozen mechanisms recover exactly the same stress cases."
            ),
        }
    raise RuntimeError("frozen corrected interpretation partition is not exhaustive")


def evaluate_corrected_all_nine_recovery_comparison(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    generated: dict[str, Any],
    generated_intents_sha256: str,
    baseline_report: dict[str, Any],
    baseline_report_sha256: str,
    design: dict[str, Any],
    workdir: Any,
) -> dict[str, Any]:
    if (
        design["design_id"]
        != "local-extract-corrected-all-nine-recovery-comparison-v1"
    ):
        raise RuntimeError("unexpected corrected all-nine design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "ALL_NINE_FIXED_BASELINE_RECOVERY_COMPARISON_V1":
        raise RuntimeError("unexpected corrected all-nine experiment")

    frozen = experiment["frozen_inputs"]
    if int(frozen["paper_count"]) != PAPER_COUNT:
        raise RuntimeError("unexpected frozen paper count")
    if int(frozen["fixed_baseline_selected_count"]) != TOTAL_BUDGET:
        raise RuntimeError("unexpected frozen selected count")
    if int(frozen["fixed_baseline_reservation_event_count"]) != RESERVATION_COUNT:
        raise RuntimeError("unexpected frozen reservation event count")
    if int(frozen["recovery_evidence_depth"]) != RECOVERY_DEPTH:
        raise RuntimeError("unexpected frozen recovery depth")
    if int(frozen["neighborhood_radius"]) != 1:
        raise RuntimeError("unexpected frozen neighborhood radius")

    generated_identity_pass = (
        generated_intents_sha256
        == str(frozen["generated_intents_sha256"])
    )
    baseline_report_identity_pass = (
        baseline_report_sha256
        == str(frozen["all_nine_baseline_report_sha256"])
    )
    generated_boundary_pass = bool(
        generated.get("scientific_state_materialized") is False
        and generated.get("production_routing_changed") is False
    )
    baseline_report_gate_pass = bool(
        baseline_report.get("baseline_depth_fallback_decision")
        == "DEPTH_FALLBACK_COMPLETES_ALL_NINE"
        and int(baseline_report.get("paper_count", -1)) == PAPER_COUNT
        and int(baseline_report.get("final_complete_paper_count", -1))
        == PAPER_COUNT
    )

    fixture_papers = [dict(row) for row in fixture["papers"]]
    if len(fixture_papers) != PAPER_COUNT:
        raise RuntimeError("fixture paper count mismatch")
    fixture_ids = {str(row["item_id"]) for row in fixture_papers}
    generated_ids = {
        str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        for row in generated["papers"]
    }
    baseline_ids = {
        str(row["item_id"])
        for row in baseline_report.get("papers") or []
    }
    paper_set_identity_pass = bool(
        fixture_ids == generated_ids == baseline_ids
    )

    per_intent_depth = int(wave7a3_contract["retrieval"]["per_intent_depth"])
    if per_intent_depth != RECOVERY_DEPTH:
        raise RuntimeError("Wave7A.3 per-intent depth is not frozen top12")

    papers: list[dict[str, Any]] = []
    all_materialized = True
    all_locatable = True
    total_stresses = 0
    counts = {
        "local_exact_removed_proposition_recovery_count": 0,
        "allocation_exact_removed_proposition_recovery_count": 0,
        "both_exact_recovery_count": 0,
        "local_only_exact_recovery_count": 0,
        "allocation_only_exact_recovery_count": 0,
        "neither_exact_recovery_count": 0,
        "allocation_exact_recoveries_from_shallow_frontier_count": 0,
        "allocation_exact_recoveries_from_depth_extension_count": 0,
    }
    secondary = {
        "removed_proposition_on_historical_evidence_page_count": 0,
        "local_selected_on_historical_evidence_page_count": 0,
        "allocation_selected_on_historical_evidence_page_count": 0,
        "local_same_page_as_removed_count": 0,
        "allocation_same_page_as_removed_count": 0,
    }
    cost = {
        "local_total_unique_candidate_count": 0,
        "allocation_total_unique_candidate_count": 0,
        "local_total_candidate_raw_source_chars": 0,
        "allocation_total_candidate_raw_source_chars": 0,
        "local_total_selected_raw_source_chars": 0,
        "allocation_total_selected_raw_source_chars": 0,
    }

    for paper in fixture_papers:
        item_id = str(paper["item_id"])
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise RuntimeError(f"page-count mismatch for {item_id}")
        propositions = full_pdf_propositions(pages)
        source = {
            str(row["proposition_id"]): dict(row)
            for row in propositions
        }
        generated_paper = _generated_paper(generated, item_id)
        baseline_row = _baseline_report_paper(baseline_report, item_id)

        materialized = bool(
            bool(baseline_row.get("final_complete"))
            and int(baseline_row.get("final_selected_count", -1))
            == TOTAL_BUDGET
            and int(baseline_row.get("final_event_count", -1))
            == RESERVATION_COUNT
            and bool(
                baseline_row.get(
                    "final_selected_propositions_locatable",
                    False,
                )
            )
        )
        all_materialized = all_materialized and materialized
        if not materialized:
            papers.append({
                "item_id": item_id,
                "arxiv_id": str(paper["arxiv_id"]),
                "title": title,
                "acquisition": acquisition,
                "page_count": len(pages),
                "proposition_count": len(propositions),
                "baseline_materialized": False,
                "baseline_selected_count": int(
                    baseline_row.get("final_selected_count", -1)
                ),
                "baseline_event_count": int(
                    baseline_row.get("final_event_count", -1)
                ),
                "baseline_propositions_locatable": False,
                "baseline_reservation_events": list(
                    baseline_row.get("final_reservation_events") or []
                ),
                "stresses": [],
            })
            continue

        baseline = _materialize_baseline(
            propositions=propositions,
            generated_paper=generated_paper,
            baseline_row=baseline_row,
        )
        selected_ids = [
            str(row["proposition_id"])
            for row in baseline["selected"]
        ]
        locatable = bool(
            len(selected_ids) == TOTAL_BUDGET
            and len(set(selected_ids)) == TOTAL_BUDGET
            and all(pid in source for pid in selected_ids)
        )
        all_locatable = all_locatable and locatable

        stresses = []
        if locatable:
            for index in range(RESERVATION_COUNT):
                stress = _stress_case(
                    paper=paper,
                    propositions=propositions,
                    baseline=baseline,
                    event_index=index,
                )
                stresses.append(stress)
                total_stresses += 1

                local_exact = bool(
                    stress["local"]["exact_removed_proposition_recovery"]
                )
                allocation_exact = bool(
                    stress["allocation"]["exact_removed_proposition_recovery"]
                )
                counts[
                    "local_exact_removed_proposition_recovery_count"
                ] += int(local_exact)
                counts[
                    "allocation_exact_removed_proposition_recovery_count"
                ] += int(allocation_exact)
                if local_exact and allocation_exact:
                    counts["both_exact_recovery_count"] += 1
                elif local_exact:
                    counts["local_only_exact_recovery_count"] += 1
                elif allocation_exact:
                    counts["allocation_only_exact_recovery_count"] += 1
                else:
                    counts["neither_exact_recovery_count"] += 1

                if allocation_exact:
                    stratum = stress["allocation"][
                        "exact_recovery_rank_stratum"
                    ]
                    if stratum == "SHALLOW_FRONTIER":
                        counts[
                            "allocation_exact_recoveries_from_shallow_frontier_count"
                        ] += 1
                    elif stratum == "DEPTH_EXTENSION":
                        counts[
                            "allocation_exact_recoveries_from_depth_extension_count"
                        ] += 1
                    else:
                        raise RuntimeError(
                            f"unexpected exact-recovery stratum {stratum}"
                        )

                secondary[
                    "removed_proposition_on_historical_evidence_page_count"
                ] += int(stress["removed_on_historical_evidence_page"])
                secondary[
                    "local_selected_on_historical_evidence_page_count"
                ] += int(
                    stress["local"]["chosen_on_historical_evidence_page"]
                )
                secondary[
                    "allocation_selected_on_historical_evidence_page_count"
                ] += int(
                    stress["allocation"][
                        "chosen_on_historical_evidence_page"
                    ]
                )
                secondary["local_same_page_as_removed_count"] += int(
                    stress["local"]["chosen_same_page_as_removed"]
                )
                secondary["allocation_same_page_as_removed_count"] += int(
                    stress["allocation"]["chosen_same_page_as_removed"]
                )

                cost["local_total_unique_candidate_count"] += int(
                    stress["local"]["generated_candidate_count"]
                )
                cost["allocation_total_unique_candidate_count"] += int(
                    stress["allocation"]["candidate_count"]
                )
                cost["local_total_candidate_raw_source_chars"] += int(
                    stress["local"]["candidate_raw_source_chars"]
                )
                cost["allocation_total_candidate_raw_source_chars"] += int(
                    stress["allocation"]["candidate_raw_source_chars"]
                )
                if stress["local"]["chosen"] is not None:
                    cost["local_total_selected_raw_source_chars"] += len(
                        str(stress["local"]["chosen"]["raw_source_form"])
                    )
                if stress["allocation"]["chosen"] is not None:
                    cost["allocation_total_selected_raw_source_chars"] += len(
                        str(stress["allocation"]["chosen"]["raw_source_form"])
                    )

        papers.append({
            "item_id": item_id,
            "arxiv_id": str(paper["arxiv_id"]),
            "title": title,
            "acquisition": acquisition,
            "page_count": len(pages),
            "proposition_count": len(propositions),
            "historical_evidence_pages": [
                int(value) for value in paper["historical_evidence_pages"]
            ],
            "baseline_materialized": materialized,
            "baseline_selected_count": len(baseline["selected"]),
            "baseline_event_count": len(baseline["events"]),
            "baseline_propositions_locatable": locatable,
            "baseline_reservation_events": list(baseline["events"]),
            "stresses": stresses,
        })

    expected_stresses = int(
        experiment["stress_generator"][
            "expected_stress_cases_if_all_papers_complete"
        ]
    )
    complete_gate = bool(
        all_materialized
        and all_locatable
        and len(papers) == PAPER_COUNT
        and total_stresses == expected_stresses
    )

    hard_safety = {
        "all_nine_baseline_report_identity_pass": (
            baseline_report_identity_pass
        ),
        "all_nine_baseline_expected_gate_pass": baseline_report_gate_pass,
        "generated_intents_identity_pass": generated_identity_pass,
        "generated_intents_boundary_pass": generated_boundary_pass,
        "paper_set_identity_pass": paper_set_identity_pass,
        "all_nine_baselines_materialized": all_materialized,
        "all_baseline_propositions_locatable": all_locatable,
        "expected_stress_case_count_pass": (
            total_stresses == expected_stresses
        ),
        "exact_all_nine_baseline_artifact_required": True,
        "baseline_recomputation_for_selection": False,
        "teacher_used_for_selection": False,
        "teacher_used_for_stress_generation": False,
        "teacher_used_for_candidate_generation": False,
        "teacher_used_for_candidate_ranking": False,
        "historical_page_oracle_used_only_posthoc": True,
        "proposition_semantic_generalization_claim": False,
        "new_query_generation_false": True,
        "new_model_or_ranker_false": True,
        "pairwise_semantic_similarity_false": True,
        "threshold_calibration_false": True,
        "recovery_arm_changed_false": True,
        "nli_rescue_false": True,
        "predicate_argument_rescue_false": True,
        "amr_rescue_false": True,
        "production_routing_changed_false": True,
        "qwen_verify_changed_false": True,
        "wave7b_started_false": True,
        "semantic_control_or_ontology_output_false": True,
    }
    safe = bool(
        baseline_report_identity_pass
        and baseline_report_gate_pass
        and generated_identity_pass
        and generated_boundary_pass
        and paper_set_identity_pass
        and complete_gate
    )

    decision = _verdict(
        safe=safe,
        counts=counts,
    )
    return {
        "schema_version": 1,
        "operation": (
            "LOCAL_EXTRACT_CORRECTED_ALL_NINE_STRUCTURAL_RECOVERY_COMPARISON"
        ),
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "experiment_id": experiment["id"],
        "generated_intents_sha256": generated_intents_sha256,
        "all_nine_baseline_report_sha256": baseline_report_sha256,
        **decision,
        "paper_count": len(papers),
        "valid_paper_count": sum(
            int(
                row["baseline_materialized"]
                and row["baseline_propositions_locatable"]
            )
            for row in papers
        ),
        "valid_stress_case_count": total_stresses,
        **counts,
        "secondary_page_context": secondary,
        "cost": cost,
        "papers": papers,
        "hard_safety": hard_safety,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "rejected_mechanism_rescue": False,
    }
