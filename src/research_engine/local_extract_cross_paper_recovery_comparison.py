from __future__ import annotations

from typing import Any

from research_engine.local_extract_diagnostics import full_pdf_propositions
from research_engine.local_extract_facet_reserved_evidence_planner import (
    _rank_intents,
    _reservation_stream,
)
from research_engine.local_extract_neighborhood_rescue import _candidate_neighbors
from research_engine.local_extract_oracle_query import _bm25_scores, _rank
from research_engine.local_extract_post_wave7a3_coverage import (
    CONTROL_ID,
    RESIDUAL_ID,
)
from research_engine.local_extract_wave1 import (
    download_exact_pdf,
    extract_pdf_pages,
)
from research_engine.local_extract_wave7_evaluate import _rrf_merge


PAPER_COUNT = 9
CORE_BUDGET = 24
TOTAL_BUDGET = 28
RESERVATION_COUNT = TOTAL_BUDGET - CORE_BUDGET
RESERVATION_DEPTH = 3
RECOVERY_DEPTH = 12


def _generated_paper(
    generated: dict[str, Any],
    item_id: str,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in generated["papers"]
        if str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        == item_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one generated paper {item_id}, got {len(rows)}"
        )
    return rows[0]


def _author_roles(author_arm: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["anchor_id"]): str(row["role"])
        for row in author_arm.get("anchors") or []
    }


def _tier1_rankings(
    *,
    propositions: list[dict[str, Any]],
    generated_paper: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_arm = dict(generated_paper["generator_arms"][CONTROL_ID])
    author_arm = dict(generated_paper["generator_arms"][RESIDUAL_ID])

    source_rankings = _rank_intents(
        propositions=propositions,
        accepted=[dict(row) for row in source_arm.get("accepted") or []],
        generator_id=CONTROL_ID,
        depth=RECOVERY_DEPTH,
    )
    author_rankings = _rank_intents(
        propositions=propositions,
        accepted=[dict(row) for row in author_arm.get("accepted") or []],
        generator_id=RESIDUAL_ID,
        depth=RECOVERY_DEPTH,
        author_roles=_author_roles(author_arm),
    )
    tier1_source = [
        row for row in source_rankings
        if str(row.get("facet") or "") == "method"
    ]
    tier1_author = [
        row for row in author_rankings
        if str(row.get("anchor_role") or "") == "contribution"
    ]
    return source_rankings, tier1_source, tier1_author


def _teacher_blind_baseline(
    *,
    propositions: list[dict[str, Any]],
    generated_paper: dict[str, Any],
    rrf_k: int,
    per_intent_depth: int,
) -> dict[str, Any]:
    source_rankings, tier1_source, tier1_author = _tier1_rankings(
        propositions=propositions,
        generated_paper=generated_paper,
    )
    core = _rrf_merge(
        [list(row["ranking"]) for row in source_rankings],
        rrf_k=rrf_k,
        per_intent_depth=per_intent_depth,
        budget=CORE_BUDGET,
    )
    if len(core) != CORE_BUDGET:
        return {
            "complete": False,
            "reason": f"CORE_SIZE:{len(core)}",
            "selected": [],
            "events": [],
            "tier1_source": tier1_source,
            "tier1_author": tier1_author,
        }

    tiers = [{
        "tier": 1,
        "source": tier1_source,
        "author": tier1_author,
    }]
    selected, events = _reservation_stream(
        core=core,
        tiers=tiers,
        maximum_budget=TOTAL_BUDGET,
    )
    complete = (
        len(selected) == TOTAL_BUDGET
        and len(events) == RESERVATION_COUNT
        and all(int(event["tier"]) == 1 for event in events)
    )
    return {
        "complete": complete,
        "reason": None if complete else (
            f"BASELINE_SELECTED:{len(selected)};EVENTS:{len(events)}"
        ),
        "selected": selected,
        "events": events,
        "tier1_source": tier1_source,
        "tier1_author": tier1_author,
    }


def _surviving_paths(
    *,
    baseline: dict[str, Any],
    removed_event: dict[str, Any],
) -> list[dict[str, Any]]:
    removed_key = (
        str(removed_event["generator_id"]),
        int(removed_event["intent_index"]),
    )
    paths = []
    order = 0
    for row in list(baseline["tier1_source"]) + list(baseline["tier1_author"]):
        key = (str(row["generator_id"]), int(row["intent_index"]))
        if key == removed_key:
            continue
        copied = dict(row)
        copied["surviving_path_order"] = order
        copied["group"] = (
            str(row.get("facet"))
            if row["generator_id"] == CONTROL_ID
            else str(row.get("anchor_role"))
        )
        paths.append(copied)
        order += 1
    return paths


def _score_local_candidates(
    *,
    candidates: list[dict[str, Any]],
    paths: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relation_order = {"predecessor": 0, "successor": 1}
    eligible = []
    ineligible = []
    for candidate in candidates:
        pid = str(candidate["proposition_id"])
        hits = []
        for path in paths:
            matched = next(
                (
                    row for row in path["ranking"][:RECOVERY_DEPTH]
                    if str(row["proposition_id"]) == pid
                ),
                None,
            )
            if matched is None:
                continue
            hits.append({
                "generator_id": str(path["generator_id"]),
                "intent_index": int(path["intent_index"]),
                "surviving_path_order": int(path["surviving_path_order"]),
                "bm25_rank": int(matched["oracle_rank"]),
            })
        row = dict(candidate)
        row["surviving_path_hits"] = hits
        row["support_count"] = len(hits)
        row["best_rank"] = (
            min(int(hit["bm25_rank"]) for hit in hits)
            if hits else None
        )
        if hits:
            eligible.append(row)
        else:
            ineligible.append(row)

    eligible.sort(
        key=lambda row: (
            -int(row["support_count"]),
            int(row["best_rank"]),
            int(row["seed_frontier_rank"]),
            relation_order[str(row["neighbor_relation"])],
            str(row["proposition_id"]),
        )
    )
    ineligible.sort(key=lambda row: str(row["proposition_id"]))
    return eligible, ineligible


def _allocation_candidates(
    *,
    paths: list[dict[str, Any]],
    stressed_ids: set[str],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for candidate in path["ranking"][:RECOVERY_DEPTH]:
            pid = str(candidate["proposition_id"])
            if pid in stressed_ids:
                continue
            row = by_id.setdefault(
                pid,
                {
                    "proposition_id": pid,
                    "page_number": int(candidate["page_number"]),
                    "raw_source_form": str(candidate["raw_source_form"]),
                    "surviving_path_hits": [],
                },
            )
            row["surviving_path_hits"].append({
                "generator_id": str(path["generator_id"]),
                "intent_index": int(path["intent_index"]),
                "surviving_path_order": int(path["surviving_path_order"]),
                "bm25_rank": int(candidate["oracle_rank"]),
            })

    rows = []
    for row in by_id.values():
        hits = list(row["surviving_path_hits"])
        best_rank = min(int(hit["bm25_rank"]) for hit in hits)
        earliest_path = min(
            int(hit["surviving_path_order"]) for hit in hits
        )
        copied = dict(row)
        copied["support_count"] = len(hits)
        copied["best_rank"] = best_rank
        copied["rank_stratum"] = (
            "SHALLOW_FRONTIER"
            if best_rank <= RESERVATION_DEPTH
            else "DEPTH_EXTENSION"
        )
        copied["earliest_surviving_path_order"] = earliest_path
        rows.append(copied)

    rows.sort(
        key=lambda row: (
            -int(row["support_count"]),
            int(row["best_rank"]),
            int(row["earliest_surviving_path_order"]),
            str(row["proposition_id"]),
        )
    )
    return rows


def _seeds_for_stress(
    *,
    events: list[dict[str, Any]],
    removed_index: int,
    source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    seeds = []
    for index, event in enumerate(events):
        if index == removed_index:
            continue
        pid = str(event["proposition_id"])
        row = source[pid]
        seeds.append({
            "frontier_rank": int(event["frontier_rank"]),
            "proposition_id": pid,
            "page_number": int(row["page_number"]),
        })
    return seeds


def _stress_case(
    *,
    paper: dict[str, Any],
    propositions: list[dict[str, Any]],
    baseline: dict[str, Any],
    event_index: int,
) -> dict[str, Any]:
    source = {
        str(row["proposition_id"]): dict(row)
        for row in propositions
    }
    selected = [dict(row) for row in baseline["selected"]]
    event = dict(baseline["events"][event_index])
    removed_pid = str(event["proposition_id"])
    if removed_pid not in source:
        raise RuntimeError(f"removed proposition not locatable: {removed_pid}")

    stressed = [
        dict(row)
        for row in selected
        if str(row["proposition_id"]) != removed_pid
    ]
    if len(stressed) != TOTAL_BUDGET - 1:
        raise RuntimeError("origin-drop stress did not produce 27 propositions")
    stressed_ids = {
        str(row["proposition_id"]) for row in stressed
    }
    paths = _surviving_paths(
        baseline=baseline,
        removed_event=event,
    )

    seeds = _seeds_for_stress(
        events=list(baseline["events"]),
        removed_index=event_index,
        source=source,
    )
    local_generated = _candidate_neighbors(
        seeds=seeds,
        propositions=propositions,
        stressed_ids=stressed_ids,
    )
    local_eligible, local_ineligible = _score_local_candidates(
        candidates=local_generated,
        paths=paths,
    )
    allocation = _allocation_candidates(
        paths=paths,
        stressed_ids=stressed_ids,
    )

    local_chosen = dict(local_eligible[0]) if local_eligible else None
    allocation_chosen = dict(allocation[0]) if allocation else None
    local_exact = bool(
        local_chosen is not None
        and str(local_chosen["proposition_id"]) == removed_pid
    )
    allocation_exact = bool(
        allocation_chosen is not None
        and str(allocation_chosen["proposition_id"]) == removed_pid
    )

    historical_pages = {
        int(value) for value in paper["historical_evidence_pages"]
    }
    removed_page = int(source[removed_pid]["page_number"])

    return {
        "stress_index": event_index,
        "removed_event": event,
        "removed_proposition_id": removed_pid,
        "removed_page_number": removed_page,
        "removed_on_historical_evidence_page": removed_page in historical_pages,
        "excluded_selecting_intent": {
            "generator_id": str(event["generator_id"]),
            "intent_index": int(event["intent_index"]),
        },
        "stressed_selected_count": len(stressed),
        "surviving_path_count": len(paths),
        "local": {
            "generated_candidate_count": len(local_generated),
            "eligible_candidate_count": len(local_eligible),
            "ineligible_candidate_count": len(local_ineligible),
            "candidate_raw_source_chars": sum(
                len(str(row["raw_source_form"]))
                for row in local_generated
            ),
            "chosen": local_chosen,
            "exact_removed_proposition_recovery": local_exact,
            "chosen_on_historical_evidence_page": (
                bool(
                    local_chosen is not None
                    and int(local_chosen["page_number"]) in historical_pages
                )
            ),
            "chosen_same_page_as_removed": bool(
                local_chosen is not None
                and int(local_chosen["page_number"]) == removed_page
            ),
        },
        "allocation": {
            "candidate_count": len(allocation),
            "candidate_raw_source_chars": sum(
                len(str(row["raw_source_form"]))
                for row in allocation
            ),
            "chosen": allocation_chosen,
            "exact_removed_proposition_recovery": allocation_exact,
            "exact_recovery_rank_stratum": (
                str(allocation_chosen["rank_stratum"])
                if allocation_exact and allocation_chosen is not None
                else None
            ),
            "chosen_on_historical_evidence_page": (
                bool(
                    allocation_chosen is not None
                    and int(allocation_chosen["page_number"]) in historical_pages
                )
            ),
            "chosen_same_page_as_removed": bool(
                allocation_chosen is not None
                and int(allocation_chosen["page_number"]) == removed_page
            ),
        },
    }


def _verdict(
    *,
    safe: bool,
    baseline_complete: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    if not safe or not baseline_complete:
        return {
            "cross_paper_recovery_decision": "INSUFFICIENT_BASELINE_GENERALIZATION",
            "interpretation": (
                "At least one all-nine baseline or identity/provenance/"
                "teacher-boundary gate failed; no recovery-family preference."
            ),
        }

    local_only = counts["local_only_exact_recovery_count"]
    allocation_only = counts["allocation_only_exact_recovery_count"]
    both = counts["both_exact_recovery_count"]
    local_total = counts["local_exact_removed_proposition_recovery_count"]
    allocation_total = counts[
        "allocation_exact_removed_proposition_recovery_count"
    ]

    if local_only > 0 and allocation_only > 0:
        return {
            "cross_paper_recovery_decision": "COMPLEMENTARY_STRUCTURAL_RECOVERY",
            "interpretation": (
                "Local context and surviving-intent allocation recover different "
                "origin-dropped selections."
            ),
        }
    if local_only > 0 and allocation_only == 0:
        return {
            "cross_paper_recovery_decision": "LOCAL_STRUCTURAL_ADVANTAGE",
            "interpretation": (
                "Radius-1 rescue adds structural recovery unavailable to the "
                "global surviving-intent allocation arm."
            ),
        }
    if allocation_only > 0 and local_only == 0:
        return {
            "cross_paper_recovery_decision": "SURVIVING_ALLOCATION_ADVANTAGE",
            "interpretation": (
                "Global surviving-intent allocation adds structural recovery "
                "unavailable to radius-1 rescue."
            ),
        }
    if local_total == 0 and allocation_total == 0:
        return {
            "cross_paper_recovery_decision": "BOTH_STRUCTURALLY_WEAK",
            "interpretation": (
                "Neither static recovery family reconstructs any origin-dropped "
                "selection."
            ),
        }
    if local_only == 0 and allocation_only == 0 and both > 0:
        return {
            "cross_paper_recovery_decision": "EQUIVALENT_STRUCTURAL_RECOVERY",
            "interpretation": (
                "Both mechanisms recover exactly the same stress cases; only "
                "cost/simplicity can discriminate the bounded follow-up."
            ),
        }
    raise RuntimeError("frozen interpretation partition is not exhaustive")


def evaluate_cross_paper_recovery_comparison(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    generated: dict[str, Any],
    design: dict[str, Any],
    generated_intents_sha256: str,
    workdir: Any,
) -> dict[str, Any]:
    if design["design_id"] != "local-extract-cross-paper-recovery-comparison-v1":
        raise RuntimeError("unexpected frozen cross-paper design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "LEAVE_ONE_RESERVATION_ORIGIN_OUT_COMPARISON_V1":
        raise RuntimeError("unexpected frozen experiment")

    frozen = experiment["frozen_inputs"]
    if int(frozen["paper_count"]) != PAPER_COUNT:
        raise RuntimeError("unexpected frozen paper count")
    if int(frozen["baseline_core_budget"]) != CORE_BUDGET:
        raise RuntimeError("unexpected core budget")
    if int(frozen["baseline_total_budget"]) != TOTAL_BUDGET:
        raise RuntimeError("unexpected total budget")
    if int(frozen["baseline_reservation_depth"]) != RESERVATION_DEPTH:
        raise RuntimeError("unexpected reservation depth")
    if int(frozen["recovery_evidence_depth"]) != RECOVERY_DEPTH:
        raise RuntimeError("unexpected recovery depth")
    if int(frozen["neighborhood_radius"]) != 1:
        raise RuntimeError("unexpected neighborhood radius")
    if str(frozen["retriever"]) != "BM25_DIRECT":
        raise RuntimeError("unexpected retriever")

    generated_identity_pass = (
        generated_intents_sha256
        == str(frozen["generated_intents_sha256"])
    )
    generated_boundary_pass = bool(
        generated.get("scientific_state_materialized") is False
        and generated.get("production_routing_changed") is False
    )

    fixture_papers = [dict(row) for row in fixture["papers"]]
    if len(fixture_papers) != PAPER_COUNT:
        raise RuntimeError("fixture paper count mismatch")
    generated_ids = {
        str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        for row in generated["papers"]
    }
    fixture_ids = {str(row["item_id"]) for row in fixture_papers}
    generated_paper_set_pass = generated_ids == fixture_ids

    per_intent_depth = int(wave7a3_contract["retrieval"]["per_intent_depth"])
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    if per_intent_depth != RECOVERY_DEPTH:
        raise RuntimeError("Wave7A.3 per-intent depth is not frozen top12")

    papers = []
    all_baselines_complete = True
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
        page_count_pass = len(pages) == int(paper["total_pdf_pages"])
        if not page_count_pass:
            raise RuntimeError(f"page-count mismatch for {item_id}")
        propositions = full_pdf_propositions(pages)
        source = {
            str(row["proposition_id"]): dict(row)
            for row in propositions
        }
        generated_paper = _generated_paper(generated, item_id)
        baseline = _teacher_blind_baseline(
            propositions=propositions,
            generated_paper=generated_paper,
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
        )
        baseline_ids = [
            str(row["proposition_id"])
            for row in baseline["selected"]
        ]
        baseline_locatable = bool(
            baseline["complete"]
            and len(baseline_ids) == TOTAL_BUDGET
            and len(set(baseline_ids)) == TOTAL_BUDGET
            and all(pid in source for pid in baseline_ids)
        )
        all_baselines_complete = (
            all_baselines_complete and bool(baseline["complete"])
        )
        all_locatable = all_locatable and baseline_locatable

        stresses = []
        if baseline["complete"] and baseline_locatable:
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
            "baseline_complete": bool(baseline["complete"]),
            "baseline_reason": baseline["reason"],
            "baseline_selected_count": len(baseline["selected"]),
            "baseline_event_count": len(baseline["events"]),
            "baseline_propositions_locatable": baseline_locatable,
            "baseline_reservation_events": baseline["events"],
            "stresses": stresses,
        })

    expected_stresses = int(
        experiment["stress_generator"][
            "expected_stress_cases_if_all_papers_complete"
        ]
    )
    baseline_complete = bool(
        all_baselines_complete
        and all_locatable
        and total_stresses == expected_stresses
    )
    hard_safety = {
        "generated_intents_identity_pass": generated_identity_pass,
        "generated_intents_boundary_pass": generated_boundary_pass,
        "generated_paper_set_matches_fixture": generated_paper_set_pass,
        "all_nine_baselines_complete": all_baselines_complete,
        "all_baseline_propositions_locatable": all_locatable,
        "expected_stress_case_count_pass": total_stresses == expected_stresses,
        "teacher_used_for_cross_paper_baseline": False,
        "teacher_used_for_stress_generation": False,
        "teacher_used_for_candidate_generation": False,
        "teacher_used_for_candidate_ranking": False,
        "historical_page_oracle_used_only_posthoc": True,
        "proposition_semantic_generalization_claim": False,
        "new_query_generation_false": True,
        "new_model_or_ranker_false": True,
        "pairwise_semantic_similarity_false": True,
        "threshold_calibration_false": True,
        "production_routing_changed_false": True,
        "qwen_verify_changed_false": True,
        "wave7b_started_false": True,
        "semantic_control_or_ontology_output_false": True,
        "rejected_mechanism_rescue_false": True,
    }
    safe = bool(
        generated_identity_pass
        and generated_boundary_pass
        and generated_paper_set_pass
        and all_locatable
    )

    decision = _verdict(
        safe=safe,
        baseline_complete=baseline_complete,
        counts=counts,
    )
    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_CROSS_PAPER_STRUCTURAL_RECOVERY_COMPARISON",
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "experiment_id": experiment["id"],
        "generated_intents_sha256": generated_intents_sha256,
        **decision,
        "paper_count": len(papers),
        "valid_paper_count": sum(
            int(row["baseline_complete"] and row["baseline_propositions_locatable"])
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
