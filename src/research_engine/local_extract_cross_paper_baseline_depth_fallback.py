from __future__ import annotations

from typing import Any

from research_engine.local_extract_cross_paper_recovery_comparison import (
    CORE_BUDGET,
    PAPER_COUNT,
    RECOVERY_DEPTH,
    RESERVATION_COUNT,
    RESERVATION_DEPTH,
    TOTAL_BUDGET,
    _generated_paper,
    _teacher_blind_baseline,
)
from research_engine.local_extract_diagnostics import full_pdf_propositions
from research_engine.local_extract_post_wave7a3_coverage import CONTROL_ID
from research_engine.local_extract_wave1 import download_exact_pdf, extract_pdf_pages


FAIL_ITEM_ID = "arxiv:2501.15391"
EXPECTED_ORIGINAL_COMPLETE_PAPERS = 8


def _source_report_paper(
    source_report: dict[str, Any],
    item_id: str,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in source_report["papers"]
        if str(row["item_id"]) == item_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one source-report paper {item_id}, got {len(rows)}"
        )
    return rows[0]


def _event_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "frontier_rank",
        "tier",
        "visit_group",
        "round",
        "generator_id",
        "intent_index",
        "group_kind",
        "group_value",
        "bm25_rank",
        "proposition_id",
    )
    return [
        {key: row[key] for key in keys}
        for row in rows
    ]


def _fallback_append(
    *,
    intent: dict[str, Any],
    group_name: str,
    selected: list[dict[str, Any]],
    seen: set[str],
    round_number: int,
    examined: dict[tuple[str, int, str], int],
) -> dict[str, Any] | None:
    generator_id = str(intent["generator_id"])
    intent_index = int(intent["intent_index"])
    for candidate in list(intent["ranking"])[RESERVATION_DEPTH:RECOVERY_DEPTH]:
        pid = str(candidate["proposition_id"])
        examined.setdefault(
            (generator_id, intent_index, pid),
            len(str(candidate["raw_source_form"])),
        )
        if pid in seen:
            continue

        seen.add(pid)
        row = dict(candidate)
        row["selection_origin"] = "tier1_depth_fallback_reservation"
        row["reservation_tier"] = 1
        row["reservation_round"] = round_number
        row["reservation_generator"] = generator_id
        row["reservation_intent_index"] = intent_index
        row["reservation_question"] = str(intent["question"])
        row["reservation_bm25_rank"] = int(candidate["oracle_rank"])
        if generator_id == CONTROL_ID:
            row["reservation_group_kind"] = "facet"
            row["reservation_group_value"] = str(intent["facet"])
        else:
            row["reservation_group_kind"] = "anchor_role"
            row["reservation_group_value"] = str(intent["anchor_role"])
            row["reservation_anchor_id"] = str(intent["anchor_id"])
        selected.append(row)

        return {
            "frontier_rank": len(selected),
            "tier": 1,
            "visit_group": group_name,
            "round": round_number,
            "generator_id": generator_id,
            "intent_index": intent_index,
            "group_kind": str(row["reservation_group_kind"]),
            "group_value": str(row["reservation_group_value"]),
            "bm25_rank": int(row["reservation_bm25_rank"]),
            "proposition_id": pid,
            "fallback_depth_extension": True,
        }
    return None


def _apply_depth_fallback(
    baseline: dict[str, Any],
) -> dict[str, Any]:
    selected = [dict(row) for row in baseline["selected"]]
    events = [dict(row) for row in baseline["events"]]
    original_ids = [str(row["proposition_id"]) for row in selected]
    original_event_identity = _event_identity(events)

    if bool(baseline["complete"]):
        return {
            "fallback_applied": False,
            "selected": selected,
            "events": events,
            "fallback_events": [],
            "fallback_candidate_count_examined": 0,
            "fallback_raw_source_chars_examined": 0,
            "complete": True,
            "original_selected_ids": original_ids,
            "original_event_identity": original_event_identity,
        }

    # The reviewed residual is a complete 24-core with fewer than four
    # reservation events. A different failure shape is preserved as incomplete;
    # it is not repaired by inventing a broader fallback.
    if len(selected) < CORE_BUDGET:
        return {
            "fallback_applied": False,
            "selected": selected,
            "events": events,
            "fallback_events": [],
            "fallback_candidate_count_examined": 0,
            "fallback_raw_source_chars_examined": 0,
            "complete": False,
            "original_selected_ids": original_ids,
            "original_event_identity": original_event_identity,
        }

    seen = {str(row["proposition_id"]) for row in selected}
    source = [dict(row) for row in baseline["tier1_source"]]
    author = [dict(row) for row in baseline["tier1_author"]]
    examined: dict[tuple[str, int, str], int] = {}
    fallback_events: list[dict[str, Any]] = []
    round_number = 1

    while (
        len(selected) < TOTAL_BUDGET
        and len(events) < RESERVATION_COUNT
    ):
        any_added = False
        width = max(len(source), len(author))
        for index in range(width):
            for group_name, group in (("source", source), ("author", author)):
                if (
                    index >= len(group)
                    or len(selected) >= TOTAL_BUDGET
                    or len(events) >= RESERVATION_COUNT
                ):
                    continue
                event = _fallback_append(
                    intent=group[index],
                    group_name=group_name,
                    selected=selected,
                    seen=seen,
                    round_number=round_number,
                    examined=examined,
                )
                if event is not None:
                    events.append(event)
                    fallback_events.append(event)
                    any_added = True
        if not any_added:
            break
        round_number += 1

    for rank, row in enumerate(selected, start=1):
        row["frontier_rank"] = rank

    complete = bool(
        len(selected) == TOTAL_BUDGET
        and len(events) == RESERVATION_COUNT
    )
    return {
        "fallback_applied": True,
        "selected": selected,
        "events": events,
        "fallback_events": fallback_events,
        "fallback_candidate_count_examined": len(examined),
        "fallback_raw_source_chars_examined": sum(examined.values()),
        "complete": complete,
        "original_selected_ids": original_ids,
        "original_event_identity": original_event_identity,
    }


def _classify(
    *,
    safe: bool,
    identity_invariants_pass: bool,
    final_complete_paper_count: int,
) -> dict[str, Any]:
    if not safe:
        return {
            "baseline_depth_fallback_decision": "INVALID_EXECUTION",
            "interpretation": (
                "Artifact/PDF identity, provenance, exact baseline replay, or "
                "teacher-boundary safety failed; no scientific verdict."
            ),
        }
    if not identity_invariants_pass:
        return {
            "baseline_depth_fallback_decision": "BASELINE_IDENTITY_REGRESSION",
            "interpretation": (
                "Fallback altered a previously complete baseline or the failing "
                "paper's original three-event prefix, so the operator is not "
                "isolated to the frozen residual."
            ),
        }
    if final_complete_paper_count == PAPER_COUNT:
        return {
            "baseline_depth_fallback_decision": (
                "DEPTH_FALLBACK_COMPLETES_ALL_NINE"
            ),
            "interpretation": (
                "Ranks4-12 of the same existing tier1 intents supply the missing "
                "reservation event while all prior complete baselines remain "
                "unchanged; the baseline failure was shallow-depth exhaustion."
            ),
        }
    return {
        "baseline_depth_fallback_decision": (
            "DEPTH_FALLBACK_STILL_INCOMPLETE"
        ),
        "interpretation": (
            "Same-tier depth through rank12 is insufficient to complete all nine "
            "baselines; the next residual is tier1 role coverage/structural "
            "diversity rather than additional post-hoc depth."
        ),
    }


def evaluate_cross_paper_baseline_depth_fallback(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    generated: dict[str, Any],
    generated_intents_sha256: str,
    source_report: dict[str, Any],
    source_report_sha256: str,
    design: dict[str, Any],
    workdir: Any,
) -> dict[str, Any]:
    if (
        design["design_id"]
        != "local-extract-cross-paper-baseline-depth-fallback-v1"
    ):
        raise RuntimeError("unexpected frozen baseline fallback design")
    experiment = design["selected_discriminating_experiment"]
    if experiment["id"] != "TIER1_BASELINE_DEPTH_FALLBACK_DIAGNOSTIC_V1":
        raise RuntimeError("unexpected frozen baseline fallback experiment")

    frozen = experiment["frozen_inputs"]
    if int(frozen["paper_count"]) != PAPER_COUNT:
        raise RuntimeError("unexpected paper count")
    if int(frozen["core_budget"]) != CORE_BUDGET:
        raise RuntimeError("unexpected core budget")
    if int(frozen["total_budget"]) != TOTAL_BUDGET:
        raise RuntimeError("unexpected total budget")
    if int(frozen["primary_reservation_depth"]) != RESERVATION_DEPTH:
        raise RuntimeError("unexpected primary reservation depth")
    if int(frozen["fallback_max_depth"]) != RECOVERY_DEPTH:
        raise RuntimeError("unexpected fallback depth")
    if int(frozen["required_reservation_event_count"]) != RESERVATION_COUNT:
        raise RuntimeError("unexpected reservation event count")

    generated_identity_pass = (
        generated_intents_sha256
        == str(frozen["generated_intents_sha256"])
    )
    source_report_identity_pass = (
        source_report_sha256
        == str(frozen["source_cross_paper_report_sha256"])
    )
    generated_boundary_pass = bool(
        generated.get("scientific_state_materialized") is False
        and generated.get("production_routing_changed") is False
    )
    source_report_gate_pass = bool(
        source_report.get("cross_paper_recovery_decision")
        == "INSUFFICIENT_BASELINE_GENERALIZATION"
        and int(source_report.get("paper_count", -1)) == PAPER_COUNT
        and int(source_report.get("valid_paper_count", -1))
        == EXPECTED_ORIGINAL_COMPLETE_PAPERS
    )

    fixture_papers = [dict(row) for row in fixture["papers"]]
    if len(fixture_papers) != PAPER_COUNT:
        raise RuntimeError("fixture paper count mismatch")
    fixture_ids = {str(row["item_id"]) for row in fixture_papers}
    generated_ids = {
        str(row.get("item_id") or f"arxiv:{row.get('arxiv_id')}")
        for row in generated["papers"]
    }
    source_report_ids = {
        str(row["item_id"]) for row in source_report["papers"]
    }
    paper_set_identity_pass = bool(
        fixture_ids == generated_ids == source_report_ids
    )

    source_incomplete_ids = sorted(
        str(row["item_id"])
        for row in source_report["papers"]
        if not bool(row["baseline_complete"])
    )
    source_failure_shape_pass = source_incomplete_ids == [FAIL_ITEM_ID]

    per_intent_depth = int(wave7a3_contract["retrieval"]["per_intent_depth"])
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    if per_intent_depth != RECOVERY_DEPTH:
        raise RuntimeError("Wave7A.3 per-intent depth is not frozen top12")

    papers: list[dict[str, Any]] = []
    original_replay_pass = True
    complete_selected_identity_pass = True
    complete_event_identity_pass = True
    failing_prefix_identity_pass = True
    all_final_selected_locatable = True
    total_fallback_candidates = 0
    total_fallback_chars = 0

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
        original = _teacher_blind_baseline(
            propositions=propositions,
            generated_paper=generated_paper,
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
        )
        historical = _source_report_paper(source_report, item_id)

        current_replay_pass = bool(
            bool(original["complete"]) == bool(historical["baseline_complete"])
            and len(original["selected"])
            == int(historical["baseline_selected_count"])
            and len(original["events"])
            == int(historical["baseline_event_count"])
            and original["reason"] == historical["baseline_reason"]
            and _event_identity(list(original["events"]))
            == _event_identity(list(historical["baseline_reservation_events"]))
        )
        original_replay_pass = original_replay_pass and current_replay_pass

        result = _apply_depth_fallback(original)
        original_ids = [
            str(row["proposition_id"]) for row in original["selected"]
        ]
        final_ids = [
            str(row["proposition_id"]) for row in result["selected"]
        ]

        if bool(historical["baseline_complete"]):
            complete_selected_identity_pass = bool(
                complete_selected_identity_pass
                and final_ids == original_ids
            )
            complete_event_identity_pass = bool(
                complete_event_identity_pass
                and _event_identity(list(result["events"]))
                == _event_identity(list(original["events"]))
            )

        if item_id == FAIL_ITEM_ID:
            expected_prefix = _event_identity(
                list(historical["baseline_reservation_events"])
            )
            final_prefix = _event_identity(
                list(result["events"])[: len(expected_prefix)]
            )
            failing_prefix_identity_pass = bool(
                current_replay_pass
                and len(expected_prefix) == 3
                and final_prefix == expected_prefix
            )

        final_locatable = bool(
            len(final_ids) == len(set(final_ids))
            and all(pid in source for pid in final_ids)
        )
        all_final_selected_locatable = (
            all_final_selected_locatable and final_locatable
        )

        total_fallback_candidates += int(
            result["fallback_candidate_count_examined"]
        )
        total_fallback_chars += int(
            result["fallback_raw_source_chars_examined"]
        )

        papers.append({
            "item_id": item_id,
            "arxiv_id": str(paper["arxiv_id"]),
            "title": title,
            "acquisition": acquisition,
            "page_count": len(pages),
            "proposition_count": len(propositions),
            "source_report_replay_pass": current_replay_pass,
            "original_baseline_complete": bool(original["complete"]),
            "original_selected_count": len(original["selected"]),
            "original_event_count": len(original["events"]),
            "original_reason": original["reason"],
            "original_reservation_events": list(original["events"]),
            "fallback_applied": bool(result["fallback_applied"]),
            "fallback_events": list(result["fallback_events"]),
            "fallback_candidate_count_examined": int(
                result["fallback_candidate_count_examined"]
            ),
            "fallback_raw_source_chars_examined": int(
                result["fallback_raw_source_chars_examined"]
            ),
            "final_complete": bool(result["complete"]),
            "final_selected_count": len(result["selected"]),
            "final_event_count": len(result["events"]),
            "final_reservation_events": list(result["events"]),
            "final_selected_proposition_ids": final_ids,
            "final_selected_propositions_locatable": final_locatable,
        })

    final_complete_ids = [
        str(row["item_id"])
        for row in papers
        if bool(row["final_complete"])
    ]
    final_complete_count = len(final_complete_ids)
    identity_invariants_pass = bool(
        complete_selected_identity_pass
        and complete_event_identity_pass
        and failing_prefix_identity_pass
    )

    hard_safety = {
        "source_report_identity_pass": source_report_identity_pass,
        "generated_intents_identity_pass": generated_identity_pass,
        "generated_intents_boundary_pass": generated_boundary_pass,
        "source_report_expected_gate_pass": source_report_gate_pass,
        "paper_set_identity_pass": paper_set_identity_pass,
        "source_failure_shape_pass": source_failure_shape_pass,
        "original_baseline_replay_pass": original_replay_pass,
        "all_final_selected_propositions_locatable": (
            all_final_selected_locatable
        ),
        "eight_complete_selected_identity_pass": (
            complete_selected_identity_pass
        ),
        "eight_complete_event_identity_pass": complete_event_identity_pass,
        "failing_paper_original_three_event_prefix_identity_pass": (
            failing_prefix_identity_pass
        ),
        "teacher_used_for_selection": False,
        "teacher_used_for_baseline": False,
        "teacher_used_for_fallback": False,
        "recovery_outcomes_used_for_selection": False,
        "removed_propositions_used_for_selection": False,
        "historical_page_oracle_used_for_selection": False,
        "new_query_generation_false": True,
        "new_model_or_ranker_false": True,
        "pairwise_semantic_similarity_false": True,
        "threshold_calibration_false": True,
        "recovery_arm_changed_false": True,
        "production_routing_changed_false": True,
        "qwen_verify_changed_false": True,
        "wave7b_started_false": True,
        "semantic_control_or_ontology_output_false": True,
    }
    safe = bool(
        source_report_identity_pass
        and generated_identity_pass
        and generated_boundary_pass
        and source_report_gate_pass
        and paper_set_identity_pass
        and source_failure_shape_pass
        and original_replay_pass
        and all_final_selected_locatable
    )

    decision = _classify(
        safe=safe,
        identity_invariants_pass=identity_invariants_pass,
        final_complete_paper_count=final_complete_count,
    )

    failing = next(
        row for row in papers
        if str(row["item_id"]) == FAIL_ITEM_ID
    )
    fallback_event = (
        dict(failing["fallback_events"][0])
        if failing["fallback_events"]
        else None
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_CROSS_PAPER_BASELINE_DEPTH_FALLBACK_DIAGNOSTIC",
        "design_id": design["design_id"],
        "architecture_id": design["selected_architecture"]["id"],
        "experiment_id": experiment["id"],
        "generated_intents_sha256": generated_intents_sha256,
        "source_cross_paper_report_sha256": source_report_sha256,
        **decision,
        "paper_count": len(papers),
        "original_complete_paper_count": sum(
            int(row["original_baseline_complete"]) for row in papers
        ),
        "final_complete_paper_count": final_complete_count,
        "final_complete_paper_ids": final_complete_ids,
        "failing_paper": {
            "item_id": FAIL_ITEM_ID,
            "original_selected_count": int(
                failing["original_selected_count"]
            ),
            "original_event_count": int(failing["original_event_count"]),
            "final_selected_count": int(failing["final_selected_count"]),
            "final_event_count": int(failing["final_event_count"]),
            "fallback_event_added": fallback_event is not None,
            "fallback_event": fallback_event,
        },
        "invariance": {
            "eight_previously_complete_baselines_selected_identity_pass": (
                complete_selected_identity_pass
            ),
            "eight_previously_complete_baselines_event_identity_pass": (
                complete_event_identity_pass
            ),
            "failing_paper_original_three_event_prefix_identity_pass": (
                failing_prefix_identity_pass
            ),
        },
        "cost": {
            "fallback_candidate_count_examined": total_fallback_candidates,
            "fallback_raw_source_chars_examined": total_fallback_chars,
        },
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
