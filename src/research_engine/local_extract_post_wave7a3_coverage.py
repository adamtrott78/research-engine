from __future__ import annotations

from typing import Any

from research_engine.local_extract_oracle_query import _bm25_scores, _rank
from research_engine.local_extract_wave7_evaluate import (
    _prepare_papers,
    _rrf_merge,
    _teacher_leak_filter,
    _teacher_support,
)


CONTROL_ID = "QWEN35_SOURCE_AWARE_INTENTS_V1"
RESIDUAL_ID = "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1"
FACILITY_ID = "BALANCED_FACILITY_LOCATION_RRF_V1"
BRANCH_RRF_ID = "EQUAL_WEIGHT_BRANCH_RRF_V1"
QUOTA_ID = "CONTROL20_RESIDUAL4_QUOTA_V1"
CONTROL_RRF_ID = "FROZEN_SOURCE_AWARE_BM25_RRF_V1"
RESIDUAL_RRF_ID = "FROZEN_AUTHOR_ANCHORED_BM25_RRF_V1"

CONTROL_SUPPORTED_ATOMS = {
    "e2.a",
    "e2.b",
    "e3.a",
    "e3.b",
    "e4.a",
    "e4.b",
    "e6.b",
}
OBSERVED_RESIDUAL_ATOMS = {"e1.a", "e6.a"}


def _arm_rankings(
    *,
    paper: dict[str, Any],
    generator_id: str,
    forbidden: list[str],
    per_intent_depth: int,
) -> dict[str, Any]:
    arm = dict(paper["generator_arms"][generator_id])
    intents = [dict(row) for row in arm.get("accepted") or []]
    leak_rejected: list[dict[str, Any]] = []
    if paper.get("teacher_claims"):
        intents, leak_rejected = _teacher_leak_filter(
            intents,
            forbidden=forbidden,
        )

    rankings = []
    for intent_index, intent in enumerate(intents):
        question = str(intent["question"])
        ranking = _rank(
            paper["propositions"],
            _bm25_scores(question, paper["propositions"]),
        )[:per_intent_depth]
        rankings.append(
            {
                "intent_index": intent_index,
                "question": question,
                "ranking": ranking,
            }
        )

    return {
        "generator_id": generator_id,
        "raw_count": int(arm.get("raw_count") or 0),
        "guard_accepted_count": len(arm.get("accepted") or []),
        "post_leak_count": len(intents),
        "teacher_leak_rejected": leak_rejected,
        "rankings": rankings,
    }


def _plain_rankings(rows: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [list(item["ranking"]) for item in rows["rankings"]]


def _branch_rrf_merge(
    control: list[dict[str, Any]],
    residual: list[dict[str, Any]],
    *,
    rrf_k: int,
    budget: int,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    contributors: dict[str, list[dict[str, Any]]] = {}
    for branch, ranking in (("control", control), ("residual", residual)):
        for rank, row in enumerate(ranking, start=1):
            pid = str(row["proposition_id"])
            rows[pid] = dict(row)
            scores[pid] = scores.get(pid, 0.0) + 0.5 / (rrf_k + rank)
            contributors.setdefault(pid, []).append(
                {"branch": branch, "rank": rank}
            )
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))[:budget]
    merged = []
    for rank, pid in enumerate(ordered, start=1):
        row = dict(rows[pid])
        row["fusion_rank"] = rank
        row["fusion_score"] = float(scores[pid])
        row["fusion_contributors"] = contributors[pid]
        merged.append(row)
    return merged


def _quota_merge(
    control: list[dict[str, Any]],
    residual: list[dict[str, Any]],
    *,
    control_quota: int,
    residual_quota: int,
    budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rows: list[dict[str, Any]], limit: int) -> None:
        added = 0
        for source in rows:
            if added >= limit or len(selected) >= budget:
                break
            pid = str(source["proposition_id"])
            if pid in seen:
                continue
            seen.add(pid)
            selected.append(dict(source))
            added += 1

    add(control, control_quota)
    add(residual, residual_quota)
    if len(selected) < budget:
        add(control, budget)
    if len(selected) < budget:
        add(residual, budget)

    for rank, row in enumerate(selected, start=1):
        row["fusion_rank"] = rank
        row["fusion_score"] = float(1.0 / rank)
    return selected[:budget]


def _facility_location_merge(
    branches: dict[str, dict[str, Any]],
    *,
    rrf_k: int,
    budget: int,
) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    candidate_rows: dict[str, dict[str, Any]] = {}

    for branch in ("control", "residual"):
        rankings = list(branches[branch]["rankings"])
        if not rankings:
            continue
        intent_weight = 0.5 / len(rankings)
        for intent in rankings:
            scores: dict[str, float] = {}
            ranks: dict[str, int] = {}
            for rank, row in enumerate(intent["ranking"], start=1):
                pid = str(row["proposition_id"])
                candidate_rows[pid] = dict(row)
                scores[pid] = float(1.0 / (rrf_k + rank))
                ranks[pid] = rank
            terms.append(
                {
                    "branch": branch,
                    "intent_index": int(intent["intent_index"]),
                    "weight": float(intent_weight),
                    "scores": scores,
                    "ranks": ranks,
                }
            )

    covered = [0.0 for _ in terms]
    remaining = set(candidate_rows)
    selected: list[dict[str, Any]] = []
    objective = 0.0

    while remaining and len(selected) < budget:
        best_pid: str | None = None
        best_gain = -1.0
        for pid in sorted(remaining):
            gain = 0.0
            for index, term in enumerate(terms):
                score = float(term["scores"].get(pid, 0.0))
                if score > covered[index]:
                    gain += float(term["weight"]) * (score - covered[index])
            if gain > best_gain + 1e-15:
                best_pid = pid
                best_gain = gain

        assert best_pid is not None
        contributors = []
        for index, term in enumerate(terms):
            score = float(term["scores"].get(best_pid, 0.0))
            if score > 0.0:
                contributors.append(
                    {
                        "branch": term["branch"],
                        "intent_index": int(term["intent_index"]),
                        "rank": int(term["ranks"][best_pid]),
                        "weighted_relevance": float(term["weight"]) * score,
                    }
                )
            if score > covered[index]:
                covered[index] = score

        objective += best_gain
        row = dict(candidate_rows[best_pid])
        row["fusion_rank"] = len(selected) + 1
        row["fusion_score"] = float(best_gain)
        row["coverage_objective_after"] = float(objective)
        row["fusion_contributors"] = contributors
        selected.append(row)
        remaining.remove(best_pid)

    return selected


def _arm_summary(
    *,
    papers: list[dict[str, Any]],
    selected_by_paper: dict[str, list[dict[str, Any]]],
    atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    total_expected_pages = 0
    total_coarse_hits = 0
    total_merged_hits = 0
    teacher = None
    paper_rows = []

    for paper in papers:
        item_id = str(paper["item_id"])
        merged = selected_by_paper[item_id]
        expected = {int(value) for value in paper["historical_evidence_pages"]}
        selected_pages = {int(value) for value in paper["selected_page_numbers"]}
        merged_pages = {int(row["page_number"]) for row in merged}

        total_expected_pages += len(expected)
        total_coarse_hits += len(expected & selected_pages)
        total_merged_hits += len(expected & merged_pages)

        row = {
            "item_id": item_id,
            "merged_proposition_count": len(merged),
            "merged_page_numbers": sorted(merged_pages),
            "historical_page_hits": len(expected & merged_pages),
            "historical_page_count": len(expected),
            "merged_propositions": [
                {
                    "proposition_id": str(value["proposition_id"]),
                    "page_number": int(value["page_number"]),
                    "fusion_rank": int(value["fusion_rank"]),
                    "fusion_score": float(value["fusion_score"]),
                    "raw_source_form": str(value["raw_source_form"]),
                }
                for value in merged
            ],
        }
        if paper.get("teacher_claims"):
            teacher = _teacher_support(atoms, merged)
            row["teacher_support"] = teacher
        paper_rows.append(row)

    coarse_ceiling = (
        total_coarse_hits / total_expected_pages if total_expected_pages else 1.0
    )
    page_recall = (
        total_merged_hits / total_expected_pages if total_expected_pages else 1.0
    )
    fraction = page_recall / coarse_ceiling if coarse_ceiling else 1.0

    teacher = teacher or {
        "supported_atoms": [],
        "supported_atoms_at_24": 0,
        "support_recall_at_24": 0.0,
        "known_miss_recovered": [],
        "known_miss_recovery_count": 0,
        "mean_best_token_f1_at_24": 0.0,
        "atoms": [],
    }
    supported = set(teacher["supported_atoms"])
    teacher = {
        **teacher,
        "observed_residual_atoms_recovered": sorted(
            supported & OBSERVED_RESIDUAL_ATOMS
        ),
        "observed_residual_recovery_count": len(
            supported & OBSERVED_RESIDUAL_ATOMS
        ),
        "frozen_control_atoms_retained": sorted(
            supported & CONTROL_SUPPORTED_ATOMS
        ),
        "frozen_control_retention_count": len(
            supported & CONTROL_SUPPORTED_ATOMS
        ),
    }

    return {
        "teacher": teacher,
        "nine_paper": {
            "historical_evidence_bearing_page_count": total_expected_pages,
            "coarse_selected_page_hits": total_coarse_hits,
            "coarse_selected_page_recall_ceiling": coarse_ceiling,
            "merged24_page_hits": total_merged_hits,
            "historical_evidence_bearing_page_recall_in_merged24": page_recall,
            "merged24_page_recall_fraction_of_ceiling": fraction,
        },
        "papers": paper_rows,
    }


def _composition_verdict(candidate: dict[str, Any]) -> dict[str, Any]:
    teacher = candidate["teacher"]
    nine = candidate["nine_paper"]
    support = int(teacher["supported_atoms_at_24"])
    residual = int(teacher["observed_residual_recovery_count"])
    retained = int(teacher["frozen_control_retention_count"])
    page_fraction = float(nine["merged24_page_recall_fraction_of_ceiling"])

    if (
        support >= 9
        and residual == 2
        and retained == 7
        and page_fraction >= 0.90
    ):
        return {
            "composition_decision": "COMPOSITION_CONFIRMED",
            "rationale": (
                "The frozen teacher-blind coverage selector preserves all seven "
                "source-aware control atoms while adding both observed residual "
                "atoms within the unchanged top-24 budget."
            ),
        }
    if support >= 8 and support > 7 and residual >= 1 and retained >= 6:
        return {
            "composition_decision": "COMPOSITION_PARTIAL",
            "rationale": (
                "The selector improves total support and preserves most frozen "
                "control coverage, but does not meet the complete composition gate."
            ),
        }
    return {
        "composition_decision": "COMPOSITION_NEGATIVE",
        "rationale": (
            "The frozen selector does not establish coverage-preserving "
            "complementarity under the unchanged top-24 budget."
        ),
    }


def evaluate_post_wave7a3_coverage(
    *,
    fixture: dict[str, Any],
    wave7a3_contract: dict[str, Any],
    support_fixture: dict[str, Any],
    generated: dict[str, Any],
    design: dict[str, Any],
    generated_intents_sha256: str,
    workdir: Any,
) -> dict[str, Any]:
    if generated["scientific_state_materialized"] or generated["production_routing_changed"]:
        raise RuntimeError("generated-intents artifact violates scientific boundary")

    selected = design["selected_first_diagnostic"]
    retrieval = selected["retrieval"]
    selector = selected["selector"]
    if selector["id"] != FACILITY_ID:
        raise RuntimeError("unexpected frozen selector")
    if retrieval["retriever"] != "BM25_DIRECT_ONLY_FOR_FIRST_DIAGNOSTIC":
        raise RuntimeError("unexpected frozen retriever")

    per_intent_depth = int(retrieval["per_intent_depth"])
    budget = int(retrieval["final_global_proposition_budget"])
    rrf_k = int(wave7a3_contract["retrieval"]["fusion"]["k"])
    forbidden = list(
        wave7a3_contract["teacher_anti_leak_audit"][
            "forbidden_normalized_substrings"
        ]
    )

    papers = _prepare_papers(
        fixture=fixture,
        generated=generated,
        workdir=workdir,
    )
    atoms = [dict(row) for row in support_fixture["atoms"]]

    selected_by_arm = {
        FACILITY_ID: {},
        BRANCH_RRF_ID: {},
        QUOTA_ID: {},
        CONTROL_RRF_ID: {},
        RESIDUAL_RRF_ID: {},
    }
    paper_diagnostics = []
    total_leak_rejected = 0

    for paper in papers:
        control = _arm_rankings(
            paper=paper,
            generator_id=CONTROL_ID,
            forbidden=forbidden,
            per_intent_depth=per_intent_depth,
        )
        residual = _arm_rankings(
            paper=paper,
            generator_id=RESIDUAL_ID,
            forbidden=forbidden,
            per_intent_depth=per_intent_depth,
        )
        total_leak_rejected += len(control["teacher_leak_rejected"])
        total_leak_rejected += len(residual["teacher_leak_rejected"])

        control_rrf = _rrf_merge(
            _plain_rankings(control),
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=budget,
        )
        residual_rrf = _rrf_merge(
            _plain_rankings(residual),
            rrf_k=rrf_k,
            per_intent_depth=per_intent_depth,
            budget=budget,
        )
        facility = _facility_location_merge(
            {"control": control, "residual": residual},
            rrf_k=rrf_k,
            budget=budget,
        )
        branch_rrf = _branch_rrf_merge(
            control_rrf,
            residual_rrf,
            rrf_k=rrf_k,
            budget=budget,
        )
        quota = _quota_merge(
            control_rrf,
            residual_rrf,
            control_quota=20,
            residual_quota=4,
            budget=budget,
        )

        item_id = str(paper["item_id"])
        selected_by_arm[FACILITY_ID][item_id] = facility
        selected_by_arm[BRANCH_RRF_ID][item_id] = branch_rrf
        selected_by_arm[QUOTA_ID][item_id] = quota
        selected_by_arm[CONTROL_RRF_ID][item_id] = control_rrf
        selected_by_arm[RESIDUAL_RRF_ID][item_id] = residual_rrf
        paper_diagnostics.append(
            {
                "item_id": item_id,
                "control_post_leak_intents": control["post_leak_count"],
                "residual_post_leak_intents": residual["post_leak_count"],
                "control_teacher_leak_rejected": control["teacher_leak_rejected"],
                "residual_teacher_leak_rejected": residual["teacher_leak_rejected"],
                "facility_candidate_pool_count": len(
                    {
                        str(row["proposition_id"])
                        for branch in (control, residual)
                        for intent in branch["rankings"]
                        for row in intent["ranking"]
                    }
                ),
            }
        )

    arms = {
        arm_id: _arm_summary(
            papers=papers,
            selected_by_paper=selected_rows,
            atoms=atoms,
        )
        for arm_id, selected_rows in selected_by_arm.items()
    }
    decision = _composition_verdict(arms[FACILITY_ID])

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_POST_WAVE7A3_COVERAGE_DIAGNOSTIC",
        "design_id": design["design_id"],
        "baseline_sha": design["baseline_sha"],
        "generated_intents_sha256": generated_intents_sha256,
        "candidate_selector_id": FACILITY_ID,
        **decision,
        "arms": arms,
        "paper_diagnostics": paper_diagnostics,
        "teacher_leak_reject_count": total_leak_rejected,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "hard_safety": {
            "teacher_blind_selector": True,
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "ontology_or_semantic_control_in_extract_false": True,
        },
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "ontology_or_semantic_control_in_extract": False,
    }
