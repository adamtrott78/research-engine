from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from research_engine.local_extract_facet_reserved_evidence_planner import (
    _generic_support,
    _generated_teacher_paper,
    _locatable,
)
from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _prepare_teacher_pool,
    _rank,
)

DESIGN_ID = "local-extract-atomic-obligation-residual-closure-design-v1"
ARCHITECTURE_ID = "ATOMIC_OBLIGATION_COVERAGE_AND_COMPLETENESS_V1"
EXPERIMENT_ID = "ATOMIC_OBLIGATION_RESIDUAL_CLOSURE_DIAGNOSTIC_V1"
BASELINE_DECISION = "LOCAL_RESCUE_CONFIRMED"
TEACHER_ITEM_ID = "arxiv:2403.04036"
ATOMIC_SELECTION_ORIGIN = "atomic_obligation_reservation"
ATOMIC_APPEND_ORIGIN = "atomic_obligation_append_sensitivity"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_AUTOMATIC_GENERATOR_ORDER = (
    "DETERMINISTIC_SCIENCE_SCHEMA_V1",
    "QWEN35_SOURCE_AWARE_INTENTS_V1",
    "QWEN35_AUTHOR_ANCHORED_OBLIGATIONS_V1",
)


def _normalized_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", errors="ignore").decode("ascii")
    return _TOKEN_RE.findall(normalized.lower())


def _eligible_witness(
    candidate: dict[str, Any],
    witness_groups: list[list[str]],
) -> bool:
    tokens = _normalized_tokens(
        str(candidate.get("raw_source_form") or candidate.get("display_text") or "")
    )
    if not tokens:
        return False
    return all(
        any(
            token.startswith(str(prefix).lower())
            for token in tokens
            for prefix in group
        )
        for group in witness_groups
    )


def _rank_obligation(
    *,
    propositions: list[dict[str, Any]],
    obligation: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    ranked = _rank(
        propositions,
        _bm25_scores(str(obligation["bm25_query"]), propositions),
    )
    rows: list[dict[str, Any]] = []
    for row in ranked[:depth]:
        item = dict(row)
        item["bm25_rank"] = int(item.pop("oracle_rank"))
        item["bm25_score"] = float(item.pop("oracle_score"))
        item["eligible_witness"] = _eligible_witness(
            item,
            [list(group) for group in obligation["witness_groups"]],
        )
        rows.append(item)
    return rows


def _automatic_intent_rankings(
    *,
    propositions: list[dict[str, Any]],
    generated_teacher: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    arms = dict(generated_teacher["generator_arms"])
    for generator_id in _AUTOMATIC_GENERATOR_ORDER:
        arm = dict(arms[generator_id])
        for index, intent in enumerate(arm.get("accepted") or []):
            ranked = _rank(
                propositions,
                _bm25_scores(str(intent["question"]), propositions),
            )[:depth]
            rankings.append(
                {
                    "generator_id": generator_id,
                    "intent_index": index,
                    "question": str(intent["question"]),
                    "top12": [
                        {
                            "bm25_rank": int(row["oracle_rank"]),
                            "proposition_id": str(row["proposition_id"]),
                        }
                        for row in ranked
                    ],
                }
            )
    return rankings


def _structural_stats(
    rankings: list[dict[str, Any]],
) -> dict[str, dict[str, int | None]]:
    appearances: Counter[str] = Counter()
    best_rank: dict[str, int] = {}
    for intent in rankings:
        for row in intent["top12"]:
            pid = str(row["proposition_id"])
            rank = int(row["bm25_rank"])
            appearances[pid] += 1
            best_rank[pid] = min(best_rank.get(pid, rank), rank)
    ids = set(appearances) | set(best_rank)
    return {
        pid: {
            "appearance_count": int(appearances[pid]),
            "best_rank": int(best_rank[pid]) if pid in best_rank else None,
        }
        for pid in ids
    }


def _validate_starting_set(
    *,
    selected: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> None:
    if len(selected) != 28:
        raise RuntimeError("frozen starting selection is not budget28")
    ids = [str(row["proposition_id"]) for row in selected]
    if len(set(ids)) != len(ids):
        raise RuntimeError("frozen starting selection contains duplicate IDs")

    counts = Counter(str(row.get("selection_origin")) for row in selected)
    expected = {
        str(key): int(value)
        for key, value in selection_cfg[
            "expected_starting_selection_origin_counts"
        ].items()
    }
    if dict(counts) != expected:
        raise RuntimeError(
            f"starting selection_origin counts mismatch: {dict(counts)} != {expected}"
        )

    allowed = set(selection_cfg["protected_selection_origins"]) | set(
        selection_cfg["evictable_selection_origins"]
    )
    unknown = sorted(set(counts) - allowed)
    if unknown:
        raise RuntimeError(f"unknown starting selection_origin values: {unknown}")


def _eviction_candidate(
    *,
    selected: list[dict[str, Any]],
    structural_stats: dict[str, dict[str, int | None]],
    evictable_origins: set[str],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for position, row in enumerate(selected, start=1):
        if str(row.get("selection_origin")) not in evictable_origins:
            continue
        pid = str(row["proposition_id"])
        stats = dict(structural_stats.get(pid) or {})
        appearance_count = int(stats.get("appearance_count") or 0)
        best_rank_raw = stats.get("best_rank")
        best_rank = int(best_rank_raw) if best_rank_raw is not None else 13
        candidates.append(
            {
                "row": row,
                "position": position,
                "appearance_count": appearance_count,
                "best_rank": best_rank,
            }
        )
    if not candidates:
        return None

    primary = min(
        (
            int(item["appearance_count"]),
            -int(item["best_rank"]),
            -int(item["position"]),
        )
        for item in candidates
    )
    tied = [
        item
        for item in candidates
        if (
            int(item["appearance_count"]),
            -int(item["best_rank"]),
            -int(item["position"]),
        )
        == primary
    ]
    chosen = max(tied, key=lambda item: str(item["row"]["proposition_id"]))
    return {
        **dict(chosen["row"]),
        "structural_appearance_count": int(chosen["appearance_count"]),
        "structural_best_rank": (
            None if int(chosen["best_rank"]) == 13 else int(chosen["best_rank"])
        ),
        "starting_position": int(chosen["position"]),
    }


def _selected_witness(
    *,
    selected: list[dict[str, Any]],
    obligation: dict[str, Any],
) -> dict[str, Any] | None:
    witnesses = [
        row
        for row in selected
        if _eligible_witness(
            row,
            [list(group) for group in obligation["witness_groups"]],
        )
    ]
    if not witnesses:
        return None
    return min(
        witnesses,
        key=lambda row: str(row["proposition_id"]),
    )


def _runtime_primary_selection(
    *,
    starting_selected: list[dict[str, Any]],
    obligations: list[dict[str, Any]],
    obligation_rankings: dict[str, list[dict[str, Any]]],
    structural_stats: dict[str, dict[str, int | None]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    selected = [dict(row) for row in starting_selected]
    order = [str(value) for value in selection_cfg["obligation_order"]]
    by_id = {str(row["id"]): row for row in obligations}
    if set(order) != set(by_id):
        raise RuntimeError("frozen obligation order does not match obligations")

    evictable = {
        str(value) for value in selection_cfg["evictable_selection_origins"]
    }
    actions: list[dict[str, Any]] = []

    for obligation_id in order:
        obligation = dict(by_id[obligation_id])
        already = _selected_witness(selected=selected, obligation=obligation)
        if already is not None:
            actions.append(
                {
                    "obligation_id": obligation_id,
                    "action": "ALREADY_SATISFIED",
                    "selected_witness_proposition_id": str(
                        already["proposition_id"]
                    ),
                    "added_proposition_id": None,
                    "evicted_proposition_id": None,
                }
            )
            continue

        selected_ids = {str(row["proposition_id"]) for row in selected}
        eligible = [
            row
            for row in obligation_rankings[obligation_id]
            if bool(row["eligible_witness"])
            and str(row["proposition_id"]) not in selected_ids
        ]
        if not eligible:
            actions.append(
                {
                    "obligation_id": obligation_id,
                    "action": "NO_ELIGIBLE_TOP12_CANDIDATE",
                    "selected_witness_proposition_id": None,
                    "added_proposition_id": None,
                    "evicted_proposition_id": None,
                }
            )
            continue

        candidate = min(
            eligible,
            key=lambda row: (
                int(row["bm25_rank"]),
                str(row["proposition_id"]),
            ),
        )
        eviction = _eviction_candidate(
            selected=selected,
            structural_stats=structural_stats,
            evictable_origins=evictable,
        )
        if eviction is None:
            actions.append(
                {
                    "obligation_id": obligation_id,
                    "action": "NO_EVICTABLE_BASELINE_CANDIDATE",
                    "selected_witness_proposition_id": None,
                    "added_proposition_id": None,
                    "evicted_proposition_id": None,
                }
            )
            continue

        evicted_id = str(eviction["proposition_id"])
        selected = [
            row
            for row in selected
            if str(row["proposition_id"]) != evicted_id
        ]
        added = dict(candidate)
        added["selection_origin"] = ATOMIC_SELECTION_ORIGIN
        added["atomic_obligation_id"] = obligation_id
        selected.append(added)
        actions.append(
            {
                "obligation_id": obligation_id,
                "action": "SWAP",
                "selected_witness_proposition_id": str(
                    candidate["proposition_id"]
                ),
                "added_proposition_id": str(candidate["proposition_id"]),
                "added_bm25_rank": int(candidate["bm25_rank"]),
                "evicted_proposition_id": evicted_id,
                "evicted_selection_origin": str(
                    eviction["selection_origin"]
                ),
                "evicted_structural_appearance_count": int(
                    eviction["structural_appearance_count"]
                ),
                "evicted_structural_best_rank": eviction[
                    "structural_best_rank"
                ],
                "evicted_starting_position": int(
                    eviction["starting_position"]
                ),
            }
        )

    if len(selected) != 28 or len(
        {str(row["proposition_id"]) for row in selected}
    ) != 28:
        raise RuntimeError("primary runtime selection violated budget28/uniqueness")

    satisfaction: dict[str, Any] = {}
    for obligation in obligations:
        witness = _selected_witness(selected=selected, obligation=obligation)
        satisfaction[str(obligation["id"])] = {
            "satisfied": witness is not None,
            "witness_proposition_id": (
                str(witness["proposition_id"]) if witness is not None else None
            ),
        }

    complete = all(bool(row["satisfied"]) for row in satisfaction.values())
    return {
        "selected": selected,
        "actions": actions,
        "obligation_satisfaction": satisfaction,
        "core_obligations_satisfied": sum(
            int(bool(row["satisfied"])) for row in satisfaction.values()
        ),
        "controller_complete": complete,
        "fallback_required": not complete,
        "controller_action": (
            "LOCAL_COMPLETE"
            if complete
            else "LOCAL_INCOMPLETE_REQUIRE_FALLBACK"
        ),
    }


def _runtime_append_sensitivity(
    *,
    starting_selected: list[dict[str, Any]],
    obligations: list[dict[str, Any]],
    obligation_rankings: dict[str, list[dict[str, Any]]],
    selection_cfg: dict[str, Any],
    maximum_budget: int,
) -> dict[str, Any]:
    selected = [dict(row) for row in starting_selected]
    by_id = {str(row["id"]): row for row in obligations}
    actions: list[dict[str, Any]] = []

    for obligation_id in selection_cfg["obligation_order"]:
        obligation = dict(by_id[str(obligation_id)])
        witness = _selected_witness(selected=selected, obligation=obligation)
        if witness is not None:
            actions.append(
                {
                    "obligation_id": str(obligation_id),
                    "action": "ALREADY_SATISFIED",
                    "added_proposition_id": None,
                }
            )
            continue
        if len(selected) >= maximum_budget:
            actions.append(
                {
                    "obligation_id": str(obligation_id),
                    "action": "BUDGET_CEILING",
                    "added_proposition_id": None,
                }
            )
            continue
        selected_ids = {str(row["proposition_id"]) for row in selected}
        eligible = [
            row
            for row in obligation_rankings[str(obligation_id)]
            if bool(row["eligible_witness"])
            and str(row["proposition_id"]) not in selected_ids
        ]
        if not eligible:
            actions.append(
                {
                    "obligation_id": str(obligation_id),
                    "action": "NO_ELIGIBLE_TOP12_CANDIDATE",
                    "added_proposition_id": None,
                }
            )
            continue
        candidate = min(
            eligible,
            key=lambda row: (
                int(row["bm25_rank"]),
                str(row["proposition_id"]),
            ),
        )
        added = dict(candidate)
        added["selection_origin"] = ATOMIC_APPEND_ORIGIN
        added["atomic_obligation_id"] = str(obligation_id)
        selected.append(added)
        actions.append(
            {
                "obligation_id": str(obligation_id),
                "action": "APPEND",
                "added_proposition_id": str(candidate["proposition_id"]),
                "added_bm25_rank": int(candidate["bm25_rank"]),
            }
        )

    if len(selected) > maximum_budget:
        raise RuntimeError("append-only sensitivity exceeded frozen budget")
    return {
        "selected": selected,
        "actions": actions,
    }


def _source_fidelity_support(
    *,
    selected: list[dict[str, Any]],
    equivalence_sets: dict[str, list[str]],
) -> dict[str, Any]:
    selected_ids = {str(row["proposition_id"]) for row in selected}
    rows: dict[str, Any] = {}
    for atom_id, proposition_ids in equivalence_sets.items():
        hits = [
            str(pid)
            for pid in proposition_ids
            if str(pid) in selected_ids
        ]
        rows[str(atom_id)] = {
            "supported": bool(hits),
            "supporting_proposition_ids": hits,
        }
    supported = sorted(
        atom_id for atom_id, row in rows.items() if bool(row["supported"])
    )
    return {
        "supported_atoms": supported,
        "supported_count": len(supported),
        "atoms": rows,
    }


def _mechanism_outcome(
    *,
    safe: bool,
    preservation_regression: bool,
    primary_source_fidelity_count: int,
    sensitivity_source_fidelity_count: int,
) -> str:
    if not safe:
        return "INVALID_EXECUTION"
    if preservation_regression:
        return "PRESERVATION_REGRESSION"
    if primary_source_fidelity_count == 4:
        return "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED"
    if sensitivity_source_fidelity_count == 4:
        return "BUDGET_SELECTION_COLLISION"
    return "OBLIGATION_EXPOSURE_GAP"


def _completeness_outcome(
    *,
    safe: bool,
    controller_complete: bool,
    core_obligations_satisfied: int,
    source_fidelity_count: int,
) -> str:
    if not safe:
        return "INVALID_EXECUTION"
    if source_fidelity_count < 4 and controller_complete:
        return "UNSAFE_FALSE_COMPLETE"
    if (
        source_fidelity_count == 4
        and core_obligations_satisfied == 4
        and controller_complete
    ):
        return "COMPLETE_WITH_CLOSURE"
    if not controller_complete and (
        core_obligations_satisfied < 4 or source_fidelity_count < 4
    ):
        return "SAFE_INCOMPLETENESS_CONFIRMED"
    return "INVALID_EXECUTION"


def _acceptance_boundary_evidence(
    *,
    mechanism_outcome: str,
    completeness_outcome: str,
    preservation_regression: bool,
    frozen_controls_retained: int,
) -> str:
    if (
        mechanism_outcome == "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED"
        and completeness_outcome == "COMPLETE_WITH_CLOSURE"
        and not preservation_regression
    ):
        return "RESIDUAL_CLOSURE"
    if (
        completeness_outcome == "SAFE_INCOMPLETENESS_CONFIRMED"
        and frozen_controls_retained == 7
    ):
        return "SAFE_INCOMPLETENESS"
    return "NONE"


def evaluate_atomic_obligation_residual_closure(
    *,
    fixture: dict[str, Any],
    support_fixture: dict[str, Any],
    generated: dict[str, Any],
    generated_sha256: str,
    automatic_report_sha256: str,
    rescue_report: dict[str, Any],
    rescue_report_sha256: str,
    design: dict[str, Any],
    workdir: Any,
) -> dict[str, Any]:
    if design.get("design_id") != DESIGN_ID:
        raise RuntimeError("unexpected atomic-obligation design")
    if design.get("selected_architecture", {}).get("id") != ARCHITECTURE_ID:
        raise RuntimeError("unexpected atomic-obligation architecture")
    experiment = design["selected_discriminating_experiment"]
    if experiment.get("id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected atomic-obligation experiment")

    frozen = experiment["frozen_inputs"]
    selection_cfg = experiment["primary_budget28_selection"]
    depth = int(frozen["per_obligation_bm25_depth"])
    if int(frozen["exact_primary_budget"]) != 28:
        raise RuntimeError("unexpected primary budget")
    if int(frozen["diagnostic_append_only_budget"]) != 32:
        raise RuntimeError("unexpected append-only diagnostic budget")

    teacher_pool = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    propositions = [dict(row) for row in teacher_pool["page_pool"]]
    acquisition = dict(teacher_pool["acquisition"])
    generated_teacher = _generated_teacher_paper(generated)

    starting_selected = [
        dict(row) for row in rescue_report.get("selected_propositions") or []
    ]
    _validate_starting_set(
        selected=starting_selected,
        selection_cfg=selection_cfg,
    )

    atoms = [dict(row) for row in support_fixture["atoms"]]
    residual_ids = {str(value) for value in frozen["residual_atom_ids"]}
    prior_supported_ids = set(
        str(value)
        for value in rescue_report["rescued_teacher"]["supported_atoms"]
    )
    if len(prior_supported_ids) != int(
        frozen["expected_prior_supported_atom_count"]
    ):
        raise RuntimeError("unexpected prior supported atom count")

    automatic_rankings = _automatic_intent_rankings(
        propositions=propositions,
        generated_teacher=generated_teacher,
        depth=depth,
    )
    structural_stats = _structural_stats(automatic_rankings)

    obligations = [dict(row) for row in experiment["runtime_obligations"]]
    obligation_rankings = {
        str(obligation["id"]): _rank_obligation(
            propositions=propositions,
            obligation=obligation,
            depth=depth,
        )
        for obligation in obligations
    }

    # Runtime selection pass: no teacher atoms, expected pages, equivalence sets,
    # or posthoc support labels are supplied to these functions.
    primary = _runtime_primary_selection(
        starting_selected=starting_selected,
        obligations=obligations,
        obligation_rankings=obligation_rankings,
        structural_stats=structural_stats,
        selection_cfg=selection_cfg,
    )
    sensitivity = _runtime_append_sensitivity(
        starting_selected=starting_selected,
        obligations=obligations,
        obligation_rankings=obligation_rankings,
        selection_cfg=selection_cfg,
        maximum_budget=int(frozen["diagnostic_append_only_budget"]),
    )

    primary_selected = [dict(row) for row in primary["selected"]]
    sensitivity_selected = [dict(row) for row in sensitivity["selected"]]

    # Posthoc evaluation pass begins here.
    primary_legacy = _generic_support(atoms, primary_selected)
    sensitivity_legacy = _generic_support(atoms, sensitivity_selected)
    equivalence_sets = {
        str(atom_id): [str(pid) for pid in proposition_ids]
        for atom_id, proposition_ids in experiment[
            "posthoc_evaluation_only"
        ]["source_fidelity_equivalence_sets"].items()
    }
    primary_fidelity = _source_fidelity_support(
        selected=primary_selected,
        equivalence_sets=equivalence_sets,
    )
    sensitivity_fidelity = _source_fidelity_support(
        selected=sensitivity_selected,
        equivalence_sets=equivalence_sets,
    )

    primary_supported = set(primary_legacy["supported_atoms"])
    prior_retained = sorted(prior_supported_ids & primary_supported)
    frozen_controls = set(
        rescue_report["rescued_teacher"]["frozen_control_atoms_retained"]
    )
    controls_retained = sorted(frozen_controls & primary_supported)
    preservation_regression = bool(
        len(prior_retained)
        != int(frozen["expected_prior_supported_atom_count"])
        or len(controls_retained)
        != int(frozen["expected_frozen_control_count"])
    )

    legacy_residual_supported = sorted(primary_supported & residual_ids)
    undercount_atoms = sorted(
        set(primary_fidelity["supported_atoms"])
        - set(legacy_residual_supported)
    )

    source_identity = {
        "rescue_report_identity_pass": rescue_report_sha256
        == str(frozen["baseline_rescue_report_sha256"]),
        "generated_intents_identity_pass": generated_sha256
        == str(frozen["automatic_generated_intents_sha256"]),
        "automatic_report_identity_pass": automatic_report_sha256
        == str(frozen["automatic_intents_report_sha256"]),
        "teacher_item_identity_pass": str(teacher_pool["item_id"])
        == str(frozen["teacher_paper"]),
        "teacher_pdf_sha256_pass": str(acquisition.get("sha256"))
        == str(frozen["teacher_pdf_sha256"]),
        "teacher_page_count_pass": int(frozen["teacher_pdf_total_pages"])
        == int(
            next(
                row["total_pdf_pages"]
                for row in fixture["papers"]
                if str(row["item_id"]) == TEACHER_ITEM_ID
            )
        ),
        "baseline_decision_pass": rescue_report.get(
            "neighborhood_rescue_decision"
        )
        == BASELINE_DECISION,
        "baseline_selected_count_pass": len(starting_selected) == 28,
        "baseline_selected_locatable_pass": _locatable(
            starting_selected,
            propositions,
        ),
        "baseline_supported_atom_count_pass": int(
            rescue_report["rescued_teacher"]["supported_atom_count"]
        )
        == int(frozen["expected_prior_supported_atom_count"]),
        "baseline_frozen_control_count_pass": int(
            rescue_report["rescued_teacher"]["frozen_control_retention_count"]
        )
        == int(frozen["expected_frozen_control_count"]),
        "generated_boundary_pass": bool(
            generated.get("scientific_state_materialized") is False
            and generated.get("production_routing_changed") is False
            and generated.get("qwen_verify_changed") is False
            and generated.get("wave7b_started") is False
        ),
    }

    runtime_boundary = {
        "teacher_labels_used_for_runtime_selection": False,
        "source_fidelity_equivalence_sets_used_for_runtime_selection": False,
        "legacy_exact_anchors_used_for_runtime_selection": False,
        "expected_pages_used_for_runtime_selection": False,
        "new_model_or_ranker": False,
        "new_query_generation": False,
        "threshold_calibration": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
    }
    runtime_boundary_pass = not any(runtime_boundary.values())

    exact_budget_pass = bool(
        len(primary_selected) == 28
        and len({str(row["proposition_id"]) for row in primary_selected}) == 28
        and len(sensitivity_selected) <= 32
    )
    locatable_pass = bool(
        _locatable(primary_selected, propositions)
        and _locatable(sensitivity_selected, propositions)
    )

    safe = bool(
        all(source_identity.values())
        and runtime_boundary_pass
        and exact_budget_pass
        and locatable_pass
    )
    mechanism = _mechanism_outcome(
        safe=safe,
        preservation_regression=preservation_regression,
        primary_source_fidelity_count=int(primary_fidelity["supported_count"]),
        sensitivity_source_fidelity_count=int(
            sensitivity_fidelity["supported_count"]
        ),
    )
    completeness = _completeness_outcome(
        safe=safe,
        controller_complete=bool(primary["controller_complete"]),
        core_obligations_satisfied=int(primary["core_obligations_satisfied"]),
        source_fidelity_count=int(primary_fidelity["supported_count"]),
    )
    acceptance_evidence = _acceptance_boundary_evidence(
        mechanism_outcome=mechanism,
        completeness_outcome=completeness,
        preservation_regression=preservation_regression,
        frozen_controls_retained=len(controls_retained),
    )
    secondary = (
        "LEGACY_ANCHOR_UNDERCOUNT_CONFIRMED"
        if undercount_atoms
        else "NO_LEGACY_ANCHOR_UNDERCOUNT_OBSERVED"
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_ATOMIC_OBLIGATION_RESIDUAL_CLOSURE_DIAGNOSTIC",
        "design_id": DESIGN_ID,
        "architecture_id": ARCHITECTURE_ID,
        "experiment_id": EXPERIMENT_ID,
        "teacher_item_id": str(teacher_pool["item_id"]),
        "teacher_pdf_sha256": str(acquisition.get("sha256")),
        "generated_intents_sha256": generated_sha256,
        "automatic_report_sha256": automatic_report_sha256,
        "baseline_rescue_report_sha256": rescue_report_sha256,
        "mechanism_outcome": mechanism,
        "completeness_safety_outcome": completeness,
        "acceptance_boundary_evidence": acceptance_evidence,
        "secondary_evaluator_outcome": secondary,
        "primary_budget28": {
            "selected_count": len(primary_selected),
            "core_obligations_satisfied": int(
                primary["core_obligations_satisfied"]
            ),
            "controller_complete": bool(primary["controller_complete"]),
            "fallback_required": bool(primary["fallback_required"]),
            "controller_action": str(primary["controller_action"]),
            "runtime_obligation_satisfaction": dict(
                primary["obligation_satisfaction"]
            ),
            "actions": list(primary["actions"]),
            "legacy_exact_anchor_residual_supported_atoms": (
                legacy_residual_supported
            ),
            "legacy_exact_anchor_residual_support_count": len(
                legacy_residual_supported
            ),
            "source_fidelity_residual_support": primary_fidelity,
            "prior_supported_atom_retention_count": len(prior_retained),
            "prior_supported_atoms_retained": prior_retained,
            "frozen_control_retention_count": len(controls_retained),
            "frozen_controls_retained": controls_retained,
            "preservation_regression": preservation_regression,
            "selected_propositions": primary_selected,
        },
        "append_only_budget32_sensitivity": {
            "selected_count": len(sensitivity_selected),
            "actions": list(sensitivity["actions"]),
            "legacy_exact_anchor_support": sensitivity_legacy,
            "source_fidelity_residual_support": sensitivity_fidelity,
            "selected_propositions": sensitivity_selected,
        },
        "legacy_anchor_undercount": {
            "atoms": undercount_atoms,
            "source_fidelity_count": int(primary_fidelity["supported_count"]),
            "legacy_exact_anchor_count": len(legacy_residual_supported),
            "used_for_runtime_selection": False,
        },
        "obligation_rankings": {
            obligation_id: [
                {
                    "bm25_rank": int(row["bm25_rank"]),
                    "proposition_id": str(row["proposition_id"]),
                    "page_number": int(row["page_number"]),
                    "eligible_witness": bool(row["eligible_witness"]),
                    "raw_source_form": str(row["raw_source_form"]),
                }
                for row in rows
            ]
            for obligation_id, rows in obligation_rankings.items()
        },
        "structural_dispensability": {
            "automatic_intent_ranking_count": len(automatic_rankings),
            "depth": depth,
            "ranking_generator_ids": list(_AUTOMATIC_GENERATOR_ORDER),
            "used_teacher_labels": False,
        },
        "hard_safety": {
            **source_identity,
            "runtime_boundary_pass": runtime_boundary_pass,
            "exact_budget_pass": exact_budget_pass,
            "all_selected_propositions_exactly_locatable": locatable_pass,
            "teacher_labels_used_for_runtime_selection": False,
            "source_fidelity_equivalence_sets_used_for_runtime_selection": False,
            "legacy_exact_anchors_used_for_runtime_selection": False,
            "expected_pages_used_for_runtime_selection": False,
            "teacher_used_only_posthoc": True,
            "new_model_or_ranker_false": True,
            "new_query_generation_false": True,
            "threshold_calibration_false": True,
            "scientific_state_materialized_false": True,
            "production_routing_changed_false": True,
            "qwen_verify_changed_false": True,
            "wave7b_started_false": True,
            "semantic_control_or_ontology_output_false": True,
            "sealed_acceptance_benchmark_accessed_false": True,
            "acceptance_candidate_frozen_false": True,
        },
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
    }
