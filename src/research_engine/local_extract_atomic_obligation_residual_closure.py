from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from research_engine.local_extract_oracle_query import (
    _bm25_scores,
    _prepare_teacher_pool,
    _rank,
)

DESIGN_ID = "local-extract-atomic-obligation-residual-closure-design-v1"
ARCHITECTURE_ID = "ATOMIC_OBLIGATION_COVERAGE_AND_COMPLETENESS_V1"
EXPERIMENT_ID = "ATOMIC_OBLIGATION_RESIDUAL_CLOSURE_DIAGNOSTIC_V1"
COMPLETENESS_ID = "ATOMIC_CORE_OBLIGATION_COMPLETENESS_V1"
FIXED_BUDGET = 28
DIAGNOSTIC_BUDGET = 32
CONTROL_IDS = ("e2.a", "e2.b", "e3.a", "e3.b", "e4.a", "e4.b", "e6.b")
RESIDUAL_IDS = ("e1.b", "e1.c", "e3.c", "e6.c")


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", ascii_text.lower())


def _eligible_for_obligation(
    row: dict[str, Any],
    obligation: dict[str, Any],
) -> bool:
    tokens = _tokens(str(row.get("raw_source_form") or ""))
    if not tokens:
        return False
    for group in obligation["witness_groups"]:
        prefixes = [str(value).lower() for value in group]
        if not any(
            token.startswith(prefix)
            for token in tokens
            for prefix in prefixes
        ):
            return False
    return True


def _obligation_witnesses(
    selected: list[dict[str, Any]],
    obligations: list[dict[str, Any]],
) -> dict[str, list[str]]:
    return {
        str(obligation["id"]): [
            str(row["proposition_id"])
            for row in selected
            if _eligible_for_obligation(row, obligation)
        ]
        for obligation in obligations
    }


def _normalize_anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _legacy_support(
    atoms: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    details = []
    supported = []
    for atom in atoms:
        page = int(atom["page_number"])
        candidates = [
            row
            for row in selected
            if int(row["page_number"]) == page
        ]
        anchor_hits = []
        for anchor in atom["required_anchors"]:
            target = str(anchor)
            hit = any(
                target
                in _normalize_anchor(str(row.get("raw_source_form") or ""))
                for row in candidates
            )
            anchor_hits.append(bool(hit))
        ok = all(anchor_hits)
        atom_id = str(atom["id"])
        if ok:
            supported.append(atom_id)
        details.append(
            {
                "id": atom_id,
                "supported": ok,
                "anchor_hits": anchor_hits,
            }
        )
    return {
        "supported_atoms": supported,
        "supported_atom_count": len(supported),
        "atom_details": details,
    }


def _source_fidelity_support(
    selected: list[dict[str, Any]],
    equivalence_sets: dict[str, list[str]],
) -> dict[str, Any]:
    selected_ids = {str(row["proposition_id"]) for row in selected}
    details = {}
    supported = []
    for atom_id in RESIDUAL_IDS:
        allowed = [str(value) for value in equivalence_sets[atom_id]]
        hits = [pid for pid in allowed if pid in selected_ids]
        details[atom_id] = {
            "supported": bool(hits),
            "supporting_proposition_ids": hits,
            "predeclared_equivalence_set": allowed,
        }
        if hits:
            supported.append(atom_id)
    return {
        "supported_atoms": supported,
        "supported_count": len(supported),
        "details": details,
    }


def _generated_teacher(generated: dict[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in generated["papers"]
        if str(row.get("item_id") or "") == "arxiv:2403.04036"
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one generated teacher paper, got {len(rows)}"
        )
    return rows[0]


def _automatic_intent_rankings(
    *,
    propositions: list[dict[str, Any]],
    generated: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    teacher = _generated_teacher(generated)
    rows = []
    for generator_id in sorted(teacher["generator_arms"]):
        arm = dict(teacher["generator_arms"][generator_id])
        accepted = [dict(value) for value in arm.get("accepted") or []]
        for index, intent in enumerate(accepted):
            question = str(intent["question"])
            ranking = _rank(
                propositions,
                _bm25_scores(question, propositions),
            )[:depth]
            rows.append(
                {
                    "generator_id": generator_id,
                    "intent_index": index,
                    "question": question,
                    "ranking": [
                        {
                            "rank": int(value["oracle_rank"]),
                            "proposition_id": str(value["proposition_id"]),
                        }
                        for value in ranking
                    ],
                }
            )
    return rows


def _dispensability(
    *,
    baseline: list[dict[str, Any]],
    automatic_rankings: list[dict[str, Any]],
    depth: int,
) -> dict[str, dict[str, Any]]:
    result = {}
    for position, row in enumerate(baseline, start=1):
        pid = str(row["proposition_id"])
        hits = []
        for ranking in automatic_rankings:
            match = next(
                (
                    item
                    for item in ranking["ranking"]
                    if str(item["proposition_id"]) == pid
                ),
                None,
            )
            if match is not None:
                hits.append(int(match["rank"]))
        result[pid] = {
            "automatic_top12_appearance_count": len(hits),
            "best_automatic_rank": min(hits) if hits else depth + 1,
            "original_selected_position": position,
        }
    return result


def _choose_eviction(
    *,
    selected: list[dict[str, Any]],
    dispensability: dict[str, dict[str, Any]],
    evictable_origins: set[str],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in selected
        if str(row.get("selection_origin") or "") in evictable_origins
        and str(row["proposition_id"]) in dispensability
    ]
    if not candidates:
        return None

    minimum_appearance = min(
        int(dispensability[str(row["proposition_id"])][
            "automatic_top12_appearance_count"
        ])
        for row in candidates
    )
    candidates = [
        row
        for row in candidates
        if int(dispensability[str(row["proposition_id"])][
            "automatic_top12_appearance_count"
        ])
        == minimum_appearance
    ]

    worst_best_rank = max(
        int(dispensability[str(row["proposition_id"])]["best_automatic_rank"])
        for row in candidates
    )
    candidates = [
        row
        for row in candidates
        if int(dispensability[str(row["proposition_id"])]["best_automatic_rank"])
        == worst_best_rank
    ]

    latest_position = max(
        int(dispensability[str(row["proposition_id"])][
            "original_selected_position"
        ])
        for row in candidates
    )
    candidates = [
        row
        for row in candidates
        if int(dispensability[str(row["proposition_id"])][
            "original_selected_position"
        ])
        == latest_position
    ]

    chosen = max(candidates, key=lambda row: str(row["proposition_id"]))
    pid = str(chosen["proposition_id"])
    return {
        "row": dict(chosen),
        "metrics": dict(dispensability[pid]),
    }


def _obligation_rankings(
    *,
    propositions: list[dict[str, Any]],
    obligations: list[dict[str, Any]],
    depth: int,
) -> dict[str, dict[str, Any]]:
    result = {}
    for obligation in obligations:
        obligation_id = str(obligation["id"])
        ranking = _rank(
            propositions,
            _bm25_scores(str(obligation["bm25_query"]), propositions),
        )[:depth]
        candidates = []
        for row in ranking:
            candidates.append(
                {
                    "rank": int(row["oracle_rank"]),
                    "score": float(row["oracle_score"]),
                    "proposition_id": str(row["proposition_id"]),
                    "page_number": int(row["page_number"]),
                    "raw_source_form": str(row["raw_source_form"]),
                    "eligible": _eligible_for_obligation(row, obligation),
                }
            )
        result[obligation_id] = {
            "question": str(obligation["generic_information_need"]),
            "bm25_query": str(obligation["bm25_query"]),
            "top12": candidates,
        }
    return result


def _highest_ranked_missing_witness(
    *,
    selected_ids: set[str],
    ranking: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in ranking["top12"]:
        if not bool(candidate["eligible"]):
            continue
        if str(candidate["proposition_id"]) in selected_ids:
            continue
        return dict(candidate)
    return None


def _primary_budget_selection(
    *,
    baseline: list[dict[str, Any]],
    propositions_by_id: dict[str, dict[str, Any]],
    obligations: list[dict[str, Any]],
    rankings: dict[str, dict[str, Any]],
    dispensability: dict[str, dict[str, Any]],
    evictable_origins: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [dict(row) for row in baseline]
    events = []
    for obligation in obligations:
        obligation_id = str(obligation["id"])
        existing = [
            str(row["proposition_id"])
            for row in selected
            if _eligible_for_obligation(row, obligation)
        ]
        if existing:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "ALREADY_SATISFIED",
                    "selected_witnesses": existing,
                }
            )
            continue

        selected_ids = {str(row["proposition_id"]) for row in selected}
        candidate = _highest_ranked_missing_witness(
            selected_ids=selected_ids,
            ranking=rankings[obligation_id],
        )
        if candidate is None:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "NO_ELIGIBLE_TOP12_CANDIDATE",
                }
            )
            continue

        eviction = _choose_eviction(
            selected=selected,
            dispensability=dispensability,
            evictable_origins=evictable_origins,
        )
        if eviction is None:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "NO_EVICTABLE_SLOT",
                    "candidate": candidate,
                }
            )
            continue

        removed = dict(eviction["row"])
        removed_pid = str(removed["proposition_id"])
        selected = [
            row
            for row in selected
            if str(row["proposition_id"]) != removed_pid
        ]
        source = dict(propositions_by_id[str(candidate["proposition_id"])])
        source["selection_origin"] = "atomic_obligation_residual_closure"
        source["atomic_obligation_id"] = obligation_id
        source["atomic_obligation_bm25_rank"] = int(candidate["rank"])
        selected.append(source)
        if len(selected) != FIXED_BUDGET:
            raise RuntimeError("primary selection violated fixed budget")
        if len({str(row["proposition_id"]) for row in selected}) != len(selected):
            raise RuntimeError("primary selection introduced duplicate proposition")
        events.append(
            {
                "obligation_id": obligation_id,
                "action": "SWAP",
                "added_proposition_id": str(candidate["proposition_id"]),
                "added_bm25_rank": int(candidate["rank"]),
                "removed_proposition_id": removed_pid,
                "removed_selection_origin": str(removed["selection_origin"]),
                "removed_dispensability": dict(eviction["metrics"]),
            }
        )
    return selected, events


def _append_only_sensitivity(
    *,
    baseline: list[dict[str, Any]],
    propositions_by_id: dict[str, dict[str, Any]],
    obligations: list[dict[str, Any]],
    rankings: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [dict(row) for row in baseline]
    events = []
    for obligation in obligations:
        obligation_id = str(obligation["id"])
        existing = [
            str(row["proposition_id"])
            for row in selected
            if _eligible_for_obligation(row, obligation)
        ]
        if existing:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "ALREADY_SATISFIED",
                    "selected_witnesses": existing,
                }
            )
            continue
        if len(selected) >= DIAGNOSTIC_BUDGET:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "DIAGNOSTIC_BUDGET_EXHAUSTED",
                }
            )
            continue
        candidate = _highest_ranked_missing_witness(
            selected_ids={str(row["proposition_id"]) for row in selected},
            ranking=rankings[obligation_id],
        )
        if candidate is None:
            events.append(
                {
                    "obligation_id": obligation_id,
                    "action": "NO_ELIGIBLE_TOP12_CANDIDATE",
                }
            )
            continue
        source = dict(propositions_by_id[str(candidate["proposition_id"])])
        source["selection_origin"] = "atomic_obligation_append_only_sensitivity"
        source["atomic_obligation_id"] = obligation_id
        source["atomic_obligation_bm25_rank"] = int(candidate["rank"])
        selected.append(source)
        events.append(
            {
                "obligation_id": obligation_id,
                "action": "APPEND",
                "added_proposition_id": str(candidate["proposition_id"]),
                "added_bm25_rank": int(candidate["rank"]),
            }
        )
    if len(selected) > DIAGNOSTIC_BUDGET:
        raise RuntimeError("append-only sensitivity exceeded diagnostic budget")
    return selected, events


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
    runtime_complete: bool,
    primary_source_fidelity_count: int,
) -> str:
    if not safe:
        return "INVALID_EXECUTION"
    if runtime_complete and primary_source_fidelity_count < 4:
        return "UNSAFE_FALSE_COMPLETE"
    if runtime_complete and primary_source_fidelity_count == 4:
        return "COMPLETE_WITH_CLOSURE"
    return "SAFE_INCOMPLETENESS_CONFIRMED"


def _acceptance_boundary_evidence(
    *,
    mechanism_outcome: str,
    completeness_outcome: str,
    prior_retention_count: int,
    control_retention_count: int,
) -> str:
    if (
        mechanism_outcome == "ATOMIC_RESIDUAL_CLOSURE_CONFIRMED"
        and completeness_outcome == "COMPLETE_WITH_CLOSURE"
        and prior_retention_count == 9
        and control_retention_count == 7
    ):
        return "RESIDUAL_CLOSURE"
    if (
        completeness_outcome == "SAFE_INCOMPLETENESS_CONFIRMED"
        and prior_retention_count == 9
        and control_retention_count == 7
    ):
        return "SAFE_INCOMPLETENESS"
    return "NONE"


def evaluate_atomic_obligation_residual_closure(
    *,
    fixture: dict[str, Any],
    fixture_git_blob: str,
    support_fixture: dict[str, Any],
    support_fixture_git_blob: str,
    design: dict[str, Any],
    design_git_blob: str,
    baseline_report: dict[str, Any],
    baseline_report_sha256: str,
    generated: dict[str, Any],
    generated_sha256: str,
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

    obligations = [dict(row) for row in experiment["runtime_obligations"]]
    if len(obligations) != 4 or not all(bool(row["core"]) for row in obligations):
        raise RuntimeError("expected exactly four frozen CORE obligations")
    if int(frozen["exact_primary_budget"]) != FIXED_BUDGET:
        raise RuntimeError("unexpected primary budget")
    if int(frozen["diagnostic_append_only_budget"]) != DIAGNOSTIC_BUDGET:
        raise RuntimeError("unexpected diagnostic sensitivity budget")

    expected_design_blob = "90bfc59edd2e91f03de3a1ec02fe51c4dd1085b9"
    identity = {
        "design_blob_pass": design_git_blob == expected_design_blob,
        "corpus_fixture_blob_pass": fixture_git_blob
        == str(frozen["corpus_fixture_blob"]),
        "support_fixture_blob_pass": support_fixture_git_blob
        == str(frozen["teacher_oracle_blob"]),
        "baseline_report_sha256_pass": baseline_report_sha256
        == str(frozen["baseline_rescue_report_sha256"]),
        "generated_intents_sha256_pass": generated_sha256
        == str(frozen["automatic_generated_intents_sha256"]),
    }

    if baseline_report.get("neighborhood_rescue_decision") != "LOCAL_RESCUE_CONFIRMED":
        raise RuntimeError("unexpected baseline rescue decision")
    baseline = [dict(row) for row in baseline_report["selected_propositions"]]
    if len(baseline) != FIXED_BUDGET:
        raise RuntimeError("baseline rescue report is not budget28")
    if len({str(row["proposition_id"]) for row in baseline}) != FIXED_BUDGET:
        raise RuntimeError("baseline rescue report has duplicate propositions")

    expected_origin_counts = {
        str(key): int(value)
        for key, value in experiment["primary_budget28_selection"][
            "expected_starting_selection_origin_counts"
        ].items()
    }
    observed_origin_counts: dict[str, int] = {}
    for row in baseline:
        origin = str(row.get("selection_origin") or "")
        observed_origin_counts[origin] = observed_origin_counts.get(origin, 0) + 1
    origin_counts_pass = observed_origin_counts == expected_origin_counts

    teacher_pool = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    if str(teacher_pool["item_id"]) != str(frozen["teacher_paper"]):
        raise RuntimeError("unexpected teacher item")
    acquisition = dict(teacher_pool["acquisition"])
    acquisition_sha = str(
        acquisition.get("sha256")
        or acquisition.get("source_sha256")
        or acquisition.get("expected_sha256")
        or ""
    )
    teacher_pdf_identity_pass = (
        str(fixture["papers"][
            next(
                i
                for i, row in enumerate(fixture["papers"])
                if str(row.get("item_id") or "") == str(frozen["teacher_paper"])
            )
        ]["sha256"])
        == str(frozen["teacher_pdf_sha256"])
    )
    if acquisition_sha:
        teacher_pdf_identity_pass = (
            teacher_pdf_identity_pass
            and acquisition_sha == str(frozen["teacher_pdf_sha256"])
        )

    propositions = [dict(row) for row in teacher_pool["page_pool"]]
    propositions_by_id = {
        str(row["proposition_id"]): dict(row)
        for row in propositions
    }
    all_baseline_locatable = all(
        str(row["proposition_id"]) in propositions_by_id
        for row in baseline
    )

    automatic_rankings = _automatic_intent_rankings(
        propositions=propositions,
        generated=generated,
        depth=int(frozen["per_obligation_bm25_depth"]),
    )
    dispensability = _dispensability(
        baseline=baseline,
        automatic_rankings=automatic_rankings,
        depth=int(frozen["per_obligation_bm25_depth"]),
    )
    rankings = _obligation_rankings(
        propositions=propositions,
        obligations=obligations,
        depth=int(frozen["per_obligation_bm25_depth"]),
    )

    selection_cfg = experiment["primary_budget28_selection"]
    protected_origins = set(selection_cfg["protected_selection_origins"])
    evictable_origins = set(selection_cfg["evictable_selection_origins"])
    all_origins = set(observed_origin_counts)
    origin_policy_pass = bool(
        protected_origins.isdisjoint(evictable_origins)
        and all_origins <= protected_origins | evictable_origins
    )

    primary, primary_events = _primary_budget_selection(
        baseline=baseline,
        propositions_by_id=propositions_by_id,
        obligations=obligations,
        rankings=rankings,
        dispensability=dispensability,
        evictable_origins=evictable_origins,
    )
    sensitivity, sensitivity_events = _append_only_sensitivity(
        baseline=baseline,
        propositions_by_id=propositions_by_id,
        obligations=obligations,
        rankings=rankings,
    )

    atoms = [dict(row) for row in support_fixture["atoms"]]
    baseline_legacy = _legacy_support(atoms, baseline)
    primary_legacy = _legacy_support(atoms, primary)
    sensitivity_legacy = _legacy_support(atoms, sensitivity)

    baseline_supported = set(baseline_legacy["supported_atoms"])
    expected_prior = int(frozen["expected_prior_supported_atom_count"])
    expected_controls = int(frozen["expected_frozen_control_count"])
    baseline_support_identity_pass = bool(
        len(baseline_supported) == expected_prior
        and set(CONTROL_IDS) <= baseline_supported
        and {"e1.a", "e6.a"} <= baseline_supported
    )

    primary_supported = set(primary_legacy["supported_atoms"])
    prior_retained = sorted(baseline_supported & primary_supported)
    prior_retention_count = len(prior_retained)
    control_retained = sorted(set(CONTROL_IDS) & primary_supported)
    control_retention_count = len(control_retained)
    preservation_regression = bool(
        prior_retention_count < expected_prior
        or control_retention_count < expected_controls
    )

    equivalence_sets = {
        str(key): [str(value) for value in values]
        for key, values in experiment["posthoc_evaluation_only"][
            "source_fidelity_equivalence_sets"
        ].items()
    }
    primary_sf = _source_fidelity_support(primary, equivalence_sets)
    sensitivity_sf = _source_fidelity_support(sensitivity, equivalence_sets)
    primary_exact_residual = sorted(
        set(RESIDUAL_IDS) & primary_supported
    )
    sensitivity_exact_residual = sorted(
        set(RESIDUAL_IDS) & set(sensitivity_legacy["supported_atoms"])
    )

    primary_witnesses = _obligation_witnesses(primary, obligations)
    runtime_complete = all(primary_witnesses[str(row["id"])] for row in obligations)
    completeness_action = (
        "LOCAL_COMPLETE"
        if runtime_complete
        else str(experiment["completeness_controller"]["incomplete_action"])
    )

    selected_count_pass = len(primary) == FIXED_BUDGET
    sensitivity_budget_pass = (
        FIXED_BUDGET <= len(sensitivity) <= DIAGNOSTIC_BUDGET
    )
    unique_primary_pass = len(
        {str(row["proposition_id"]) for row in primary}
    ) == len(primary)
    unique_sensitivity_pass = len(
        {str(row["proposition_id"]) for row in sensitivity}
    ) == len(sensitivity)
    primary_locatable = all(
        str(row["proposition_id"]) in propositions_by_id
        for row in primary
    )
    sensitivity_locatable = all(
        str(row["proposition_id"]) in propositions_by_id
        for row in sensitivity
    )

    hard_safety = {
        **identity,
        "teacher_pdf_identity_pass": teacher_pdf_identity_pass,
        "baseline_origin_counts_pass": origin_counts_pass,
        "baseline_support_identity_pass": baseline_support_identity_pass,
        "origin_policy_pass": origin_policy_pass,
        "baseline_propositions_locatable": all_baseline_locatable,
        "primary_exact_budget28_pass": selected_count_pass,
        "sensitivity_budget32_ceiling_pass": sensitivity_budget_pass,
        "primary_unique_propositions_pass": unique_primary_pass,
        "sensitivity_unique_propositions_pass": unique_sensitivity_pass,
        "primary_propositions_locatable": primary_locatable,
        "sensitivity_propositions_locatable": sensitivity_locatable,
        "teacher_atoms_used_for_runtime_selection": False,
        "source_fidelity_sets_used_for_runtime_selection": False,
        "teacher_used_only_for_posthoc_evaluation": True,
        "new_model_or_ranker": False,
        "qwen_model_calls": 0,
        "threshold_search": False,
        "pairwise_semantic_similarity": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
    }
    safe = bool(
        all(identity.values())
        and teacher_pdf_identity_pass
        and origin_counts_pass
        and baseline_support_identity_pass
        and origin_policy_pass
        and all_baseline_locatable
        and selected_count_pass
        and sensitivity_budget_pass
        and unique_primary_pass
        and unique_sensitivity_pass
        and primary_locatable
        and sensitivity_locatable
    )

    mechanism = _mechanism_outcome(
        safe=safe,
        preservation_regression=preservation_regression,
        primary_source_fidelity_count=int(primary_sf["supported_count"]),
        sensitivity_source_fidelity_count=int(sensitivity_sf["supported_count"]),
    )
    completeness = _completeness_outcome(
        safe=safe,
        runtime_complete=runtime_complete,
        primary_source_fidelity_count=int(primary_sf["supported_count"]),
    )
    boundary_evidence = _acceptance_boundary_evidence(
        mechanism_outcome=mechanism,
        completeness_outcome=completeness,
        prior_retention_count=prior_retention_count,
        control_retention_count=control_retention_count,
    )
    legacy_undercount_atoms = sorted(
        set(primary_sf["supported_atoms"]) - set(primary_exact_residual)
    )
    evaluator_outcome = (
        "LEGACY_ANCHOR_UNDERCOUNT_CONFIRMED"
        if legacy_undercount_atoms
        else "NO_LEGACY_ANCHOR_UNDERCOUNT_OBSERVED"
    )

    return {
        "schema_version": 1,
        "operation": "ATOMIC_OBLIGATION_RESIDUAL_CLOSURE_DIAGNOSTIC",
        "design_id": DESIGN_ID,
        "architecture_id": ARCHITECTURE_ID,
        "experiment_id": EXPERIMENT_ID,
        "completeness_controller_id": COMPLETENESS_ID,
        "mechanism_outcome": mechanism,
        "completeness_safety_outcome": completeness,
        "acceptance_boundary_evidence": boundary_evidence,
        "secondary_evaluator_outcome": evaluator_outcome,
        "legacy_anchor_undercount_atoms": legacy_undercount_atoms,
        "primary_budget": FIXED_BUDGET,
        "diagnostic_sensitivity_budget": DIAGNOSTIC_BUDGET,
        "primary_selected_count": len(primary),
        "sensitivity_selected_count": len(sensitivity),
        "runtime_core_obligations_satisfied": sum(
            1 for values in primary_witnesses.values() if values
        ),
        "runtime_core_obligation_witnesses": primary_witnesses,
        "completeness_controller_action": completeness_action,
        "fallback_required": not runtime_complete,
        "primary_source_fidelity_residual_support": primary_sf,
        "sensitivity_source_fidelity_residual_support": sensitivity_sf,
        "primary_legacy_exact_residual_support": {
            "supported_atoms": primary_exact_residual,
            "supported_count": len(primary_exact_residual),
        },
        "sensitivity_legacy_exact_residual_support": {
            "supported_atoms": sensitivity_exact_residual,
            "supported_count": len(sensitivity_exact_residual),
        },
        "prior_supported_atom_retention": {
            "expected_count": expected_prior,
            "retained_count": prior_retention_count,
            "retained_atoms": prior_retained,
            "baseline_supported_atoms": sorted(baseline_supported),
        },
        "frozen_control_retention": {
            "expected_count": expected_controls,
            "retained_count": control_retention_count,
            "retained_atoms": control_retained,
        },
        "primary_events": primary_events,
        "sensitivity_events": sensitivity_events,
        "obligation_rankings": rankings,
        "automatic_intent_ranking_count": len(automatic_rankings),
        "automatic_intent_rankings": automatic_rankings,
        "baseline_dispensability": dispensability,
        "primary_selected_propositions": primary,
        "sensitivity_selected_propositions": sensitivity,
        "baseline_legacy_support": baseline_legacy,
        "primary_legacy_support": primary_legacy,
        "sensitivity_legacy_support": sensitivity_legacy,
        "input_identity": {
            "design_git_blob": design_git_blob,
            "fixture_git_blob": fixture_git_blob,
            "support_fixture_git_blob": support_fixture_git_blob,
            "baseline_report_sha256": baseline_report_sha256,
            "generated_intents_sha256": generated_sha256,
            "teacher_pdf_sha256": str(frozen["teacher_pdf_sha256"]),
        },
        "hard_safety": hard_safety,
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
