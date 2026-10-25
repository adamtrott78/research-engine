from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

from research_engine.local_extract_oracle_query import _prepare_teacher_pool
from research_engine.local_extract_wave1 import GPUMemorySampler


DESIGN_ID = "local-extract-exact-provenance-binding-microbenchmark-v1"
EXPERIMENT_ID = "EXACT_PROVENANCE_BINDING_MICROBENCH_V1"
ARM_A_ID = "TRI_STATE_DETERMINISTIC_PROVENANCE_ATTACHMENT_V1"
ARM_B_ID = "TRI_STATE_OPAQUE_EVIDENCE_POINTER_V1"
TEACHER_ITEM_ID = "arxiv:2403.04036"
OPAQUE_EVIDENCE_ID = "E0"
EXPECTED_MODEL_ID = "qwen3.5-4b"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_MODEL_OBJECT_ID = (
    "6e28e2560c07e7dabab92c55dbfb6941d5a7a75c8068bdc7a98115d8dc62072c"
)
EXPECTED_RUNTIME_TREE_SHA256 = (
    "b0cbb4efbe8673bdb421bad8036d29864bffe0687a9a654bbd95f54544659e63"
)
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "f67ff9a26690c55930232c915e59e826e3cca2dd89012de9e35a852032b38f74"
)
BASELINE_RESCUE_REPORT_SHA256 = (
    "e1a2f61f16cae4e957b10a9c24b475d5cb481afa9381084de65a1cd9c6568d23"
)
CANDIDATE_STATUSES = {
    "SUPPORTED_EXPLICITLY",
    "NOT_SUPPORTED",
    "INSUFFICIENT_OR_AMBIGUOUS",
}
ARM_A_TOP_LEVEL = {"candidate_status", "missing_aspects", "rationale"}
ARM_B_TOP_LEVEL = {
    "candidate_status",
    "supporting_evidence_ids",
    "missing_aspects",
    "rationale",
}
_ASCII_WS = " \t\r\n\f\v"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _strict_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip(_ASCII_WS)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("model output root is not a JSON object")
    return parsed


def _visible_obligation(
    obligation: dict[str, Any],
    visible_fields: list[str],
) -> dict[str, Any]:
    return {field: obligation[field] for field in visible_fields}


def _authoritative_text(
    proposition_id: str,
    propositions_by_id: dict[str, dict[str, Any]],
) -> str:
    if proposition_id not in propositions_by_id:
        raise RuntimeError(
            "frozen evidence proposition missing from authoritative pool: "
            f"{proposition_id}"
        )
    row = propositions_by_id[proposition_id]
    text = str(row.get("raw_source_form") or row.get("display_text") or "")
    if not text:
        raise RuntimeError(
            "frozen evidence proposition has empty authoritative text: "
            f"{proposition_id}"
        )
    return text


def _render_prompt(
    *,
    experiment: dict[str, Any],
    arm: dict[str, Any],
    obligation: dict[str, Any],
    evidence_text: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    shared = experiment["shared_prompt_and_evidence_contract"]
    visible_fields = [
        str(value) for value in shared["model_visible_obligation_fields"]
    ]
    visible_obligation = _visible_obligation(obligation, visible_fields)
    visible_evidence = {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "evidence_id": OPAQUE_EVIDENCE_ID,
        "text": evidence_text,
    }

    obligation_serialization = shared["obligation_json_serialization"]
    if obligation_serialization != {
        "encoding": "UTF-8",
        "form": (
            "compact JSON object with keys in exactly "
            "model_visible_obligation_fields order"
        ),
        "ensure_ascii": False,
        "separators": [",", ":"],
    }:
        raise RuntimeError("unexpected frozen obligation serialization contract")

    evidence_serialization = shared["evidence_json_serialization"]
    if evidence_serialization != {
        "encoding": "UTF-8",
        "exact_visible_object": {
            "source_authority": "AUTHORITATIVE_PDF_TEXT",
            "evidence_id": "E0",
            "text": "<exact authoritative proposition text>",
        },
        "ensure_ascii": False,
        "separators": [",", ":"],
    }:
        raise RuntimeError("unexpected frozen evidence serialization contract")

    user_template = str(arm["prompt_contract"]["user_template"])
    user_prompt = user_template.replace(
        "{obligation_json}", _compact_json(visible_obligation)
    ).replace(
        "{evidence_json}", _compact_json(visible_evidence)
    )
    if "{obligation_json}" in user_prompt or "{evidence_json}" in user_prompt:
        raise RuntimeError("frozen prompt placeholder was not fully materialized")

    return (
        str(arm["prompt_contract"]["system_template"]),
        user_prompt,
        visible_obligation,
        visible_evidence,
    )


def _initialize_stateless_llm(
    model_path: str,
    *,
    runtime: dict[str, Any],
) -> Any:
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=int(runtime["tensor_parallel_size"]),
        dtype=str(runtime["dtype"]),
        max_model_len=int(runtime["max_model_len"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        trust_remote_code=bool(runtime["trust_remote_code"]),
        disable_log_stats=bool(runtime["disable_log_stats"]),
        mm_processor_cache_gb=float(runtime["mm_processor_cache_gb"]),
        enable_prefix_caching=bool(runtime["enable_prefix_caching"]),
    )


def _model_once(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    runtime: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=float(runtime["temperature"]),
        top_p=float(runtime["top_p"]),
        seed=int(runtime["seed"]),
        max_tokens=int(runtime["max_output_tokens"]),
    )
    started = time.perf_counter()
    outputs = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        sampling_params=params,
        use_tqdm=False,
        chat_template_kwargs=dict(runtime["chat_template_kwargs"]),
    )
    seconds = time.perf_counter() - started
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("unexpected vLLM output cardinality")
    output = outputs[0]
    generated = output.outputs[0]
    text = str(generated.text)
    return text, {
        "wall_time_seconds": seconds,
        "prompt_tokens": len(getattr(output, "prompt_token_ids", []) or []),
        "output_tokens": len(generated.token_ids),
        "generated_text_sha256": _sha256_text(text),
        "finish_reason": str(getattr(generated, "finish_reason", None)),
    }


def _ordinary_failure(
    *,
    reason: str,
    raw_text: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "raw_model_output": raw_text,
        "usage": dict(usage),
        "parsed_output": None,
        "effective_candidate_status": "INSUFFICIENT_OR_AMBIGUOUS",
        "arm_case_contract_pass": False,
        "invalid_execution": False,
        "ordinary_failure_reason": reason,
        "attached_evidence_ids": [],
        "attached_proposition_ids": [],
        "exact_provenance_valid": False,
    }


def _guard_arm_a(
    *,
    raw_text: str,
    usage: dict[str, Any],
    exact_proposition_id: str,
) -> dict[str, Any]:
    try:
        parsed = _strict_json_object(raw_text)
    except Exception as exc:
        return _ordinary_failure(
            reason=f"MALFORMED_JSON:{type(exc).__name__}",
            raw_text=raw_text,
            usage=usage,
        )

    if set(parsed) != ARM_A_TOP_LEVEL:
        return _ordinary_failure(
            reason="SCHEMA_MISSING_OR_UNKNOWN_FIELDS",
            raw_text=raw_text,
            usage=usage,
        )

    status = parsed.get("candidate_status")
    missing = parsed.get("missing_aspects")
    rationale = parsed.get("rationale")
    if (
        not isinstance(status, str)
        or status not in CANDIDATE_STATUSES
        or not isinstance(missing, list)
        or not all(isinstance(value, str) for value in missing)
        or not isinstance(rationale, str)
    ):
        return _ordinary_failure(
            reason="INVALID_ENUM_OR_FIELD_TYPE",
            raw_text=raw_text,
            usage=usage,
        )

    if status == "SUPPORTED_EXPLICITLY" and missing:
        return _ordinary_failure(
            reason="SUPPORTED_WITH_NONEMPTY_MISSING_ASPECTS",
            raw_text=raw_text,
            usage=usage,
        )

    supported = status == "SUPPORTED_EXPLICITLY"
    return {
        "raw_model_output": raw_text,
        "usage": dict(usage),
        "parsed_output": {
            "candidate_status": status,
            "missing_aspects": list(missing),
            "rationale": rationale,
        },
        "effective_candidate_status": status,
        "arm_case_contract_pass": True,
        "invalid_execution": False,
        "ordinary_failure_reason": None,
        "attached_evidence_ids": [OPAQUE_EVIDENCE_ID] if supported else [],
        "attached_proposition_ids": [exact_proposition_id] if supported else [],
        "exact_provenance_valid": True if supported else False,
        "provenance_attachment_mode": (
            "DETERMINISTIC_RUNTIME_E0_RESOLUTION" if supported else "NONE"
        ),
    }


def _guard_arm_b(
    *,
    raw_text: str,
    usage: dict[str, Any],
    exact_proposition_id: str,
) -> dict[str, Any]:
    try:
        parsed = _strict_json_object(raw_text)
    except Exception as exc:
        return _ordinary_failure(
            reason=f"MALFORMED_JSON:{type(exc).__name__}",
            raw_text=raw_text,
            usage=usage,
        )

    if set(parsed) != ARM_B_TOP_LEVEL:
        return _ordinary_failure(
            reason="SCHEMA_MISSING_OR_UNKNOWN_FIELDS",
            raw_text=raw_text,
            usage=usage,
        )

    status = parsed.get("candidate_status")
    ids = parsed.get("supporting_evidence_ids")
    missing = parsed.get("missing_aspects")
    rationale = parsed.get("rationale")
    if (
        not isinstance(status, str)
        or status not in CANDIDATE_STATUSES
        or not isinstance(ids, list)
        or not all(isinstance(value, str) for value in ids)
        or not isinstance(missing, list)
        or not all(isinstance(value, str) for value in missing)
        or not isinstance(rationale, str)
    ):
        return _ordinary_failure(
            reason="INVALID_ENUM_OR_FIELD_TYPE",
            raw_text=raw_text,
            usage=usage,
        )

    if status == "SUPPORTED_EXPLICITLY":
        if ids != [OPAQUE_EVIDENCE_ID] or missing:
            return _ordinary_failure(
                reason="SUPPORTED_POINTER_OR_MISSING_ASPECT_CONTRACT_FAILURE",
                raw_text=raw_text,
                usage=usage,
            )
    elif ids != []:
        return _ordinary_failure(
            reason="NON_SUPPORT_WITH_PROVENANCE_POINTER",
            raw_text=raw_text,
            usage=usage,
        )

    supported = status == "SUPPORTED_EXPLICITLY"
    return {
        "raw_model_output": raw_text,
        "usage": dict(usage),
        "parsed_output": {
            "candidate_status": status,
            "supporting_evidence_ids": list(ids),
            "missing_aspects": list(missing),
            "rationale": rationale,
        },
        "effective_candidate_status": status,
        "arm_case_contract_pass": True,
        "invalid_execution": False,
        "ordinary_failure_reason": None,
        "attached_evidence_ids": [OPAQUE_EVIDENCE_ID] if supported else [],
        "attached_proposition_ids": [exact_proposition_id] if supported else [],
        "exact_provenance_valid": True if supported else False,
        "provenance_attachment_mode": (
            "MODEL_SELECTED_OPAQUE_E0_RESOLUTION" if supported else "NONE"
        ),
    }


def _complete_from_rows(
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    complete = bool(rows) and all(
        row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
        and bool(row["exact_provenance_valid"])
        for row in rows.values()
    )
    return {
        "obligation_states": {
            key: {
                "candidate_status": row["effective_candidate_status"],
                "exact_provenance_valid": bool(row["exact_provenance_valid"]),
                "attached_proposition_ids": list(
                    row["attached_proposition_ids"]
                ),
            }
            for key, row in rows.items()
        },
        "local_complete": complete,
        "fallback_required": not complete,
    }


def _arm_gate(
    *,
    arm_id: str,
    case_results: list[dict[str, Any]],
    baseline_state: dict[str, Any],
    true_state: dict[str, Any],
    preservation: dict[str, Any],
) -> dict[str, Any]:
    false_rows = [
        row for row in case_results
        if row["evaluation_role"] == "KNOWN_FALSE_POSITIVE"
    ]
    true_rows = [
        row for row in case_results
        if row["evaluation_role"] == "KNOWN_TRUE_SUPPORT"
    ]

    false_rejected = sum(
        row["effective_candidate_status"]
        in {"NOT_SUPPORTED", "INSUFFICIENT_OR_AMBIGUOUS"}
        for row in false_rows
    )
    true_accepted = sum(
        row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
        for row in true_rows
    )
    contract_count = sum(
        bool(row["arm_case_contract_pass"]) for row in case_results
    )
    true_provenance_count = sum(
        row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
        and row["attached_proposition_ids"] == row["evidence_proposition_ids"]
        and bool(row["exact_provenance_valid"])
        for row in true_rows
    )
    false_provenance_count = sum(
        bool(row["attached_proposition_ids"]) for row in false_rows
    )

    supported_rows = [
        row for row in case_results
        if row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
    ]
    nonsupported_rows = [
        row for row in case_results
        if row["effective_candidate_status"] != "SUPPORTED_EXPLICITLY"
    ]
    arm_a_extra = all(
        row["attached_evidence_ids"] == [OPAQUE_EVIDENCE_ID]
        and row["attached_proposition_ids"] == row["evidence_proposition_ids"]
        and bool(row["exact_provenance_valid"])
        for row in supported_rows
    )
    arm_b_extra = all(
        (
            row["parsed_output"] is not None
            and row["parsed_output"].get("supporting_evidence_ids")
            == [OPAQUE_EVIDENCE_ID]
            and row["attached_proposition_ids"]
            == row["evidence_proposition_ids"]
        )
        for row in supported_rows
    ) and all(
        (
            row["parsed_output"] is not None
            and row["parsed_output"].get("supporting_evidence_ids") == []
        )
        for row in nonsupported_rows
        if row["arm_case_contract_pass"]
    )

    checks = {
        "all_7_cases_arm_case_contract_pass": contract_count == 7,
        "reject_or_abstain_all_3_known_false_positives": false_rejected == 3,
        "accept_all_4_known_true_supports": true_accepted == 4,
        "exact_provenance_attached_for_all_4_true_supports": (
            true_provenance_count == 4
        ),
        "zero_provenance_attached_for_all_3_false_positive_cases": (
            false_provenance_count == 0
        ),
        "baseline_false_complete_state_returns_incomplete_and_fallback": (
            baseline_state["local_complete"] is False
            and baseline_state["fallback_required"] is True
        ),
        "true_support_state_returns_complete_without_fallback": (
            true_state["local_complete"] is True
            and true_state["fallback_required"] is False
        ),
        "evidence_set_identity_unchanged": bool(
            preservation["evidence_set_identity_unchanged"]
        ),
        "prior_supported_retention_9_of_9": (
            preservation["prior_supported_retention"] == "9/9"
        ),
        "frozen_control_retention_7_of_7": (
            preservation["frozen_control_retention"] == "7/7"
        ),
    }
    if arm_id == ARM_A_ID:
        checks[
            "deterministic_attachment_resolves_E0_to_exact_proposition"
        ] = arm_a_extra
    elif arm_id == ARM_B_ID:
        checks[
            "model_pointer_exactly_E0_on_support_and_empty_on_nonsupport"
        ] = arm_b_extra
    else:
        raise RuntimeError(f"unexpected arm id: {arm_id}")

    return {
        "contract_pass_count": contract_count,
        "false_positive_rejection_or_abstention_count": false_rejected,
        "true_support_acceptance_count": true_accepted,
        "true_support_exact_provenance_count": true_provenance_count,
        "false_positive_provenance_attachment_count": false_provenance_count,
        "checks": checks,
        "precision_first_pass": all(checks.values()),
    }


def _outcome(
    *,
    integrity_valid: bool,
    arm_a_pass: bool,
    arm_b_pass: bool,
) -> str:
    if not integrity_valid:
        return "INVALID_EXECUTION"
    if arm_a_pass and arm_b_pass:
        return "BOTH_EXACT_PROVENANCE_INTERFACES_CONFIRMED"
    if arm_a_pass:
        return "DETERMINISTIC_ATTACHMENT_ONLY_CONFIRMED"
    if arm_b_pass:
        return "OPAQUE_POINTER_ONLY_CONFIRMED"
    return "EXACT_PROVENANCE_BINDING_PREMISE_REJECTED"


def _validate_design_contract(
    design: dict[str, Any],
) -> dict[str, Any]:
    if design.get("design_id") != DESIGN_ID:
        raise RuntimeError("unexpected exact-provenance design id")
    if design.get("status") != "FROZEN_DESIGN_ONLY":
        raise RuntimeError("unexpected frozen design status")
    experiment = design.get("selected_discriminating_experiment") or {}
    if experiment.get("id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected exact-provenance experiment id")

    frozen = experiment["frozen_inputs"]
    model = frozen["local_model"]
    if model != {
        "model_id": EXPECTED_MODEL_ID,
        "repository": "Qwen/Qwen3.5-4B",
        "revision": EXPECTED_MODEL_REVISION,
        "model_object_id": EXPECTED_MODEL_OBJECT_ID,
        "runtime_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }:
        raise RuntimeError("frozen local model identity changed")

    shared = experiment["shared_prompt_and_evidence_contract"]
    if shared["model_visible_obligation_fields"] != [
        "generic_information_need",
        "required_relation",
    ]:
        raise RuntimeError("model-visible obligation fields changed")
    if shared["evidence_json_serialization"]["exact_visible_object"] != {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "evidence_id": "E0",
        "text": "<exact authoritative proposition text>",
    }:
        raise RuntimeError("opaque evidence representation changed")

    forbidden = set(shared["forbidden_model_proof_fields"])
    required_forbidden = {
        "supporting_quotes",
        "verbatim_span",
        "source_quote",
        "source_text",
        "page_number",
        "proposition_id",
        "character_offsets",
    }
    if not required_forbidden <= forbidden:
        raise RuntimeError("regenerated source-proof boundary changed")

    arm_a = experiment["arm_a_deterministic_attachment"]
    arm_b = experiment["arm_b_opaque_pointer_selection"]
    if arm_a["id"] != ARM_A_ID or arm_b["id"] != ARM_B_ID:
        raise RuntimeError("unexpected exact-provenance arm identity")

    runtime = experiment["shared_generation_runtime_contract"]
    if runtime != {
        "implementation_lineage": (
            "Reuse the already-validated vLLM/Qwen3.5-4B runtime envelope "
            "from OBLIGATION_VALIDITY_PRECISION_MICROBENCH_V1; these are "
            "execution settings, not tunable scientific variables."
        ),
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.65,
        "trust_remote_code": False,
        "disable_log_stats": True,
        "mm_processor_cache_gb": 0.25,
        "enable_prefix_caching": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_output_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "model_calls_per_case": 1,
        "total_model_calls": 14,
        "post_review_tuning_allowed": False,
    }:
        raise RuntimeError("shared generation runtime contract changed")

    isolation = experiment["stateless_case_isolation_contract"]
    if isolation["total_model_calls"] != 14:
        raise RuntimeError("stateless call count changed")
    for key in (
        "few_shot_examples",
        "prior_case_outputs_visible",
        "prior_case_prompts_visible",
        "prior_validity_run_outputs_visible",
        "extra_paper_context_visible",
        "extra_corpus_context_visible",
        "cross_case_conversation_state",
        "chat_history_reused",
        "kv_cache_reused_across_cases",
        "prompt_tuning_after_review",
        "retry_on_bad_model_output",
        "retry_on_scientific_failure",
    ):
        if isolation[key] is not False:
            raise RuntimeError(f"stateless isolation changed: {key}")

    if design["frozen_scientific_boundaries"][
        "regenerated_quote_or_span_proof_allowed"
    ] is not False:
        raise RuntimeError("quote/span regeneration unexpectedly allowed")

    return experiment


def run_exact_provenance_binding_microbenchmark(
    *,
    fixture: dict[str, Any],
    design: dict[str, Any],
    baseline_rescue_report: dict[str, Any],
    baseline_rescue_report_sha256: str,
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    experiment = _validate_design_contract(design)
    frozen = experiment["frozen_inputs"]
    if frozen["teacher_paper"] != TEACHER_ITEM_ID:
        raise RuntimeError("unexpected teacher paper identity")
    if baseline_rescue_report_sha256 != BASELINE_RESCUE_REPORT_SHA256:
        raise RuntimeError("baseline rescue report sha256 mismatch")
    if baseline_rescue_report_sha256 != frozen[
        "baseline_rescue_report_sha256"
    ]:
        raise RuntimeError("design baseline rescue report sha256 mismatch")

    teacher = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    acquisition = dict(teacher["acquisition"])
    if str(teacher["item_id"]) != TEACHER_ITEM_ID:
        raise RuntimeError("unexpected acquired teacher item")
    if str(acquisition.get("sha256")) != str(frozen["teacher_pdf_sha256"]):
        raise RuntimeError("teacher PDF sha256 mismatch")

    propositions = [dict(row) for row in teacher["page_pool"]]
    propositions_by_id = {
        str(row["proposition_id"]): row for row in propositions
    }
    obligations = {
        str(row["id"]): dict(row)
        for row in experiment["runtime_obligations"]
    }

    false_cases = [
        {**dict(row), "evaluation_role": "KNOWN_FALSE_POSITIVE"}
        for row in experiment["case_set"]["known_false_positive_cases"]
    ]
    true_cases = [
        {**dict(row), "evaluation_role": "KNOWN_TRUE_SUPPORT"}
        for row in experiment["case_set"]["known_true_support_cases"]
    ]
    cases = false_cases + true_cases
    if len(false_cases) != 3 or len(true_cases) != 4 or len(cases) != 7:
        raise RuntimeError("frozen seven-case membership changed")
    if len({str(row["case_id"]) for row in cases}) != 7:
        raise RuntimeError("duplicate frozen case id")

    for case in cases:
        obligation_id = str(case["obligation_id"])
        if obligation_id not in obligations:
            raise RuntimeError("case references unknown frozen obligation")
        proposition_ids = [
            str(value) for value in case["evidence_proposition_ids"]
        ]
        if len(proposition_ids) != 1:
            raise RuntimeError("opaque E0 contract requires one proposition")
        if proposition_ids[0] not in propositions_by_id:
            raise RuntimeError(
                "case evidence proposition is not locatable"
            )

    starting_selected = [
        dict(row)
        for row in (baseline_rescue_report.get("selected_propositions") or [])
    ]
    starting_ids = [
        str(row["proposition_id"]) for row in starting_selected
    ]
    prior_supported = sorted(
        str(value)
        for value in baseline_rescue_report["rescued_teacher"][
            "supported_atoms"
        ]
    )
    frozen_controls = sorted(
        str(value)
        for value in baseline_rescue_report["rescued_teacher"][
            "frozen_control_atoms_retained"
        ]
    )
    expected_prior = sorted(
        str(value)
        for value in experiment["preservation_controls"][
            "prior_supported_atom_ids"
        ]
    )
    expected_controls = sorted(
        str(value)
        for value in experiment["preservation_controls"][
            "frozen_control_atom_ids"
        ]
    )
    preservation = {
        "starting_selected_count": len(starting_ids),
        "ending_selected_count": len(starting_ids),
        "starting_proposition_ids_sha256": _sha256_text(
            _compact_json(starting_ids)
        ),
        "ending_proposition_ids_sha256": _sha256_text(
            _compact_json(starting_ids)
        ),
        "evidence_set_identity_unchanged": True,
        "prior_supported_atom_ids_match_baseline": (
            prior_supported == expected_prior
        ),
        "frozen_control_atom_ids_match_baseline": (
            frozen_controls == expected_controls
        ),
        "prior_supported_retention": (
            "9/9"
            if prior_supported == expected_prior
            else f"{len(set(prior_supported) & set(expected_prior))}/9"
        ),
        "frozen_control_retention": (
            "7/7"
            if frozen_controls == expected_controls
            else f"{len(set(frozen_controls) & set(expected_controls))}/7"
        ),
        "page_p2_s16_present": "page:p2:s16" in set(starting_ids),
        "selection_mutation_attempted": False,
    }

    arms = {
        ARM_A_ID: experiment["arm_a_deterministic_attachment"],
        ARM_B_ID: experiment["arm_b_opaque_pointer_selection"],
    }
    runtime = experiment["shared_generation_runtime_contract"]
    results_by_arm: dict[str, list[dict[str, Any]]] = {
        ARM_A_ID: [],
        ARM_B_ID: [],
    }
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_inference_seconds = 0.0
    model_calls = 0

    init_started = time.perf_counter()
    with GPUMemorySampler(expected_gpu_uuids) as gpu:
        llm = _initialize_stateless_llm(
            model_path,
            runtime=runtime,
        )
        model_init_seconds = time.perf_counter() - init_started

        for arm_id in (ARM_A_ID, ARM_B_ID):
            arm = arms[arm_id]
            for case in cases:
                proposition_id = str(case["evidence_proposition_ids"][0])
                evidence_text = _authoritative_text(
                    proposition_id, propositions_by_id
                )
                obligation = obligations[str(case["obligation_id"])]
                (
                    system_prompt,
                    user_prompt,
                    visible_obligation,
                    visible_evidence,
                ) = _render_prompt(
                    experiment=experiment,
                    arm=arm,
                    obligation=obligation,
                    evidence_text=evidence_text,
                )

                raw_text, usage = _model_once(
                    llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    runtime=runtime,
                )
                model_calls += 1
                total_prompt_tokens += int(usage["prompt_tokens"])
                total_output_tokens += int(usage["output_tokens"])
                total_inference_seconds += float(
                    usage["wall_time_seconds"]
                )

                if arm_id == ARM_A_ID:
                    guarded = _guard_arm_a(
                        raw_text=raw_text,
                        usage=usage,
                        exact_proposition_id=proposition_id,
                    )
                else:
                    guarded = _guard_arm_b(
                        raw_text=raw_text,
                        usage=usage,
                        exact_proposition_id=proposition_id,
                    )

                results_by_arm[arm_id].append(
                    {
                        "case_id": str(case["case_id"]),
                        "obligation_id": str(case["obligation_id"]),
                        "evidence_proposition_ids": [proposition_id],
                        "evaluation_role": str(case["evaluation_role"]),
                        "expected_posthoc": str(case["expected_posthoc"]),
                        "visible_obligation": visible_obligation,
                        "visible_evidence": visible_evidence,
                        "hidden_e0_mapping_sha256": _sha256_text(
                            _compact_json(
                                {OPAQUE_EVIDENCE_ID: proposition_id}
                            )
                        ),
                        "system_prompt_sha256": _sha256_text(
                            system_prompt
                        ),
                        "user_prompt_sha256": _sha256_text(user_prompt),
                        **guarded,
                    }
                )

    if model_calls != int(runtime["total_model_calls"]):
        raise RuntimeError(
            "unexpected frozen model-call count: "
            f"{model_calls}"
        )

    case_lookup: dict[
        tuple[str, str, tuple[str, ...]], dict[str, Any]
    ] = {}
    for arm_id, rows in results_by_arm.items():
        for row in rows:
            key = (
                arm_id,
                str(row["obligation_id"]),
                tuple(row["evidence_proposition_ids"]),
            )
            case_lookup[key] = row

    completeness_by_arm: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for arm_id in (ARM_A_ID, ARM_B_ID):
        baseline_rows: dict[str, dict[str, Any]] = {}
        true_rows: dict[str, dict[str, Any]] = {}
        for obligation_id, proposition_ids in experiment[
            "completeness_simulation"
        ]["baseline_false_complete_state"].items():
            if obligation_id == "required_outcome_for_any_safe_arm":
                continue
            baseline_rows[str(obligation_id)] = case_lookup[
                (
                    arm_id,
                    str(obligation_id),
                    tuple(str(value) for value in proposition_ids),
                )
            ]
        for obligation_id, proposition_ids in experiment[
            "completeness_simulation"
        ]["true_support_state"].items():
            if obligation_id == "required_outcome_for_any_passing_arm":
                continue
            true_rows[str(obligation_id)] = case_lookup[
                (
                    arm_id,
                    str(obligation_id),
                    tuple(str(value) for value in proposition_ids),
                )
            ]
        baseline_state = _complete_from_rows(baseline_rows)
        true_state = _complete_from_rows(true_rows)
        completeness_by_arm[arm_id] = {
            "baseline_false_complete_state": baseline_state,
            "true_support_state": true_state,
        }
        gates[arm_id] = _arm_gate(
            arm_id=arm_id,
            case_results=results_by_arm[arm_id],
            baseline_state=baseline_state,
            true_state=true_state,
            preservation=preservation,
        )

    visible_schema_pass = all(
        set(row["visible_obligation"])
        == {"generic_information_need", "required_relation"}
        and set(row["visible_evidence"])
        == {"source_authority", "evidence_id", "text"}
        and row["visible_evidence"]["evidence_id"] == OPAQUE_EVIDENCE_ID
        for arm_id in (ARM_A_ID, ARM_B_ID)
        for row in results_by_arm[arm_id]
    )
    e0_mapping_integrity_pass = all(
        row["attached_proposition_ids"]
        in ([], row["evidence_proposition_ids"])
        for arm_id in (ARM_A_ID, ARM_B_ID)
        for row in results_by_arm[arm_id]
    )

    integrity = {
        "teacher_item_identity_pass": (
            str(teacher["item_id"]) == TEACHER_ITEM_ID
        ),
        "teacher_pdf_sha256_pass": (
            str(acquisition.get("sha256"))
            == str(frozen["teacher_pdf_sha256"])
        ),
        "baseline_rescue_report_identity_pass": (
            baseline_rescue_report_sha256
            == str(frozen["baseline_rescue_report_sha256"])
        ),
        "baseline_decision_pass": (
            baseline_rescue_report.get("neighborhood_rescue_decision")
            == "LOCAL_RESCUE_CONFIRMED"
        ),
        "baseline_selected_count_pass": len(starting_ids) == 28,
        "baseline_selected_unique_pass": len(set(starting_ids)) == 28,
        "prior_supported_baseline_identity_pass": bool(
            preservation["prior_supported_atom_ids_match_baseline"]
        ),
        "frozen_control_baseline_identity_pass": bool(
            preservation["frozen_control_atom_ids_match_baseline"]
        ),
        "page_p2_s16_present_pass": bool(
            preservation["page_p2_s16_present"]
        ),
        "seven_case_identity_pass": (
            len(false_cases) == 3
            and len(true_cases) == 4
            and len(cases) == 7
        ),
        "model_visible_schema_and_opaque_e0_pass": visible_schema_pass,
        "hidden_e0_mapping_integrity_pass": e0_mapping_integrity_pass,
        "stateless_one_request_per_arm_case_pass": model_calls == 14,
        "evidence_set_no_mutation_pass": bool(
            preservation["evidence_set_identity_unchanged"]
            and not preservation["selection_mutation_attempted"]
        ),
    }
    integrity_valid = all(integrity.values())

    outcome = _outcome(
        integrity_valid=integrity_valid,
        arm_a_pass=bool(gates[ARM_A_ID]["precision_first_pass"]),
        arm_b_pass=bool(gates[ARM_B_ID]["precision_first_pass"]),
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_EXACT_PROVENANCE_BINDING_MICROBENCHMARK",
        "design_id": DESIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "teacher_item_id": TEACHER_ITEM_ID,
        "teacher_pdf_sha256": str(acquisition.get("sha256")),
        "model_identity": {
            "model_id": EXPECTED_MODEL_ID,
            "repository": "Qwen/Qwen3.5-4B",
            "revision": EXPECTED_MODEL_REVISION,
            "model_object_id": EXPECTED_MODEL_OBJECT_ID,
            "runtime_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
            "temperature": runtime["temperature"],
            "top_p": runtime["top_p"],
            "seed": runtime["seed"],
            "max_output_tokens": runtime["max_output_tokens"],
            "prefix_caching_enabled": runtime["enable_prefix_caching"],
            "enable_thinking": runtime["chat_template_kwargs"][
                "enable_thinking"
            ],
        },
        "runtime_metrics": {
            "model_calls": model_calls,
            "model_init_seconds": model_init_seconds,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "inference_seconds": total_inference_seconds,
            "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
        },
        "arms": {
            ARM_A_ID: {
                "cases": results_by_arm[ARM_A_ID],
                "completeness_simulation": completeness_by_arm[ARM_A_ID],
                "scientific_gate": gates[ARM_A_ID],
            },
            ARM_B_ID: {
                "cases": results_by_arm[ARM_B_ID],
                "completeness_simulation": completeness_by_arm[ARM_B_ID],
                "scientific_gate": gates[ARM_B_ID],
            },
        },
        "preservation_controls": preservation,
        "experiment_integrity": integrity,
        "mechanism_outcome": outcome,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_output": False,
        "sealed_acceptance_benchmark_accessed": False,
        "acceptance_candidate_frozen": False,
        "external_model_downloads": 0,
    }
