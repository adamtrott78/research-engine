from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

from research_engine.local_extract_oracle_query import _prepare_teacher_pool
from research_engine.local_extract_wave1 import GPUMemorySampler


DESIGN_ID = "local-extract-obligation-validity-microbenchmark-v1"
EXPERIMENT_ID = "OBLIGATION_VALIDITY_PRECISION_MICROBENCH_V1"
ARM_A_ID = "PROOF_CARRYING_EXPLICIT_SLOT_GROUNDING_V1"
ARM_B_ID = "HOLISTIC_TRI_STATE_SOURCE_GROUNDED_VERIFIER_V1"
TEACHER_ITEM_ID = "arxiv:2403.04036"
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
# Operational generation envelope only. The frozen stochastic decoding parameters
# remain temperature=0, top_p=1, seed=0. This cap is not used by any verdict rule.
OPERATIONAL_MAX_OUTPUT_TOKENS = 512
_ASCII_WS = " \t\r\n\f\v"

CANDIDATE_STATUSES = {
    "SUPPORTED_EXPLICITLY",
    "NOT_SUPPORTED",
    "INSUFFICIENT_OR_AMBIGUOUS",
}
ARM_A_TOP_LEVEL = {
    "stage_role",
    "slot_bindings",
    "relation_check",
    "candidate_status",
}
ARM_B_TOP_LEVEL = {
    "candidate_status",
    "supporting_quotes",
    "missing_aspects",
    "rationale",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_newlines(value: str) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _visible_obligation(
    obligation: dict[str, Any],
    visible_fields: list[str],
) -> dict[str, Any]:
    return {
        field: obligation[field]
        for field in visible_fields
    }


def _evidence_text(
    proposition_ids: list[str],
    propositions_by_id: dict[str, dict[str, Any]],
) -> str:
    rows: list[str] = []
    for proposition_id in proposition_ids:
        if proposition_id not in propositions_by_id:
            raise RuntimeError(
                f"frozen evidence proposition missing from authoritative pool: "
                f"{proposition_id}"
            )
        row = propositions_by_id[proposition_id]
        text = str(
            row.get("raw_source_form")
            or row.get("display_text")
            or ""
        )
        if not text:
            raise RuntimeError(
                f"frozen evidence proposition has empty authoritative text: "
                f"{proposition_id}"
            )
        rows.append(text)
    return "\n".join(rows)


def _render_prompt(
    *,
    arm: dict[str, Any],
    obligation: dict[str, Any],
    evidence_text: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    visible_fields = [
        str(value)
        for value in arm["model_visible_obligation_fields"]
    ]
    obligation_obj = _visible_obligation(
        obligation,
        visible_fields,
    )
    evidence_obj = {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "text": evidence_text,
    }
    serialization = arm["prompt_contract"]["obligation_json_serialization"]
    if serialization != {
        "encoding": "UTF-8",
        "form": (
            "compact JSON object with keys in exactly "
            "model_visible_obligation_fields order"
        ),
        "ensure_ascii": False,
        "separators": [",", ":"],
    }:
        raise RuntimeError("unexpected frozen obligation serialization contract")
    evidence_serialization = arm["prompt_contract"][
        "evidence_json_serialization"
    ]
    if evidence_serialization["encoding"] != "UTF-8":
        raise RuntimeError("unexpected evidence encoding")
    if evidence_serialization["ensure_ascii"] is not False:
        raise RuntimeError("unexpected evidence ensure_ascii setting")
    if evidence_serialization["separators"] != [",", ":"]:
        raise RuntimeError("unexpected evidence separators")
    if evidence_serialization["exact_visible_object"] != {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "text": (
            "<exact authoritative proposition text or exact concatenation "
            "of the frozen evidence bundle texts separated by a single newline>"
        ),
    }:
        raise RuntimeError("unexpected frozen evidence representation contract")

    obligation_json = _compact_json(obligation_obj)
    evidence_json = _compact_json(evidence_obj)
    user_template = str(arm["prompt_contract"]["user_template"])
    user_prompt = user_template.replace(
        "{obligation_json}",
        obligation_json,
    ).replace(
        "{evidence_json}",
        evidence_json,
    )
    if "{obligation_json}" in user_prompt or "{evidence_json}" in user_prompt:
        raise RuntimeError("frozen prompt placeholder was not fully materialized")
    system_prompt = str(arm["prompt_contract"]["system_template"])
    return system_prompt, user_prompt, obligation_obj, evidence_obj


def _initialize_stateless_llm(
    model_path: str,
    *,
    tensor_parallel_size: int,
) -> Any:
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        trust_remote_code=False,
        disable_log_stats=True,
        mm_processor_cache_gb=0.25,
        enable_prefix_caching=False,
    )


def _model_once(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=OPERATIONAL_MAX_OUTPUT_TOKENS,
    )
    started = time.perf_counter()
    outputs = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        sampling_params=params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    seconds = time.perf_counter() - started
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("unexpected vLLM output cardinality")
    output = outputs[0]
    generated = output.outputs[0]
    text = str(generated.text)
    return text, {
        "wall_time_seconds": seconds,
        "prompt_tokens": len(
            getattr(output, "prompt_token_ids", []) or []
        ),
        "output_tokens": len(generated.token_ids),
        "generated_text_sha256": _sha256_text(text),
        "finish_reason": str(
            getattr(generated, "finish_reason", None)
        ),
    }


def _strict_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip(_ASCII_WS)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("model output root is not a JSON object")
    return parsed


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
        "non_exact_quote_or_span_attempt": (
            reason == "NON_EXACT_QUOTE_OR_SPAN_ATTEMPT"
        ),
    }


def _exact_substring(candidate: str, evidence_text: str) -> bool:
    normalized_candidate = _normalize_newlines(candidate)
    normalized_evidence = _normalize_newlines(evidence_text)
    return bool(normalized_candidate) and (
        normalized_candidate in normalized_evidence
    )


def _guard_arm_a(
    *,
    raw_text: str,
    usage: dict[str, Any],
    obligation: dict[str, Any],
    evidence_text: str,
    arm: dict[str, Any],
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

    stage_role = parsed.get("stage_role")
    relation_check = parsed.get("relation_check")
    candidate_status = parsed.get("candidate_status")
    bindings = parsed.get("slot_bindings")
    stage_enum = set(arm["model_output_contract"]["stage_role"])
    relation_enum = set(
        arm["model_output_contract"]["relation_check"]
    )
    if (
        not isinstance(stage_role, str)
        or stage_role not in stage_enum
        or not isinstance(relation_check, str)
        or relation_check not in relation_enum
        or not isinstance(candidate_status, str)
        or candidate_status not in CANDIDATE_STATUSES
        or not isinstance(bindings, list)
    ):
        return _ordinary_failure(
            reason="INVALID_ENUM_OR_FIELD_TYPE",
            raw_text=raw_text,
            usage=usage,
        )

    mandatory = [
        str(value)
        for value in obligation["mandatory_slots"]
    ]
    seen: list[str] = []
    normalized_bindings: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "slot_id",
            "verbatim_span",
        }:
            return _ordinary_failure(
                reason="SCHEMA_MISSING_OR_UNKNOWN_FIELDS",
                raw_text=raw_text,
                usage=usage,
            )
        slot_id = binding.get("slot_id")
        span = binding.get("verbatim_span")
        if (
            not isinstance(slot_id, str)
            or slot_id not in mandatory
            or slot_id in seen
            or (span is not None and not isinstance(span, str))
        ):
            return _ordinary_failure(
                reason="SCHEMA_MISSING_OR_UNKNOWN_FIELDS",
                raw_text=raw_text,
                usage=usage,
            )
        if isinstance(span, str) and not _exact_substring(
            span,
            evidence_text,
        ):
            return _ordinary_failure(
                reason="NON_EXACT_QUOTE_OR_SPAN_ATTEMPT",
                raw_text=raw_text,
                usage=usage,
            )
        seen.append(slot_id)
        normalized_bindings.append(
            {
                "slot_id": slot_id,
                "verbatim_span": span,
            }
        )

    if set(seen) != set(mandatory) or len(seen) != len(mandatory):
        return _ordinary_failure(
            reason="MISSING_REQUIRED_BINDING",
            raw_text=raw_text,
            usage=usage,
        )

    all_bound = all(
        isinstance(row["verbatim_span"], str)
        and bool(row["verbatim_span"])
        for row in normalized_bindings
    )
    accepted_stage = stage_role in {
        str(value)
        for value in obligation["accepted_stage_roles"]
    }
    explicit_relation = relation_check == "EXPLICITLY_LINKED"

    if candidate_status == "SUPPORTED_EXPLICITLY" and not all_bound:
        return _ordinary_failure(
            reason="SUPPORTED_WITH_MISSING_REQUIRED_BINDING",
            raw_text=raw_text,
            usage=usage,
        )

    effective = candidate_status
    if (
        candidate_status == "SUPPORTED_EXPLICITLY"
        and (not accepted_stage or not explicit_relation)
    ):
        effective = "INSUFFICIENT_OR_AMBIGUOUS"

    return {
        "raw_model_output": raw_text,
        "usage": dict(usage),
        "parsed_output": {
            "stage_role": stage_role,
            "slot_bindings": normalized_bindings,
            "relation_check": relation_check,
            "candidate_status": candidate_status,
        },
        "effective_candidate_status": effective,
        "arm_case_contract_pass": True,
        "invalid_execution": False,
        "ordinary_failure_reason": None,
        "non_exact_quote_or_span_attempt": False,
        "deterministic_guard": {
            "all_mandatory_slots_bound": all_bound,
            "accepted_stage_role": accepted_stage,
            "required_relation_explicitly_linked": explicit_relation,
        },
    }


def _guard_arm_b(
    *,
    raw_text: str,
    usage: dict[str, Any],
    evidence_text: str,
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

    candidate_status = parsed.get("candidate_status")
    quotes = parsed.get("supporting_quotes")
    missing = parsed.get("missing_aspects")
    rationale = parsed.get("rationale")
    if (
        not isinstance(candidate_status, str)
        or candidate_status not in CANDIDATE_STATUSES
        or not isinstance(quotes, list)
        or not all(isinstance(value, str) for value in quotes)
        or not isinstance(missing, list)
        or not all(isinstance(value, str) for value in missing)
        or not isinstance(rationale, str)
    ):
        return _ordinary_failure(
            reason="INVALID_ENUM_OR_FIELD_TYPE",
            raw_text=raw_text,
            usage=usage,
        )

    if any(
        not _exact_substring(quote, evidence_text)
        for quote in quotes
    ):
        return _ordinary_failure(
            reason="NON_EXACT_QUOTE_OR_SPAN_ATTEMPT",
            raw_text=raw_text,
            usage=usage,
        )

    if candidate_status == "SUPPORTED_EXPLICITLY" and (
        not quotes or bool(missing)
    ):
        return _ordinary_failure(
            reason="SUPPORTED_WITH_MISSING_REQUIRED_BINDING_OR_ASPECT",
            raw_text=raw_text,
            usage=usage,
        )

    return {
        "raw_model_output": raw_text,
        "usage": dict(usage),
        "parsed_output": {
            "candidate_status": candidate_status,
            "supporting_quotes": list(quotes),
            "missing_aspects": list(missing),
            "rationale": rationale,
        },
        "effective_candidate_status": candidate_status,
        "arm_case_contract_pass": True,
        "invalid_execution": False,
        "ordinary_failure_reason": None,
        "non_exact_quote_or_span_attempt": False,
        "deterministic_guard": {
            "all_quotes_exact": True,
            "support_has_quote": bool(quotes)
            if candidate_status == "SUPPORTED_EXPLICITLY"
            else None,
            "support_has_no_missing_aspects": (
                not bool(missing)
                if candidate_status == "SUPPORTED_EXPLICITLY"
                else None
            ),
        },
    }


def _complete_from_statuses(
    statuses: dict[str, str],
) -> dict[str, Any]:
    complete = bool(statuses) and all(
        status == "SUPPORTED_EXPLICITLY"
        for status in statuses.values()
    )
    return {
        "obligation_statuses": dict(statuses),
        "local_complete": complete,
        "fallback_required": not complete,
    }


def _arm_gate(
    *,
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
    contract_pass_count = sum(
        bool(row["arm_case_contract_pass"])
        for row in case_results
    )
    non_exact_count = sum(
        bool(row["non_exact_quote_or_span_attempt"])
        for row in case_results
    )
    checks = {
        "all_7_cases_arm_case_contract_pass": (
            contract_pass_count == 7
        ),
        "reject_or_abstain_all_3_known_false_positives": (
            false_rejected == 3
        ),
        "accept_all_4_known_true_supports": (
            true_accepted == 4
        ),
        "zero_non_exact_supporting_quotes_or_spans": (
            non_exact_count == 0
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
    return {
        "contract_pass_count": contract_pass_count,
        "false_positive_rejection_or_abstention_count": false_rejected,
        "true_support_acceptance_count": true_accepted,
        "non_exact_quote_or_span_attempt_count": non_exact_count,
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
        return "BOTH_VALIDITY_MECHANISMS_CONFIRMED"
    if arm_a_pass:
        return "SLOT_GROUNDING_ONLY_CONFIRMED"
    if arm_b_pass:
        return "HOLISTIC_VERIFIER_ONLY_CONFIRMED"
    return "VALIDITY_PREMISE_REJECTED"


def _validate_design_contract(
    *,
    design: dict[str, Any],
) -> dict[str, Any]:
    if design.get("design_id") != DESIGN_ID:
        raise RuntimeError("unexpected validity design id")
    experiment = design.get("selected_discriminating_experiment") or {}
    if experiment.get("id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected validity experiment id")
    if design.get("status") != "FROZEN_DESIGN_ONLY":
        raise RuntimeError("unexpected frozen design status")

    model = experiment["frozen_inputs"]["local_model"]
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

    arm_a = experiment["arm_a_proof_carrying_slot_grounding"]
    arm_b = experiment["arm_b_holistic_tri_state_verifier"]
    if arm_a["id"] != ARM_A_ID or arm_b["id"] != ARM_B_ID:
        raise RuntimeError("unexpected validity arm identity")
    if arm_a["model_visible_obligation_fields"] != [
        "generic_information_need",
        "mandatory_slots",
        "accepted_stage_roles",
        "required_relation",
    ]:
        raise RuntimeError("Arm A visible fields changed")
    if arm_b["model_visible_obligation_fields"] != [
        "generic_information_need",
        "required_relation",
    ]:
        raise RuntimeError("Arm B visible fields changed")
    if "fail_closed_conditions" not in arm_a[
        "model_hidden_obligation_fields"
    ] or "fail_closed_conditions" not in arm_b[
        "model_hidden_obligation_fields"
    ]:
        raise RuntimeError("fail_closed_conditions leakage boundary changed")

    isolation = experiment["stateless_case_isolation_contract"]
    required_false = [
        "few_shot_examples",
        "prior_case_outputs_visible",
        "prior_case_prompts_visible",
        "extra_paper_context_visible",
        "extra_corpus_context_visible",
        "cross_case_conversation_state",
        "chat_history_reused",
        "kv_cache_reused_across_cases",
        "prompt_tuning_after_review",
        "retry_on_bad_model_output",
        "retry_on_scientific_failure",
    ]
    if any(bool(isolation[key]) for key in required_false):
        raise RuntimeError("stateless case-isolation contract changed")

    return experiment


def run_obligation_validity_microbenchmark(
    *,
    fixture: dict[str, Any],
    design: dict[str, Any],
    baseline_rescue_report: dict[str, Any],
    baseline_rescue_report_sha256: str,
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    experiment = _validate_design_contract(design=design)
    frozen = experiment["frozen_inputs"]
    if str(frozen["teacher_paper"]) != TEACHER_ITEM_ID:
        raise RuntimeError("unexpected teacher paper identity")
    if baseline_rescue_report_sha256 != BASELINE_RESCUE_REPORT_SHA256:
        raise RuntimeError("baseline rescue report sha256 mismatch")
    if baseline_rescue_report_sha256 != str(
        frozen["baseline_rescue_report_sha256"]
    ):
        raise RuntimeError("design baseline rescue report sha256 mismatch")

    teacher = _prepare_teacher_pool(
        fixture=fixture,
        workdir=workdir,
    )
    acquisition = dict(teacher["acquisition"])
    if str(teacher["item_id"]) != TEACHER_ITEM_ID:
        raise RuntimeError("unexpected acquired teacher item")
    if str(acquisition.get("sha256")) != str(
        frozen["teacher_pdf_sha256"]
    ):
        raise RuntimeError("teacher PDF sha256 mismatch")

    propositions = [
        dict(row)
        for row in teacher["page_pool"]
    ]
    propositions_by_id = {
        str(row["proposition_id"]): row
        for row in propositions
    }

    obligations = {
        str(row["id"]): dict(row)
        for row in experiment["runtime_obligations"]
    }
    false_cases = [
        {
            **dict(row),
            "evaluation_role": "KNOWN_FALSE_POSITIVE",
        }
        for row in experiment["case_set"]["known_false_positive_cases"]
    ]
    true_cases = [
        {
            **dict(row),
            "evaluation_role": "KNOWN_TRUE_SUPPORT",
        }
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
        for proposition_id in case["evidence_proposition_ids"]:
            if str(proposition_id) not in propositions_by_id:
                raise RuntimeError(
                    "case evidence proposition is not locatable"
                )

    starting_selected = [
        dict(row)
        for row in (
            baseline_rescue_report.get("selected_propositions")
            or []
        )
    ]
    starting_ids = [
        str(row["proposition_id"])
        for row in starting_selected
    ]
    prior_supported = sorted(
        str(value)
        for value in baseline_rescue_report[
            "rescued_teacher"
        ]["supported_atoms"]
    )
    frozen_controls = sorted(
        str(value)
        for value in baseline_rescue_report[
            "rescued_teacher"
        ]["frozen_control_atoms_retained"]
    )
    expected_prior = sorted(
        str(value)
        for value in experiment[
            "preservation_controls"
        ]["prior_supported_atom_ids"]
    )
    expected_controls = sorted(
        str(value)
        for value in experiment[
            "preservation_controls"
        ]["frozen_control_atom_ids"]
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

    model_cfg = frozen["local_model"]
    arms = {
        ARM_A_ID: experiment["arm_a_proof_carrying_slot_grounding"],
        ARM_B_ID: experiment["arm_b_holistic_tri_state_verifier"],
    }

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
            tensor_parallel_size=2,
        )
        model_init_seconds = time.perf_counter() - init_started

        for arm_id in (ARM_A_ID, ARM_B_ID):
            arm = arms[arm_id]
            for case in cases:
                obligation = obligations[
                    str(case["obligation_id"])
                ]
                proposition_ids = [
                    str(value)
                    for value in case["evidence_proposition_ids"]
                ]
                evidence_text = _evidence_text(
                    proposition_ids,
                    propositions_by_id,
                )
                (
                    system_prompt,
                    user_prompt,
                    visible_obligation,
                    visible_evidence,
                ) = _render_prompt(
                    arm=arm,
                    obligation=obligation,
                    evidence_text=evidence_text,
                )

                raw_text, usage = _model_once(
                    llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=float(model_cfg["temperature"]),
                    top_p=float(model_cfg["top_p"]),
                    seed=int(model_cfg["seed"]),
                )
                model_calls += 1
                total_prompt_tokens += int(
                    usage["prompt_tokens"]
                )
                total_output_tokens += int(
                    usage["output_tokens"]
                )
                total_inference_seconds += float(
                    usage["wall_time_seconds"]
                )

                if arm_id == ARM_A_ID:
                    guarded = _guard_arm_a(
                        raw_text=raw_text,
                        usage=usage,
                        obligation=obligation,
                        evidence_text=evidence_text,
                        arm=arm,
                    )
                else:
                    guarded = _guard_arm_b(
                        raw_text=raw_text,
                        usage=usage,
                        evidence_text=evidence_text,
                    )

                results_by_arm[arm_id].append(
                    {
                        "case_id": str(case["case_id"]),
                        "obligation_id": str(
                            case["obligation_id"]
                        ),
                        "evidence_proposition_ids": proposition_ids,
                        "evaluation_role": str(
                            case["evaluation_role"]
                        ),
                        "expected_posthoc": str(
                            case["expected_posthoc"]
                        ),
                        "visible_obligation": visible_obligation,
                        "visible_evidence": visible_evidence,
                        "system_prompt_sha256": _sha256_text(
                            system_prompt
                        ),
                        "user_prompt_sha256": _sha256_text(
                            user_prompt
                        ),
                        **guarded,
                    }
                )

    if model_calls != 14:
        raise RuntimeError(
            f"expected exactly 14 model calls, got {model_calls}"
        )

    case_lookup: dict[
        tuple[str, tuple[str, ...]],
        dict[str, Any],
    ] = {}
    for arm_id, arm_rows in results_by_arm.items():
        for row in arm_rows:
            key = (
                str(row["obligation_id"]),
                tuple(row["evidence_proposition_ids"]),
            )
            case_lookup[(arm_id, *key)] = row

    completeness_by_arm: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for arm_id in (ARM_A_ID, ARM_B_ID):
        baseline_statuses: dict[str, str] = {}
        true_statuses: dict[str, str] = {}
        for obligation_id, proposition_ids in experiment[
            "completeness_simulation"
        ]["baseline_false_complete_state"].items():
            if obligation_id == "required_outcome_for_any_safe_arm":
                continue
            row = case_lookup[
                (
                    arm_id,
                    str(obligation_id),
                    tuple(str(value) for value in proposition_ids),
                )
            ]
            baseline_statuses[str(obligation_id)] = str(
                row["effective_candidate_status"]
            )
        for obligation_id, proposition_ids in experiment[
            "completeness_simulation"
        ]["true_support_state"].items():
            if obligation_id == "required_outcome_for_any_passing_arm":
                continue
            row = case_lookup[
                (
                    arm_id,
                    str(obligation_id),
                    tuple(str(value) for value in proposition_ids),
                )
            ]
            true_statuses[str(obligation_id)] = str(
                row["effective_candidate_status"]
            )
        baseline_state = _complete_from_statuses(
            baseline_statuses
        )
        true_state = _complete_from_statuses(
            true_statuses
        )
        completeness_by_arm[arm_id] = {
            "baseline_false_complete_state": baseline_state,
            "true_support_state": true_state,
        }
        gates[arm_id] = _arm_gate(
            case_results=results_by_arm[arm_id],
            baseline_state=baseline_state,
            true_state=true_state,
            preservation=preservation,
        )

    hidden_markers = {
        "fail_closed_conditions",
        "case_id",
        "expected_posthoc",
        "teacher_atom",
        "legacy_anchor",
        "true/false",
    }
    visible_prompt_repr = _compact_json(
        [
            {
                "arm": arm_id,
                "visible_obligation": row["visible_obligation"],
                "visible_evidence": row["visible_evidence"],
            }
            for arm_id in (ARM_A_ID, ARM_B_ID)
            for row in results_by_arm[arm_id]
        ]
    ).lower()
    hidden_marker_absence = not any(
        marker in visible_prompt_repr
        for marker in hidden_markers
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
            baseline_rescue_report.get(
                "neighborhood_rescue_decision"
            )
            == "LOCAL_RESCUE_CONFIRMED"
        ),
        "baseline_selected_count_pass": len(starting_ids) == 28,
        "baseline_selected_unique_pass": (
            len(set(starting_ids)) == 28
        ),
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
        "hidden_evaluation_fields_absent_from_model_visible_input": (
            hidden_marker_absence
        ),
        "stateless_one_request_per_arm_case_pass": (
            model_calls == 14
        ),
        "evidence_set_no_mutation_pass": bool(
            preservation["evidence_set_identity_unchanged"]
            and not preservation["selection_mutation_attempted"]
        ),
    }
    integrity_valid = all(integrity.values())

    outcome = _outcome(
        integrity_valid=integrity_valid,
        arm_a_pass=bool(
            gates[ARM_A_ID]["precision_first_pass"]
        ),
        arm_b_pass=bool(
            gates[ARM_B_ID]["precision_first_pass"]
        ),
    )

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_OBLIGATION_VALIDITY_PRECISION_MICROBENCHMARK",
        "design_id": DESIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "teacher_item_id": TEACHER_ITEM_ID,
        "teacher_pdf_sha256": str(
            acquisition.get("sha256")
        ),
        "model_identity": {
            "model_id": EXPECTED_MODEL_ID,
            "repository": "Qwen/Qwen3.5-4B",
            "revision": EXPECTED_MODEL_REVISION,
            "model_object_id": EXPECTED_MODEL_OBJECT_ID,
            "runtime_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
            "temperature": model_cfg["temperature"],
            "top_p": model_cfg["top_p"],
            "seed": model_cfg["seed"],
            "operational_max_output_tokens": (
                OPERATIONAL_MAX_OUTPUT_TOKENS
            ),
            "prefix_caching_enabled": False,
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
                "completeness_simulation": completeness_by_arm[
                    ARM_A_ID
                ],
                "scientific_gate": gates[ARM_A_ID],
            },
            ARM_B_ID: {
                "cases": results_by_arm[ARM_B_ID],
                "completeness_simulation": completeness_by_arm[
                    ARM_B_ID
                ],
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
