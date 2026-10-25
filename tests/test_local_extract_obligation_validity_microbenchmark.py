from __future__ import annotations

import json

from research_engine.local_extract_obligation_validity_microbenchmark import (
    ARM_A_ID,
    ARM_B_ID,
    _arm_gate,
    _complete_from_statuses,
    _guard_arm_a,
    _guard_arm_b,
    _outcome,
    _render_prompt,
    _strict_json_object,
    _visible_obligation,
)


def _obligation() -> dict:
    return {
        "id": "method_training_label_source",
        "generic_information_need": (
            "During supervised classifier training, what labels are used "
            "and what data/domain supplies those labels?"
        ),
        "mandatory_slots": [
            "activity_stage",
            "classifier_training_relation",
            "device_label_use",
            "label_source_domain",
        ],
        "accepted_stage_roles": ["CLASSIFIER_TRAINING"],
        "fail_closed_conditions": [
            "pre-training without labels is not classifier training with labels",
            "labels without explicit source-domain provenance are insufficient",
        ],
        "required_relation": (
            "The classifier-training action explicitly uses device labels "
            "and those labels are explicitly sourced from the source domain."
        ),
    }


def _arm_a() -> dict:
    return {
        "id": ARM_A_ID,
        "model_visible_obligation_fields": [
            "generic_information_need",
            "mandatory_slots",
            "accepted_stage_roles",
            "required_relation",
        ],
        "model_hidden_obligation_fields": [
            "id",
            "fail_closed_conditions",
        ],
        "prompt_contract": {
            "system_template": "SYSTEM",
            "user_template": (
                "OBLIGATION_JSON\n{obligation_json}\n\n"
                "EVIDENCE_JSON\n{evidence_json}\n\n"
                'OUTPUT_SCHEMA_JSON\n{"stage_role":"X"}'
            ),
            "obligation_json_serialization": {
                "encoding": "UTF-8",
                "form": (
                    "compact JSON object with keys in exactly "
                    "model_visible_obligation_fields order"
                ),
                "ensure_ascii": False,
                "separators": [",", ":"],
            },
            "evidence_json_serialization": {
                "encoding": "UTF-8",
                "exact_visible_object": {
                    "source_authority": "AUTHORITATIVE_PDF_TEXT",
                    "text": (
                        "<exact authoritative proposition text or exact "
                        "concatenation of the frozen evidence bundle texts "
                        "separated by a single newline>"
                    ),
                },
                "hidden_from_model": [
                    "case_id",
                    "expected_posthoc",
                ],
                "ensure_ascii": False,
                "separators": [",", ":"],
            },
        },
        "model_output_contract": {
            "stage_role": [
                "PRETRAINING",
                "CLASSIFIER_TRAINING",
                "TESTING",
                "EVALUATION",
                "ACQUISITION",
                "FAILURE_ANALYSIS",
                "RESULT_INTERPRETATION",
                "OTHER",
                "AMBIGUOUS",
            ],
            "relation_check": [
                "EXPLICITLY_LINKED",
                "NOT_LINKED",
                "AMBIGUOUS",
            ],
        },
    }


def _arm_b() -> dict:
    arm = _arm_a()
    arm["id"] = ARM_B_ID
    arm["model_visible_obligation_fields"] = [
        "generic_information_need",
        "required_relation",
    ]
    arm["model_hidden_obligation_fields"] = [
        "id",
        "mandatory_slots",
        "accepted_stage_roles",
        "fail_closed_conditions",
    ]
    return arm


def _usage() -> dict:
    return {
        "wall_time_seconds": 0.1,
        "prompt_tokens": 10,
        "output_tokens": 10,
        "generated_text_sha256": "0" * 64,
        "finish_reason": "stop",
    }


def _preservation() -> dict:
    return {
        "evidence_set_identity_unchanged": True,
        "prior_supported_retention": "9/9",
        "frozen_control_retention": "7/7",
    }


def test_visible_obligation_is_exact_ordered_projection() -> None:
    obligation = _obligation()
    fields = _arm_a()["model_visible_obligation_fields"]
    visible = _visible_obligation(obligation, fields)
    assert list(visible) == fields
    assert "id" not in visible
    assert "fail_closed_conditions" not in visible


def test_prompt_materialization_hides_fail_closed_conditions() -> None:
    system, user, visible, evidence = _render_prompt(
        arm=_arm_a(),
        obligation=_obligation(),
        evidence_text="The classifier uses source-domain device labels.",
    )
    assert system == "SYSTEM"
    assert list(visible) == _arm_a()["model_visible_obligation_fields"]
    assert "fail_closed_conditions" not in user
    assert "pre-training without labels" not in user
    assert evidence == {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "text": "The classifier uses source-domain device labels.",
    }
    assert '{"generic_information_need":' in user
    assert '{"source_authority":"AUTHORITATIVE_PDF_TEXT","text":' in user


def test_arm_b_visible_fields_exclude_slot_schema_and_stage_roles() -> None:
    _, user, visible, _ = _render_prompt(
        arm=_arm_b(),
        obligation=_obligation(),
        evidence_text="evidence",
    )
    assert list(visible) == [
        "generic_information_need",
        "required_relation",
    ]
    assert "mandatory_slots" not in user
    assert "accepted_stage_roles" not in user
    assert "fail_closed_conditions" not in user


def test_strict_json_object_rejects_prefix_suffix_text() -> None:
    assert _strict_json_object('  {"x":1}\n') == {"x": 1}
    try:
        _strict_json_object('prefix {"x":1}')
    except Exception:
        pass
    else:
        raise AssertionError("prefix text must be rejected")
    try:
        _strict_json_object('{"x":1} suffix')
    except Exception:
        pass
    else:
        raise AssertionError("suffix text must be rejected")


def test_arm_a_exact_supported_case_passes_guard() -> None:
    obligation = _obligation()
    evidence = (
        "We train the classifier using device labels from the source domain."
    )
    raw = json.dumps(
        {
            "stage_role": "CLASSIFIER_TRAINING",
            "slot_bindings": [
                {
                    "slot_id": "activity_stage",
                    "verbatim_span": "train the classifier",
                },
                {
                    "slot_id": "classifier_training_relation",
                    "verbatim_span": "train the classifier using",
                },
                {
                    "slot_id": "device_label_use",
                    "verbatim_span": "device labels",
                },
                {
                    "slot_id": "label_source_domain",
                    "verbatim_span": "source domain",
                },
            ],
            "relation_check": "EXPLICITLY_LINKED",
            "candidate_status": "SUPPORTED_EXPLICITLY",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        obligation=obligation,
        evidence_text=evidence,
        arm=_arm_a(),
    )
    assert row["arm_case_contract_pass"] is True
    assert row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
    assert row["non_exact_quote_or_span_attempt"] is False


def test_arm_a_wrong_stage_fails_closed_without_contract_failure() -> None:
    obligation = _obligation()
    evidence = (
        "The goal of pre-training is to learn a domain-invariant representation."
    )
    raw = json.dumps(
        {
            "stage_role": "PRETRAINING",
            "slot_bindings": [
                {
                    "slot_id": "activity_stage",
                    "verbatim_span": "pre-training",
                },
                {
                    "slot_id": "classifier_training_relation",
                    "verbatim_span": "pre-training",
                },
                {
                    "slot_id": "device_label_use",
                    "verbatim_span": "domain-invariant",
                },
                {
                    "slot_id": "label_source_domain",
                    "verbatim_span": "domain-invariant",
                },
            ],
            "relation_check": "AMBIGUOUS",
            "candidate_status": "SUPPORTED_EXPLICITLY",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        obligation=obligation,
        evidence_text=evidence,
        arm=_arm_a(),
    )
    assert row["arm_case_contract_pass"] is True
    assert row["effective_candidate_status"] == (
        "INSUFFICIENT_OR_AMBIGUOUS"
    )
    assert row["invalid_execution"] is False


def test_arm_a_non_exact_span_counts_against_arm_not_invalid_execution() -> None:
    obligation = _obligation()
    evidence = "We train the classifier using source-domain device labels."
    raw = json.dumps(
        {
            "stage_role": "CLASSIFIER_TRAINING",
            "slot_bindings": [
                {
                    "slot_id": "activity_stage",
                    "verbatim_span": "not in source",
                },
                {
                    "slot_id": "classifier_training_relation",
                    "verbatim_span": "train the classifier",
                },
                {
                    "slot_id": "device_label_use",
                    "verbatim_span": "device labels",
                },
                {
                    "slot_id": "label_source_domain",
                    "verbatim_span": "source-domain",
                },
            ],
            "relation_check": "EXPLICITLY_LINKED",
            "candidate_status": "SUPPORTED_EXPLICITLY",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        obligation=obligation,
        evidence_text=evidence,
        arm=_arm_a(),
    )
    assert row["arm_case_contract_pass"] is False
    assert row["invalid_execution"] is False
    assert row["effective_candidate_status"] == (
        "INSUFFICIENT_OR_AMBIGUOUS"
    )
    assert row["non_exact_quote_or_span_attempt"] is True


def test_arm_b_supported_requires_exact_quote_and_no_missing_aspects() -> None:
    evidence = "The trained classifier is applied to target data during testing."
    raw = json.dumps(
        {
            "candidate_status": "SUPPORTED_EXPLICITLY",
            "supporting_quotes": [
                "applied to target data during testing"
            ],
            "missing_aspects": [],
            "rationale": "The evidence explicitly states the test-domain application.",
        }
    )
    row = _guard_arm_b(
        raw_text=raw,
        usage=_usage(),
        evidence_text=evidence,
    )
    assert row["arm_case_contract_pass"] is True
    assert row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"


def test_arm_b_non_exact_quote_is_ordinary_arm_failure() -> None:
    raw = json.dumps(
        {
            "candidate_status": "NOT_SUPPORTED",
            "supporting_quotes": ["invented quote"],
            "missing_aspects": ["explicit target-domain testing"],
            "rationale": "Not explicit.",
        }
    )
    row = _guard_arm_b(
        raw_text=raw,
        usage=_usage(),
        evidence_text="Generic domain shift is discussed.",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["invalid_execution"] is False
    assert row["non_exact_quote_or_span_attempt"] is True


def test_malformed_json_fails_closed_without_invalid_execution() -> None:
    row = _guard_arm_b(
        raw_text="{bad json",
        usage=_usage(),
        evidence_text="evidence",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["invalid_execution"] is False
    assert row["effective_candidate_status"] == (
        "INSUFFICIENT_OR_AMBIGUOUS"
    )
    assert row["ordinary_failure_reason"].startswith("MALFORMED_JSON:")


def test_completeness_is_only_all_explicit_support() -> None:
    incomplete = _complete_from_statuses(
        {
            "a": "SUPPORTED_EXPLICITLY",
            "b": "INSUFFICIENT_OR_AMBIGUOUS",
        }
    )
    assert incomplete == {
        "obligation_statuses": {
            "a": "SUPPORTED_EXPLICITLY",
            "b": "INSUFFICIENT_OR_AMBIGUOUS",
        },
        "local_complete": False,
        "fallback_required": True,
    }
    complete = _complete_from_statuses(
        {
            "a": "SUPPORTED_EXPLICITLY",
            "b": "SUPPORTED_EXPLICITLY",
        }
    )
    assert complete["local_complete"] is True
    assert complete["fallback_required"] is False


def test_precision_gate_requires_all_seven_contract_valid_cases() -> None:
    rows = []
    for index in range(3):
        rows.append(
            {
                "evaluation_role": "KNOWN_FALSE_POSITIVE",
                "effective_candidate_status": "NOT_SUPPORTED",
                "arm_case_contract_pass": True,
                "non_exact_quote_or_span_attempt": False,
            }
        )
    for index in range(4):
        rows.append(
            {
                "evaluation_role": "KNOWN_TRUE_SUPPORT",
                "effective_candidate_status": "SUPPORTED_EXPLICITLY",
                "arm_case_contract_pass": True,
                "non_exact_quote_or_span_attempt": False,
            }
        )
    baseline = {
        "local_complete": False,
        "fallback_required": True,
    }
    truth = {
        "local_complete": True,
        "fallback_required": False,
    }
    gate = _arm_gate(
        case_results=rows,
        baseline_state=baseline,
        true_state=truth,
        preservation=_preservation(),
    )
    assert gate["precision_first_pass"] is True

    rows[0]["arm_case_contract_pass"] = False
    gate = _arm_gate(
        case_results=rows,
        baseline_state=baseline,
        true_state=truth,
        preservation=_preservation(),
    )
    assert gate["precision_first_pass"] is False


def test_outcome_partition_reserves_invalid_for_integrity_failure() -> None:
    assert _outcome(
        integrity_valid=False,
        arm_a_pass=True,
        arm_b_pass=True,
    ) == "INVALID_EXECUTION"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=True,
        arm_b_pass=True,
    ) == "BOTH_VALIDITY_MECHANISMS_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=True,
        arm_b_pass=False,
    ) == "SLOT_GROUNDING_ONLY_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=False,
        arm_b_pass=True,
    ) == "HOLISTIC_VERIFIER_ONLY_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=False,
        arm_b_pass=False,
    ) == "VALIDITY_PREMISE_REJECTED"
