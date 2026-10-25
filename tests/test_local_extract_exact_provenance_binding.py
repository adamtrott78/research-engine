from __future__ import annotations

import json

from research_engine.local_extract_exact_provenance_binding import (
    ARM_A_ID,
    ARM_B_ID,
    OPAQUE_EVIDENCE_ID,
    _arm_gate,
    _complete_from_rows,
    _guard_arm_a,
    _guard_arm_b,
    _outcome,
    _render_prompt,
    _strict_json_object,
)


def _usage() -> dict:
    return {
        "wall_time_seconds": 0.1,
        "prompt_tokens": 10,
        "output_tokens": 10,
        "generated_text_sha256": "0" * 64,
        "finish_reason": "stop",
    }


def _experiment() -> dict:
    return {
        "shared_prompt_and_evidence_contract": {
            "model_visible_obligation_fields": [
                "generic_information_need",
                "required_relation",
            ],
            "evidence_json_serialization": {
                "encoding": "UTF-8",
                "exact_visible_object": {
                    "source_authority": "AUTHORITATIVE_PDF_TEXT",
                    "evidence_id": "E0",
                    "text": "<exact authoritative proposition text>",
                },
                "ensure_ascii": False,
                "separators": [",", ":"],
            },
            "obligation_json_serialization": {
                "encoding": "UTF-8",
                "form": (
                    "compact JSON object with keys in exactly "
                    "model_visible_obligation_fields order"
                ),
                "ensure_ascii": False,
                "separators": [",", ":"],
            },
        }
    }


def _obligation() -> dict:
    return {
        "id": "hidden-id",
        "generic_information_need": "What does the evidence explicitly establish?",
        "required_relation": "The required relation must be explicit.",
    }


def _arm_a() -> dict:
    return {
        "id": ARM_A_ID,
        "prompt_contract": {
            "system_template": "SYSTEM A",
            "user_template": (
                "OBLIGATION_JSON\n{obligation_json}\n\n"
                "EVIDENCE_JSON\n{evidence_json}\n\n"
                "OUTPUT_SCHEMA_JSON\n"
                '{"candidate_status":"SUPPORTED_EXPLICITLY|NOT_SUPPORTED|'
                'INSUFFICIENT_OR_AMBIGUOUS","missing_aspects":[],'
                '"rationale":"brief"}'
            ),
        },
    }


def _arm_b() -> dict:
    return {
        "id": ARM_B_ID,
        "prompt_contract": {
            "system_template": "SYSTEM B",
            "user_template": (
                "OBLIGATION_JSON\n{obligation_json}\n\n"
                "EVIDENCE_JSON\n{evidence_json}\n\n"
                "OUTPUT_SCHEMA_JSON\n"
                '{"candidate_status":"SUPPORTED_EXPLICITLY|NOT_SUPPORTED|'
                'INSUFFICIENT_OR_AMBIGUOUS","supporting_evidence_ids":["E0"],'
                '"missing_aspects":[],"rationale":"brief"}'
            ),
        },
    }


def _preservation() -> dict:
    return {
        "evidence_set_identity_unchanged": True,
        "prior_supported_retention": "9/9",
        "frozen_control_retention": "7/7",
    }


def _row(
    role: str,
    status: str,
    proposition_id: str,
    *,
    arm_id: str,
) -> dict:
    supported = status == "SUPPORTED_EXPLICITLY"
    parsed = {
        "candidate_status": status,
        "missing_aspects": [],
        "rationale": "r",
    }
    if arm_id == ARM_B_ID:
        parsed["supporting_evidence_ids"] = (
            [OPAQUE_EVIDENCE_ID] if supported else []
        )
    return {
        "evaluation_role": role,
        "effective_candidate_status": status,
        "arm_case_contract_pass": True,
        "evidence_proposition_ids": [proposition_id],
        "attached_evidence_ids": (
            [OPAQUE_EVIDENCE_ID] if supported else []
        ),
        "attached_proposition_ids": (
            [proposition_id] if supported else []
        ),
        "exact_provenance_valid": supported,
        "parsed_output": parsed,
    }


def test_prompt_visibility_is_exact_and_e0_is_opaque() -> None:
    system, user, visible_obligation, visible_evidence = _render_prompt(
        experiment=_experiment(),
        arm=_arm_a(),
        obligation=_obligation(),
        evidence_text="Authoritative proposition text.",
    )
    assert system == "SYSTEM A"
    assert visible_obligation == {
        "generic_information_need": (
            "What does the evidence explicitly establish?"
        ),
        "required_relation": "The required relation must be explicit.",
    }
    assert "hidden-id" not in user
    assert visible_evidence == {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "evidence_id": "E0",
        "text": "Authoritative proposition text.",
    }
    assert '"evidence_id":"E0"' in user


def test_strict_json_object_rejects_prefix_and_suffix() -> None:
    assert _strict_json_object(' {"x":1}\n') == {"x": 1}
    for text in ('prefix {"x":1}', '{"x":1} suffix'):
        try:
            _strict_json_object(text)
        except Exception:
            pass
        else:
            raise AssertionError("prefix/suffix text must be rejected")


def test_arm_a_supported_status_gets_deterministic_exact_provenance() -> None:
    raw = json.dumps(
        {
            "candidate_status": "SUPPORTED_EXPLICITLY",
            "missing_aspects": [],
            "rationale": "The relation is explicit.",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        exact_proposition_id="page:p4:s15",
    )
    assert row["arm_case_contract_pass"] is True
    assert row["effective_candidate_status"] == "SUPPORTED_EXPLICITLY"
    assert row["attached_evidence_ids"] == ["E0"]
    assert row["attached_proposition_ids"] == ["page:p4:s15"]
    assert row["exact_provenance_valid"] is True
    assert row["provenance_attachment_mode"] == (
        "DETERMINISTIC_RUNTIME_E0_RESOLUTION"
    )


def test_arm_a_non_support_gets_no_provenance() -> None:
    raw = json.dumps(
        {
            "candidate_status": "NOT_SUPPORTED",
            "missing_aspects": ["explicit relation"],
            "rationale": "Not explicit.",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        exact_proposition_id="page:p1:s10",
    )
    assert row["arm_case_contract_pass"] is True
    assert row["effective_candidate_status"] == "NOT_SUPPORTED"
    assert row["attached_evidence_ids"] == []
    assert row["attached_proposition_ids"] == []
    assert row["exact_provenance_valid"] is False


def test_arm_a_support_with_missing_aspect_fails_closed() -> None:
    raw = json.dumps(
        {
            "candidate_status": "SUPPORTED_EXPLICITLY",
            "missing_aspects": ["source domain"],
            "rationale": "Incomplete.",
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        exact_proposition_id="page:p4:s15",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["invalid_execution"] is False
    assert row["effective_candidate_status"] == (
        "INSUFFICIENT_OR_AMBIGUOUS"
    )
    assert row["attached_proposition_ids"] == []


def test_arm_a_forbidden_quote_field_is_unknown_and_fails_closed() -> None:
    raw = json.dumps(
        {
            "candidate_status": "SUPPORTED_EXPLICITLY",
            "missing_aspects": [],
            "rationale": "Explicit.",
            "supporting_quotes": ["some quote"],
        }
    )
    row = _guard_arm_a(
        raw_text=raw,
        usage=_usage(),
        exact_proposition_id="page:p4:s15",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["ordinary_failure_reason"] == (
        "SCHEMA_MISSING_OR_UNKNOWN_FIELDS"
    )


def test_arm_b_supported_requires_exact_e0_and_resolves_provenance() -> None:
    raw = json.dumps(
        {
            "candidate_status": "SUPPORTED_EXPLICITLY",
            "supporting_evidence_ids": ["E0"],
            "missing_aspects": [],
            "rationale": "Explicit.",
        }
    )
    row = _guard_arm_b(
        raw_text=raw,
        usage=_usage(),
        exact_proposition_id="page:p2:s17",
    )
    assert row["arm_case_contract_pass"] is True
    assert row["attached_evidence_ids"] == ["E0"]
    assert row["attached_proposition_ids"] == ["page:p2:s17"]
    assert row["exact_provenance_valid"] is True


def test_arm_b_unknown_or_missing_pointer_fails_closed() -> None:
    for ids in ([], ["E1"], ["E0", "E0"]):
        raw = json.dumps(
            {
                "candidate_status": "SUPPORTED_EXPLICITLY",
                "supporting_evidence_ids": ids,
                "missing_aspects": [],
                "rationale": "Explicit.",
            }
        )
        row = _guard_arm_b(
            raw_text=raw,
            usage=_usage(),
            exact_proposition_id="page:p2:s17",
        )
        assert row["arm_case_contract_pass"] is False
        assert row["invalid_execution"] is False
        assert row["attached_proposition_ids"] == []


def test_arm_b_non_support_must_have_empty_pointer_list() -> None:
    good = json.dumps(
        {
            "candidate_status": "NOT_SUPPORTED",
            "supporting_evidence_ids": [],
            "missing_aspects": ["explicit relation"],
            "rationale": "Not explicit.",
        }
    )
    row = _guard_arm_b(
        raw_text=good,
        usage=_usage(),
        exact_proposition_id="page:p1:s09",
    )
    assert row["arm_case_contract_pass"] is True
    assert row["attached_proposition_ids"] == []

    bad = json.dumps(
        {
            "candidate_status": "NOT_SUPPORTED",
            "supporting_evidence_ids": ["E0"],
            "missing_aspects": ["explicit relation"],
            "rationale": "Not explicit.",
        }
    )
    row = _guard_arm_b(
        raw_text=bad,
        usage=_usage(),
        exact_proposition_id="page:p1:s09",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["attached_proposition_ids"] == []


def test_malformed_output_is_ordinary_failure_not_invalid_execution() -> None:
    row = _guard_arm_b(
        raw_text="{bad json",
        usage=_usage(),
        exact_proposition_id="page:p1:s09",
    )
    assert row["arm_case_contract_pass"] is False
    assert row["invalid_execution"] is False
    assert row["effective_candidate_status"] == (
        "INSUFFICIENT_OR_AMBIGUOUS"
    )
    assert row["attached_proposition_ids"] == []


def test_completeness_requires_support_and_exact_provenance() -> None:
    supported = {
        "effective_candidate_status": "SUPPORTED_EXPLICITLY",
        "exact_provenance_valid": True,
        "attached_proposition_ids": ["page:p2:s35"],
    }
    unsupported = {
        "effective_candidate_status": "INSUFFICIENT_OR_AMBIGUOUS",
        "exact_provenance_valid": False,
        "attached_proposition_ids": [],
    }
    complete = _complete_from_rows({"a": supported, "b": supported})
    assert complete["local_complete"] is True
    assert complete["fallback_required"] is False

    incomplete = _complete_from_rows({"a": supported, "b": unsupported})
    assert incomplete["local_complete"] is False
    assert incomplete["fallback_required"] is True

    provenance_invalid = dict(supported)
    provenance_invalid["exact_provenance_valid"] = False
    incomplete = _complete_from_rows({"a": provenance_invalid})
    assert incomplete["local_complete"] is False


def test_precision_gate_requires_semantics_and_provenance() -> None:
    for arm_id in (ARM_A_ID, ARM_B_ID):
        rows = [
            _row(
                "KNOWN_FALSE_POSITIVE",
                "NOT_SUPPORTED",
                f"page:false:{i}",
                arm_id=arm_id,
            )
            for i in range(3)
        ] + [
            _row(
                "KNOWN_TRUE_SUPPORT",
                "SUPPORTED_EXPLICITLY",
                f"page:true:{i}",
                arm_id=arm_id,
            )
            for i in range(4)
        ]
        gate = _arm_gate(
            arm_id=arm_id,
            case_results=rows,
            baseline_state={
                "local_complete": False,
                "fallback_required": True,
            },
            true_state={
                "local_complete": True,
                "fallback_required": False,
            },
            preservation=_preservation(),
        )
        assert gate["precision_first_pass"] is True

        rows[-1]["attached_proposition_ids"] = []
        rows[-1]["exact_provenance_valid"] = False
        gate = _arm_gate(
            arm_id=arm_id,
            case_results=rows,
            baseline_state={
                "local_complete": False,
                "fallback_required": True,
            },
            true_state={
                "local_complete": True,
                "fallback_required": False,
            },
            preservation=_preservation(),
        )
        assert gate["precision_first_pass"] is False


def test_outcome_partition() -> None:
    assert _outcome(
        integrity_valid=False,
        arm_a_pass=True,
        arm_b_pass=True,
    ) == "INVALID_EXECUTION"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=True,
        arm_b_pass=True,
    ) == "BOTH_EXACT_PROVENANCE_INTERFACES_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=True,
        arm_b_pass=False,
    ) == "DETERMINISTIC_ATTACHMENT_ONLY_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=False,
        arm_b_pass=True,
    ) == "OPAQUE_POINTER_ONLY_CONFIRMED"
    assert _outcome(
        integrity_valid=True,
        arm_a_pass=False,
        arm_b_pass=False,
    ) == "EXACT_PROVENANCE_BINDING_PREMISE_REJECTED"
