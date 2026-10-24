from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from prototypes.semantic_control_hardening.version_skew import validate_receipt_binding


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _precondition_evidence_ids(evaluation: dict[str, Any]) -> list[str]:
    values={
        evidence_id
        for result in evaluation["precondition_results"]
        for evidence_id in result.get("evidence_ids",[])
    }
    return sorted(values)


def _authority_source_ids(evaluation: dict[str, Any]) -> list[str]:
    values={
        item["source_id"]
        for item in evaluation.get("authority_context",[])
        if item.get("source_id")
    }
    return sorted(values)


def build_decision_audit_chain(
    *,
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate.get("authorization_state")!="UNDECIDED":
        raise ValueError("candidate must remain UNDECIDED")
    if evaluation["candidate_action_id"]!=candidate["candidate_action_id"]:
        raise ValueError("evaluation candidate_action_id mismatch")

    receipt_binding={
        "status":"ABSENT",
        "receipt_id":None,
        "receipt_sha256":None,
        "authority_effect":"NONE",
    }
    if receipt is not None:
        binding=validate_receipt_binding(
            expected_evaluation_id=evaluation["evaluation_id"],
            expected_candidate_action_id=candidate["candidate_action_id"],
            receipt=receipt,
        )
        receipt_binding={
            "status":binding["receipt_status"],
            "receipt_id":receipt.get("receipt_id"),
            "receipt_sha256":canonical_sha256(receipt),
            "authority_effect":"NONE",
        }

    body={
        "schema_version":1,
        "candidate_action_id":candidate["candidate_action_id"],
        "semantic_manifest_id":evaluation["semantic_manifest_id"],
        "knowledge_checkpoint_ids":deepcopy(evaluation["knowledge_checkpoint_ids"]),
        "candidate_sha256":canonical_sha256(candidate),
        "authority_context":deepcopy(evaluation.get("authority_context",[])),
        "authority_context_sha256":canonical_sha256(evaluation.get("authority_context",[])),
        "authority_source_ids":_authority_source_ids(evaluation),
        "precondition_results":deepcopy(evaluation["precondition_results"]),
        "precondition_results_sha256":canonical_sha256(evaluation["precondition_results"]),
        "precondition_evidence_ids":_precondition_evidence_ids(evaluation),
        "decision_evaluation_id":evaluation["evaluation_id"],
        "decision_evaluation_sha256":canonical_sha256(evaluation),
        "decision":evaluation["decision"],
        "reasons":deepcopy(evaluation["reasons"]),
        "authority_rule":evaluation["authority_rule"],
        "selected_executor":evaluation.get("selected_executor"),
        "original_execution_receipt_id":evaluation.get("execution_receipt_id"),
        "receipt_binding":receipt_binding,
    }
    return {
        **body,
        "audit_chain_sha256":canonical_sha256(body),
    }


def verify_decision_audit_chain(
    *,
    chain: dict[str, Any],
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected=build_decision_audit_chain(
        candidate=candidate,
        evaluation=evaluation,
        receipt=receipt,
    )
    return {
        "valid":chain==expected,
        "expected_audit_chain_sha256":expected["audit_chain_sha256"],
        "observed_audit_chain_sha256":chain.get("audit_chain_sha256"),
        "receipt_binding_status":expected["receipt_binding"]["status"],
        "receipt_authority_effect":"NONE",
    }
