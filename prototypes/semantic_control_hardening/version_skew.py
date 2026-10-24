from __future__ import annotations

from copy import deepcopy
from typing import Any


def evaluate_policy_context(
    *,
    projected_policy: dict[str, Any] | None,
    required_policy: dict[str, Any] | None = None,
    required_policy_epoch: int | None = None,
) -> dict[str, Any]:
    if required_policy is not None:
        if projected_policy is None:
            return {
                "decision":"ABSTAIN",
                "reason_code":"REQUIRED_AUTHORITY_UNAVAILABLE",
            }
        if projected_policy.get("epoch") != required_policy.get("epoch"):
            current_decision=required_policy.get("decision")
            return {
                "decision":"DENY" if current_decision=="DENY" else "ABSTAIN",
                "reason_code":"STALE_AUTHORITY_CONTEXT",
            }
        return {
            "decision":"ALLOW" if required_policy.get("decision") in {"ALLOW","EXISTING_P5_CAPABILITY"} else "DENY",
            "reason_code":"CURRENT_AUTHORITY_EVALUATED",
        }

    if required_policy_epoch is not None:
        if projected_policy is None or projected_policy.get("epoch") != required_policy_epoch:
            return {
                "decision":"ABSTAIN",
                "reason_code":"REQUIRED_AUTHORITY_UNAVAILABLE",
            }

    if projected_policy is None:
        return {
            "decision":"ABSTAIN",
            "reason_code":"REQUIRED_AUTHORITY_UNAVAILABLE",
        }

    decision=projected_policy.get("decision")
    return {
        "decision":"ALLOW" if decision in {"ALLOW","EXISTING_P5_CAPABILITY"} else ("DENY" if decision=="DENY" else "ABSTAIN"),
        "reason_code":"PROJECTED_AUTHORITY_CURRENT",
    }


def validate_task_checkpoint(
    projected_task_checkpoint: str,
    required_task_checkpoint: str,
) -> dict[str, Any]:
    if projected_task_checkpoint != required_task_checkpoint:
        return {
            "decision":"ABSTAIN",
            "reason_code":"TASK_STATE_STALE",
        }
    return {
        "decision":"ALLOW",
        "reason_code":"TASK_STATE_CURRENT",
    }


def validate_receipt_binding(
    *,
    expected_evaluation_id: str,
    expected_candidate_action_id: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    valid=(
        receipt.get("evaluation_id")==expected_evaluation_id
        and receipt.get("candidate_action_id")==expected_candidate_action_id
    )
    return {
        "receipt_status":"VALID_BINDING" if valid else "INVALID_BINDING",
        "authority_effect":"NONE",
    }


def validate_source_blob(
    *,
    expected_blob_sha: str,
    observed_blob_sha: str,
) -> dict[str, Any]:
    if expected_blob_sha != observed_blob_sha:
        return {
            "decision":"ABSTAIN",
            "reason_code":"SOURCE_IDENTITY_DRIFT",
        }
    return {
        "decision":"ALLOW",
        "reason_code":"SOURCE_IDENTITY_MATCH",
    }


def evaluate_repository_agreement(
    *,
    source_a: dict[str, Any],
    source_b: dict[str, Any],
) -> dict[str, Any]:
    if source_a.get("value") != source_b.get("value"):
        return {
            "decision":"ABSTAIN",
            "reason_code":"CROSS_SOURCE_DISAGREEMENT",
            "sources":[deepcopy(source_a),deepcopy(source_b)],
        }
    return {
        "decision":"ALLOW",
        "reason_code":"CROSS_SOURCE_AGREEMENT",
        "sources":[deepcopy(source_a),deepcopy(source_b)],
    }


def evaluate_permission_claim(
    *,
    observation_layer: str,
    claimed_policy_decision: str,
) -> dict[str, Any]:
    if observation_layer != "NORMATIVE":
        return {
            "decision":"ABSTAIN",
            "reason_code":"NON_AUTHORITATIVE_PERMISSION_CLAIM",
        }
    return {
        "decision":"ALLOW" if claimed_policy_decision in {"ALLOW","EXISTING_P5_CAPABILITY"} else ("DENY" if claimed_policy_decision=="DENY" else "ABSTAIN"),
        "reason_code":"AUTHORITATIVE_PERMISSION_CLAIM",
    }


def validate_projection_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]]={}
    for record in records:
        observation_id=record["observation_id"]
        if observation_id in by_id:
            return {
                "projection_status":"FAIL_CLOSED",
                "reason_code":"DUPLICATE_OBSERVATION_ID",
                "authority_effect":"NONE",
            }
        by_id[observation_id]=deepcopy(record)
    return {
        "projection_status":"VALID",
        "reason_code":"UNIQUE_OBSERVATION_IDS",
        "authority_effect":"NONE",
    }


def evaluate_frozen_adversary(case: dict[str, Any]) -> dict[str, Any]:
    case_id=case["case_id"]

    if case_id=="V1_STALE_ALLOW_CURRENT_DENY":
        return {
            "case_id":case_id,
            **evaluate_policy_context(
                projected_policy=case["projected_policy"],
                required_policy=case["required_authority"],
            ),
        }

    if case_id=="V2_POLICY_EPOCH_MISMATCH_WITHOUT_CURRENT_RECORD":
        return {
            "case_id":case_id,
            **evaluate_policy_context(
                projected_policy=case["projected_policy"],
                required_policy=None,
                required_policy_epoch=case["required_policy_epoch"],
            ),
        }

    if case_id=="V3_TASK_CHECKPOINT_DRIFT":
        return {
            "case_id":case_id,
            **validate_task_checkpoint(
                case["projected_task_checkpoint"],
                case["required_task_checkpoint"],
            ),
        }

    if case_id=="V4_RECEIPT_BINDING_MISMATCH":
        return {
            "case_id":case_id,
            **validate_receipt_binding(
                expected_evaluation_id=case["evaluation_id"],
                expected_candidate_action_id=case["receipt"]["candidate_action_id"],
                receipt=case["receipt"],
            ),
        }

    if case_id=="V5_SOURCE_BLOB_DRIFT":
        return {
            "case_id":case_id,
            **validate_source_blob(
                expected_blob_sha=case["expected_blob_sha"],
                observed_blob_sha=case["observed_blob_sha"],
            ),
        }

    if case_id=="V6_REPOSITORY_DISAGREEMENT":
        return {
            "case_id":case_id,
            **evaluate_repository_agreement(
                source_a=case["source_a"],
                source_b=case["source_b"],
            ),
        }

    if case_id=="V7_FAKE_EPISTEMIC_ALLOW":
        return {
            "case_id":case_id,
            **evaluate_permission_claim(
                observation_layer=case["observation_layer"],
                claimed_policy_decision=case["claimed_policy_decision"],
            ),
        }

    if case_id=="V8_CORRUPT_PROJECTION_DUPLICATE_ID":
        return {
            "case_id":case_id,
            **validate_projection_records([
                {"observation_id":"duplicate","value":"first"},
                {"observation_id":"duplicate","value":"second"},
            ]),
        }

    raise ValueError(f"unsupported adversary case: {case_id}")
