from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


AUTHORITY_RULE = "SEMANTIC_CONTEXT_NEVER_GRANTS_AUTHORITY"
_ALLOW_POLICY_VALUES = {"ALLOW", "EXISTING_P5_CAPABILITY"}


def make_candidate_action(
    *,
    candidate_action_id: str,
    action_type: str,
    target: dict[str, Any],
    rationale_observation_ids: list[str],
    preconditions: list[dict[str, str]],
    required_capabilities: list[str],
    requested_effects: list[str],
    required_authority_context: list[str],
    provenance: list[Any],
) -> dict[str, Any]:
    """Construct a proposal that is structurally incapable of authorizing itself."""
    if not rationale_observation_ids:
        raise ValueError("candidate action requires rationale observations")
    if not required_authority_context:
        raise ValueError("candidate action requires authority context declaration")
    return {
        "schema_version": 1,
        "candidate_action_id": candidate_action_id,
        "action_type": action_type,
        "target": deepcopy(target),
        "rationale_observation_ids": list(rationale_observation_ids),
        "preconditions": deepcopy(preconditions),
        "required_capabilities": sorted(set(required_capabilities)),
        "requested_effects": sorted(set(requested_effects)),
        "required_authority_context": list(required_authority_context),
        "authorization_state": "UNDECIDED",
        "provenance": deepcopy(provenance),
    }


def _policy_subject_matches(
    observation: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    target = candidate["target"]
    subject = observation["subject"]

    if subject.get("entity_type") == "WorkloadClass":
        return target.get("workload_class") == subject.get("entity_id")

    if subject.get("entity_type") == "ActionType":
        return candidate["action_type"] == subject.get("entity_id")

    return False


def _authority_source_id(observation: dict[str, Any]) -> str:
    provenance = observation.get("provenance", [])
    if not provenance:
        raise ValueError("authoritative policy observation has no provenance")
    source = provenance[0]
    if isinstance(source, str):
        return source
    if isinstance(source, dict) and source.get("source_id"):
        return source["source_id"]
    raise ValueError("authoritative policy provenance lacks source_id")


def _eligible_authority_observations(
    candidate: dict[str, Any],
    observations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("layer") != "NORMATIVE":
            continue
        if observation.get("predicate") != "policyDecision":
            continue
        if observation.get("assertion_state") != "ACTIVE":
            continue
        if observation.get("verification", {}).get("state") != "AUTHORITATIVE_RECORD":
            continue
        if not _policy_subject_matches(observation, candidate):
            continue
        eligible.append(deepcopy(observation))
    return sorted(eligible, key=lambda item: item["observation_id"])


def _normalize_precondition_results(
    candidate: dict[str, Any],
    supplied: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    declared = {
        item["precondition_id"]
        for item in candidate.get("preconditions", [])
    }
    supplied_list = [deepcopy(item) for item in supplied]
    supplied_ids = [item["precondition_id"] for item in supplied_list]
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("duplicate precondition result")
    unknown = set(supplied_ids) - declared
    if unknown:
        raise ValueError(f"unknown precondition result: {sorted(unknown)}")

    by_id = {item["precondition_id"]: item for item in supplied_list}
    normalized: list[dict[str, Any]] = []
    for precondition_id in sorted(declared):
        if precondition_id not in by_id:
            normalized.append(
                {
                    "precondition_id": precondition_id,
                    "result": "UNKNOWN",
                    "evidence_ids": [],
                }
            )
            continue
        item = by_id[precondition_id]
        result = item["result"]
        if result not in {"PASS", "FAIL", "UNKNOWN"}:
            raise ValueError(f"invalid precondition result: {result}")
        normalized.append(
            {
                "precondition_id": precondition_id,
                "result": result,
                "evidence_ids": sorted(set(item.get("evidence_ids", []))),
            }
        )
    return normalized


def evaluate_candidate_action(
    *,
    candidate: dict[str, Any],
    observations: Iterable[dict[str, Any]],
    semantic_manifest_id: str,
    knowledge_checkpoint_ids: list[str],
    precondition_results: Iterable[dict[str, Any]],
    proposed_executor: str | None = None,
) -> dict[str, Any]:
    """Evaluate a proposal without invoking any executor or fabricating a receipt."""
    if candidate.get("authorization_state") != "UNDECIDED":
        raise ValueError("candidate action must remain UNDECIDED")

    normalized_preconditions = _normalize_precondition_results(
        candidate,
        precondition_results,
    )
    authorities = _eligible_authority_observations(candidate, observations)

    authority_context: list[dict[str, Any]] = []
    for observation in authorities:
        attributes = observation.get("attributes", {})
        authority_context.append(
            {
                "authority_kind": "POLICY_ARTIFACT",
                "source_id": _authority_source_id(observation),
                "authority_id": attributes.get("policy_id", "UNKNOWN_POLICY"),
                "authority_version": (
                    f'epoch:{attributes["policy_epoch"]}'
                    if "policy_epoch" in attributes
                    else "UNVERSIONED"
                ),
                "observation_id": observation["observation_id"],
            }
        )

    deny_authorities = [
        observation
        for observation in authorities
        if observation.get("object", {}).get("value") == "DENY"
    ]
    allow_authorities = [
        observation
        for observation in authorities
        if observation.get("object", {}).get("value") in _ALLOW_POLICY_VALUES
    ]

    if deny_authorities:
        decision = "DENY"
        reasons = [
            "explicit authoritative policy denies the candidate action"
        ]
    elif not allow_authorities:
        decision = "ABSTAIN"
        reasons = [
            "no authoritative policy context permits the candidate action"
        ]
    elif any(item["result"] == "FAIL" for item in normalized_preconditions):
        decision = "DENY"
        reasons = [
            "authoritative policy permits the action class but a required precondition failed"
        ]
    elif any(item["result"] == "UNKNOWN" for item in normalized_preconditions):
        decision = "ABSTAIN"
        reasons = [
            "authoritative policy permits the action class but a required precondition is unknown"
        ]
    else:
        decision = "ALLOW"
        reasons = [
            "authoritative policy permits the action class and all declared preconditions passed"
        ]

    return {
        "schema_version": 1,
        "evaluation_id": f'eval:{candidate["candidate_action_id"].split(":", 1)[-1]}',
        "candidate_action_id": candidate["candidate_action_id"],
        "semantic_manifest_id": semantic_manifest_id,
        "knowledge_checkpoint_ids": list(knowledge_checkpoint_ids),
        "authority_context": authority_context,
        "precondition_results": normalized_preconditions,
        "decision": decision,
        "reasons": reasons,
        "authority_rule": AUTHORITY_RULE,
        "selected_executor": proposed_executor if decision == "ALLOW" else None,
        "execution_receipt_id": None,
    }
