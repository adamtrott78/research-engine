from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from prototypes.semantic_control.decision import (
    evaluate_candidate_action,
    make_candidate_action,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None:
        parsed=parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _historical_policy_observation(case_id: str, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version":1,
        "observation_id":f"historical-policy:{case_id}",
        "layer":"NORMATIVE",
        "subject":{
            "entity_type":"WorkloadClass",
            "entity_id":policy["workload_class"],
        },
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":policy["decision"]},
        "assertion_state":"ACTIVE",
        "provenance":[{
            "source_id":"historical-pinned-workload-policy",
            "source_kind":"POLICY_ARTIFACT",
            "policy_id":policy["policy_id"],
            "policy_epoch":policy["policy_epoch"],
        }],
        "verification":{
            "state":"AUTHORITATIVE_RECORD",
            "basis":["frozen historical replay fixture"],
        },
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":{
            "policy_id":policy["policy_id"],
            "policy_epoch":policy["policy_epoch"],
            "historical_replay":True,
        },
    }


def replay_historical_decision(case: dict[str, Any]) -> dict[str, Any]:
    historical=deepcopy(case["historical"])
    policy=historical.get("policy")
    workload_class=historical["candidate"]["workload_class"]
    candidate=make_candidate_action(
        candidate_action_id=historical["candidate"]["candidate_action_id"],
        action_type=historical["candidate"]["action_type"],
        target={"workload_class":workload_class},
        rationale_observation_ids=[f'historical-rationale:{case["case_id"]}'],
        preconditions=[
            {
                "precondition_id":item["precondition_id"],
                "description":"Frozen historical precondition.",
            }
            for item in historical["preconditions"]
        ],
        required_capabilities=[workload_class],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["historical pinned policy context"],
        provenance=["historical-replay-fixture"],
    )
    observations=(
        [_historical_policy_observation(case["case_id"],policy)]
        if policy is not None
        else []
    )
    evaluation=evaluate_candidate_action(
        candidate=candidate,
        observations=observations,
        semantic_manifest_id=historical["semantic_manifest_id"],
        knowledge_checkpoint_ids=historical["knowledge_checkpoint_ids"],
        precondition_results=[
            {
                "precondition_id":item["precondition_id"],
                "result":item["result"],
                "evidence_ids":[f'historical-precondition:{item["precondition_id"]}'],
            }
            for item in historical["preconditions"]
        ],
        proposed_executor="historical-replay-executor",
    )
    result={
        "case_id":case["case_id"],
        "historical_input_sha256":_sha256(historical),
        "divergent_current_state_sha256":_sha256(case["divergent_current_state"]),
        "decision":evaluation["decision"],
        "expected_decision":case["replay_expected"],
        "matches_expected":evaluation["decision"]==case["replay_expected"],
        "semantic_manifest_id":evaluation["semantic_manifest_id"],
        "knowledge_checkpoint_ids":evaluation["knowledge_checkpoint_ids"],
        "authority_context":evaluation["authority_context"],
        "precondition_results":evaluation["precondition_results"],
        "reasons":evaluation["reasons"],
        "current_state_used_for_decision":False,
        "execution_receipt_id":evaluation["execution_receipt_id"],
    }
    return result


def replay_historical_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [replay_historical_decision(case) for case in cases]


def temporal_values_as_of(case: dict[str, Any], as_of: str) -> list[Any]:
    instant=_parse_time(as_of)
    assert instant is not None
    selected=[]
    for version in case["versions"]:
        valid_from=_parse_time(version.get("valid_from"))
        valid_to=_parse_time(version.get("valid_to"))
        if valid_from is not None and instant < valid_from:
            continue
        if valid_to is not None and instant >= valid_to:
            continue
        selected.append((version["observation_id"],deepcopy(version["value"])))
    selected.sort(key=lambda item:item[0])
    return [value for _,value in selected]


def replay_temporal_case(case: dict[str, Any]) -> dict[str, Any]:
    answers={
        instant:temporal_values_as_of(case,instant)
        for instant in sorted(case["expected"])
    }
    return {
        "case_id":case["case_id"],
        "answers":answers,
        "expected":deepcopy(case["expected"]),
        "matches_expected":answers==case["expected"],
        "history_preserved":len(case["versions"])>=2,
    }


def contradiction_view(case: dict[str, Any]) -> dict[str, Any]:
    observations=sorted(
        deepcopy(case["observations"]),
        key=lambda item:item["observation_id"],
    )
    ids=[item["observation_id"] for item in observations]
    for item in observations:
        missing=set(item.get("contradicts",[]))-set(ids)
        if missing:
            raise ValueError(
                f'{item["observation_id"]} references missing contradictions: '
                f"{sorted(missing)}"
            )
    return {
        "case_id":case["case_id"],
        "active_observation_ids":ids,
        "values":[item["value"] for item in observations],
        "global_truth_resolution":"NONE",
        "contradictions_preserved":True,
    }


def unknown_semantics(case: dict[str, Any]) -> dict[str, Any]:
    observation=case["observation"]
    if observation["assertion_state"]!="UNKNOWN":
        raise ValueError("unknown semantics fixture must use UNKNOWN assertion_state")
    return {
        "case_id":case["case_id"],
        "assertion_state":"UNKNOWN",
        "value":deepcopy(observation["value"]),
        "coerced_to_false":False,
    }
