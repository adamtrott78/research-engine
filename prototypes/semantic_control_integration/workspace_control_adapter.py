from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_index(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = source_manifest["sources"]
    index = {item["source_id"]: item for item in sources}
    if len(index) != len(sources):
        raise ValueError("duplicate source_id in workspace-control source manifest")
    return index


def _validate_inputs(snapshot_inputs: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index = _source_index(source_manifest)
    seen: set[str] = set()
    for item in snapshot_inputs["inputs"]:
        source_id = item["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate workspace input source: {source_id}")
        seen.add(source_id)
        if source_id not in index:
            raise ValueError(f"unknown workspace source: {source_id}")
        expected = index[source_id]["git_blob_sha"]
        actual = item["git_blob_sha"]
        if actual != expected:
            raise ValueError(f"workspace source blob drift for {source_id}: {actual} != {expected}")
    return index


def _prov(source: dict[str, Any], commit_sha: str) -> list[dict[str, Any]]:
    return [{
        "source_kind": source["role"],
        "repository": "adamtrott78/workspace-control",
        "commit_sha": commit_sha,
        "path": source["path"],
        "git_blob_sha": source["git_blob_sha"],
        "source_id": source["source_id"],
    }]


def _obs(*, observation_id: str, layer: str, subject_type: str, subject_id: str, predicate: str, value: Any, provenance: list[dict[str, Any]], attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version":1,
        "observation_id":observation_id,
        "layer":layer,
        "subject":{"entity_type":subject_type,"entity_id":subject_id},
        "predicate":predicate,
        "object":{"kind":"LITERAL","value":deepcopy(value)},
        "assertion_state":"ACTIVE",
        "provenance":deepcopy(provenance),
        "verification":{"state":"AUTHORITATIVE_RECORD","basis":["pinned workspace-control Git blob"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":deepcopy(attributes),
    }


def build_workspace_semantic_manifest(snapshot_inputs: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, Any]:
    source_index=_validate_inputs(snapshot_inputs,source_manifest)
    commit=snapshot_inputs["workspace_control_commit"]
    observations=[]

    for item in snapshot_inputs["inputs"]:
        source=source_index[item["source_id"]]
        provenance=_prov(source,commit)
        kind=item["kind"]
        record=item["record"]

        if kind=="workload_policy":
            observations.append(_obs(
                observation_id="wc:authority-principle",
                layer="NORMATIVE",
                subject_type="AuthorityPolicy",
                subject_id=record["policy_id"],
                predicate="authorityPrinciple",
                value=record["authority_principle"],
                provenance=provenance,
                attributes={"policy_epoch":record["policy_epoch"],"live_lambda_mutation_authorized":record["live_lambda_mutation_authorized_by_this_contract"]},
            ))
            for workload_class,workload in sorted(record["workloads"].items()):
                observations.append(_obs(
                    observation_id=f"wc:policy:{workload_class}",
                    layer="NORMATIVE",
                    subject_type="WorkloadClass",
                    subject_id=workload_class,
                    predicate="policyDecision",
                    value=workload["policy_decision"],
                    provenance=provenance,
                    attributes={"policy_id":record["policy_id"],"policy_epoch":record["policy_epoch"],**deepcopy(workload)},
                ))
            continue

        if kind=="project_routes":
            for route_id,route in sorted(record["routes"].items()):
                observations.append(_obs(
                    observation_id=f"wc:route:{route_id}",
                    layer="EXECUTION",
                    subject_type="ProjectRoute",
                    subject_id=f'{record["project_id"]}:{route_id}',
                    predicate="bindsExecutionContract",
                    value=route["execution_contract_id"],
                    provenance=provenance,
                    attributes={"project_id":record["project_id"],"project_status":record["status"],"route_id":route_id,**deepcopy(route)},
                ))
            continue

        if kind=="execution_contracts":
            for contract_id,contract in sorted(record["contracts"].items()):
                observations.append(_obs(
                    observation_id=f"wc:execution-contract:{contract_id}",
                    layer="EXECUTION",
                    subject_type="ExecutionContract",
                    subject_id=contract_id,
                    predicate="stateEffectClass",
                    value=contract["state_effect_class"],
                    provenance=provenance,
                    attributes=deepcopy(contract),
                ))
            continue

        if kind=="pilot_contract":
            observations.append(_obs(
                observation_id="wc:pilot:status",
                layer="EXECUTION",
                subject_type="ExecutionPilot",
                subject_id="lambda-self-hosted-execution-pilot",
                predicate="hasStatus",
                value=record["status"],
                provenance=provenance,
                attributes=deepcopy(record),
            ))
            continue

        if kind=="dependency_task":
            observations.append(_obs(
                observation_id=f'wc:task:{record["task_id"]}:status',
                layer="EXECUTION",
                subject_type="Task",
                subject_id=record["task_id"],
                predicate="hasStatus",
                value=record["status"],
                provenance=provenance,
                attributes={
                    "current_stage":record["current_stage"],
                    "research_engine_checkpoint":record["research_engine_checkpoint"],
                    "accepted_output_checkpoint_present":record["accepted_output_checkpoint_present"],
                    "extract_binding_authority":record["extract_binding_authority"],
                    "source_role":source["role"],
                },
            ))
            continue

        raise ValueError(f"unsupported workspace input kind: {kind}")

    observations.sort(key=lambda x:x["observation_id"])
    body={
        "schema_version":1,
        "adapter_id":"workspace-control-snapshot-adapter-v1",
        "workspace_control_commit":commit,
        "source_manifest_id":source_manifest["manifest_id"],
        "observations":observations,
    }
    return {**body,"semantic_manifest_sha256":canonical_sha256(body)}


def query_workload_authority(manifest: dict[str, Any], workload_class: str) -> dict[str, Any]:
    policy=[o for o in manifest["observations"] if o["layer"]=="NORMATIVE" and o["subject"]=={"entity_type":"WorkloadClass","entity_id":workload_class} and o["predicate"]=="policyDecision"]
    if len(policy)!=1:
        return {"workload_class":workload_class,"decision":"ABSTAIN","reason":"missing unique authoritative policy","routes":[]}
    policy_obs=policy[0]
    routes=[
        o for o in manifest["observations"]
        if o["layer"]=="EXECUTION"
        and o["subject"]["entity_type"]=="ProjectRoute"
        and o["attributes"].get("workload_class")==workload_class
    ]
    decision=policy_obs["object"]["value"]
    if decision=="DENY":
        result="DENY"
    elif decision in {"EXISTING_P5_CAPABILITY","ALLOW"}:
        result="ALLOW"
    else:
        result="ABSTAIN"
    return {
        "workload_class":workload_class,
        "decision":result,
        "policy_decision":decision,
        "policy_observation_id":policy_obs["observation_id"],
        "routes":sorted(o["attributes"]["route_id"] for o in routes),
        "route_capability_does_not_widen_authority":True,
    }


def query_extract_dependency(manifest: dict[str, Any]) -> dict[str, Any]:
    matches=[o for o in manifest["observations"] if o["subject"]=={"entity_type":"Task","entity_id":"research-engine-local-extract-redesign-v1"} and o["predicate"]=="hasStatus"]
    if len(matches)!=1:
        raise ValueError("missing unique Local EXTRACT dependency state")
    obs=matches[0]
    return {
        "task_status":obs["object"]["value"],
        "current_stage":obs["attributes"]["current_stage"],
        "research_engine_checkpoint":obs["attributes"]["research_engine_checkpoint"],
        "accepted_output_checkpoint_present":obs["attributes"]["accepted_output_checkpoint_present"],
        "extract_binding_authority":obs["attributes"]["extract_binding_authority"],
        "source_role":obs["attributes"]["source_role"],
    }
