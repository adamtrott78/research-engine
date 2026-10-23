from __future__ import annotations

import hashlib
import json
from typing import Any


SEMANTIC_MANIFEST_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for prototype identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_index(fixture_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = fixture_manifest.get("sources", [])
    index = {source["source_id"]: source for source in sources}
    if len(index) != len(sources):
        raise ValueError("duplicate source_id in fixture manifest")
    return index


def _exact_source_ref(source_id: str, source_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    try:
        source = source_index[source_id]
    except KeyError as exc:
        raise ValueError(f"unknown source_id: {source_id}") from exc
    return {
        "source_id": source["source_id"],
        "repository": source["repository"],
        "commit_sha": source["commit_sha"],
        "path": source["path"],
        "git_blob_sha": source["git_blob_sha"],
        "role": source["role"],
    }


def _observation(
    *,
    observation_id: str,
    layer: str,
    subject_type: str,
    subject_id: str,
    predicate: str,
    value: Any,
    source_ref: dict[str, Any],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_id": observation_id,
        "layer": layer,
        "subject": {"entity_type": subject_type, "entity_id": subject_id},
        "predicate": predicate,
        "object": {"kind": "LITERAL", "value": value},
        "assertion_state": "ACTIVE",
        "provenance": [source_ref],
        "verification": {
            "state": "AUTHORITATIVE_RECORD",
            "basis": ["frozen provenance-pinned fixture"],
        },
        "temporal": {
            "observed_at": None,
            "valid_from": None,
            "valid_to": None,
        },
        "supersedes": [],
        "contradicts": [],
        "attributes": attributes,
    }


def adapt_research_engine_wave7a3_result(
    item: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    record = item["record"]
    return [
        _observation(
            observation_id="obs:wave7a3-verdict",
            layer="EPISTEMIC",
            subject_type="ScientificExperiment",
            subject_id="wave7a3-author-anchored",
            predicate="hasScientificDisposition",
            value=record["decision"],
            source_ref=_exact_source_ref(item["source_id"], source_index),
            attributes={
                "implementation_validity": record["implementation_validity"],
                "scientific_gate_consumed": record["scientific_gate_consumed"],
                "author_salience_hypothesis": record["author_salience_hypothesis"],
                "wave7b_authorized": record["wave7b_authorized"],
                "production_extract_changed": record["production_extract_changed"],
                "qwen_verify_changed": record["qwen_verify_changed"],
            },
        )
    ]


def adapt_workspace_task_capsule(
    item: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    record = item["record"]
    return [
        _observation(
            observation_id="obs:extract-task-checkpointed",
            layer="EXECUTION",
            subject_type="Task",
            subject_id=record["task_id"],
            predicate="hasStatus",
            value=record["status"],
            source_ref=_exact_source_ref(item["source_id"], source_index),
            attributes={
                "current_stage": record["current_stage"],
                "next_action_class": record["next_action_class"],
            },
        )
    ]


def adapt_workspace_workload_policy(
    item: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    record = item["record"]
    return [
        _observation(
            observation_id="obs:gpu-heavy-denied",
            layer="NORMATIVE",
            subject_type="WorkloadClass",
            subject_id=record["workload_class"],
            predicate="policyDecision",
            value=record["policy_decision"],
            source_ref=_exact_source_ref(item["source_id"], source_index),
            attributes={
                "policy_id": record["policy_id"],
                "policy_epoch": record["policy_epoch"],
                "denial_reason": record["denial_reason"],
            },
        )
    ]


def adapt_workspace_project_route(
    item: dict[str, Any], source_index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    record = item["record"]
    return [
        _observation(
            observation_id="obs:wave7a3-route",
            layer="EXECUTION",
            subject_type="ProjectRoute",
            subject_id=f'{record["project_id"]}:{record["route_id"]}',
            predicate="bindsExecutionContract",
            value=record["execution_contract_id"],
            source_ref=_exact_source_ref(item["source_id"], source_index),
            attributes={
                "lane": record["lane"],
                "activation_state": record["activation_state"],
                "workflow_path": record["workflow_path"],
            },
        )
    ]


ADAPTERS = {
    "research_engine_wave7a3_result": adapt_research_engine_wave7a3_result,
    "workspace_task_capsule": adapt_workspace_task_capsule,
    "workspace_workload_policy": adapt_workspace_workload_policy,
    "workspace_project_route": adapt_workspace_project_route,
}


def build_semantic_manifest(
    adapter_inputs: dict[str, Any],
    fixture_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic semantic projection from frozen fixture inputs only."""
    source_index = _source_index(fixture_manifest)
    observations: list[dict[str, Any]] = []
    for item in adapter_inputs.get("inputs", []):
        adapter_name = item["adapter"]
        try:
            adapter = ADAPTERS[adapter_name]
        except KeyError as exc:
            raise ValueError(f"unknown adapter: {adapter_name}") from exc
        observations.extend(adapter(item, source_index))

    observations.sort(key=lambda observation: observation["observation_id"])
    body = {
        "schema_version": SEMANTIC_MANIFEST_SCHEMA_VERSION,
        "source_manifest_id": fixture_manifest["manifest_id"],
        "fixture_set_id": adapter_inputs["fixture_set_id"],
        "observations": observations,
    }
    return {
        **body,
        "semantic_manifest_sha256": sha256_hex(body),
    }


def validate_manifest_provenance(
    semantic_manifest: dict[str, Any],
    fixture_manifest: dict[str, Any],
) -> None:
    """Fail if a projected observation references anything outside the frozen manifest."""
    sources = _source_index(fixture_manifest)
    for observation in semantic_manifest["observations"]:
        provenance = observation.get("provenance", [])
        if not provenance:
            raise ValueError(f'{observation["observation_id"]} has no provenance')
        for source_ref in provenance:
            source_id = source_ref["source_id"]
            expected = _exact_source_ref(source_id, sources)
            if source_ref != expected:
                raise ValueError(
                    f'{observation["observation_id"]} provenance drift for {source_id}'
                )
