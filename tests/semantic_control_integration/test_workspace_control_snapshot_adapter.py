import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_integration"

def load(name):
    return json.loads((BASE/name).read_text())

def build(inputs=None):
    from prototypes.semantic_control_integration.workspace_control_adapter import build_workspace_semantic_manifest
    return build_workspace_semantic_manifest(
        inputs or load("workspace-control-snapshot-inputs.v1.json"),
        load("workspace-control-source-manifest.v1.json"),
    )

def test_workspace_snapshot_adapter_is_deterministic_under_input_reordering():
    inputs=load("workspace-control-snapshot-inputs.v1.json")
    first=build(inputs)
    changed=deepcopy(inputs)
    changed["inputs"]=list(reversed(changed["inputs"]))
    second=build(changed)
    assert first==second
    assert len(first["semantic_manifest_sha256"])==64

def test_workspace_snapshot_preserves_normative_and_execution_typing_and_provenance():
    manifest=build()
    assert {o["layer"] for o in manifest["observations"]}=={"NORMATIVE","EXECUTION"}
    source_manifest=load("workspace-control-source-manifest.v1.json")
    expected={s["source_id"]:s for s in source_manifest["sources"]}
    for obs in manifest["observations"]:
        assert obs["verification"]["state"]=="AUTHORITATIVE_RECORD"
        assert len(obs["provenance"])==1
        p=obs["provenance"][0]
        source=expected[p["source_id"]]
        assert p["git_blob_sha"]==source["git_blob_sha"]
        assert p["path"]==source["path"]

def test_gpu_small_is_allowed_but_generic_gpu_heavy_is_denied_even_with_route_capability():
    from prototypes.semantic_control_integration.workspace_control_adapter import query_workload_authority
    manifest=build()
    small=query_workload_authority(manifest,"gpu-small")
    heavy=query_workload_authority(manifest,"gpu-heavy")
    assert small["decision"]=="ALLOW"
    assert small["policy_decision"]=="EXISTING_P5_CAPABILITY"
    assert small["routes"]==["gpu-small"]
    assert heavy["decision"]=="DENY"
    assert heavy["policy_decision"]=="DENY"
    assert heavy["routes"]==["model-eval-local-extract-wave7a3"]
    assert heavy["route_capability_does_not_widen_authority"] is True

def test_extract_task_is_dependency_state_only_and_cannot_bind_integration():
    from prototypes.semantic_control_integration.workspace_control_adapter import query_extract_dependency
    dep=query_extract_dependency(build())
    assert dep["task_status"]=="CHECKPOINTED"
    assert dep["current_stage"]=="family_c_iterative_residual_design"
    assert dep["accepted_output_checkpoint_present"] is False
    assert dep["extract_binding_authority"] is False
    assert dep["source_role"]=="DEPENDENCY_STATE_ONLY"

def test_workspace_source_blob_drift_fails_closed():
    inputs=load("workspace-control-snapshot-inputs.v1.json")
    changed=deepcopy(inputs)
    changed["inputs"][0]["git_blob_sha"]="0"*40
    with pytest.raises(ValueError,match="blob drift"):
        build(changed)

def test_unknown_source_fails_closed():
    inputs=load("workspace-control-snapshot-inputs.v1.json")
    changed=deepcopy(inputs)
    changed["inputs"][0]["source_id"]="wc:not-real"
    with pytest.raises(ValueError,match="unknown workspace source"):
        build(changed)

def test_adapter_source_has_no_workspace_mutation_or_external_runtime_paths():
    lowered=(ROOT/"prototypes"/"semantic_control_integration"/"workspace_control_adapter.py").read_text().lower()
    for token in [
        "requests","subprocess","openai","github","update_file","create_file",
        "heartbeat","write_text","unlink(","remove(","rename(","boto3"
    ]:
        assert token not in lowered
