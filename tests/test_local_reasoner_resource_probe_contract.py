import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/"benchmarks"/"semantic_control"/"local-reasoner-resource-probe.v1.json"
W=ROOT/".github"/"workflows"/"lambda-ci-local-reasoner-resource-probe.yml"
H=ROOT/"benchmarks"/"semantic_control"/"run_local_reasoner_resource_probe.py"

def load(): return json.loads(C.read_text())

def test_probe_is_resource_only_and_frozen():
    c=load()
    assert c["status"]=="FROZEN_RESOURCE_FEASIBILITY_ONLY"
    assert c["scientific_scope"]["quality_winner_may_be_selected"] is False
    assert c["scientific_scope"]["scientific_state_materialized"] is False
    assert c["scientific_scope"]["production_routing_changed"] is False
    assert c["scientific_scope"]["external_llm_calls"]==0

def test_exact_candidate_and_runtime_identities():
    c=load(); a={x["id"]:x for x in c["candidates"]}
    assert c["runtime"]["commit"]=="620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert a["qwen35-9b-q6-resource"]["upstream"]["revision"]=="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert a["qwen35-9b-q6-resource"]["probe_artifact"]["sha256"]=="f68abc790e952b1e1a9d636d4696208732ae540871cb8cb54c64e4b5e232a5d2"
    assert a["qwythos-9b-v2-q6-resource"]["upstream"]["revision"]=="2178f73a9b5ea28ccd8f6096ef6bac5cd59c9d8b"
    assert a["qwythos-9b-v2-q6-resource"]["probe_artifact"]["sha256"]=="dd39e148823f0bab858e946d603e4bf424240aae34435580be820f09c75ca379"

def test_probe_preserves_quality_boundary():
    c=load()
    assert "not evidence" in c["interpretation"]["quality_boundary"]
    assert "equalized quantization" in c["interpretation"]["quantization_confound"]

def test_workflow_is_exact_head_gpu_only_and_nonpersistent():
    w=W.read_text()
    for s in ["lambda-research-engine-gpu","persist-credentials: false","620d2225cc436ab0b001e3200e590e3b286ef9ab","/lambda-reasoner-probe "]: assert s in w
    for bad in ["systemctl","trusted_cache.py promote","docker run"]: assert bad not in w
    c=load()
    assert c["execution_boundary"]["persistent_cache_write"] is False
    assert c["execution_boundary"]["trusted_cache_promotion"] is False
    assert c["execution_boundary"]["workload_class"] == "gpu-experiment-public"
    assert c["execution_boundary"]["runner_runtime_ceiling_seconds"] == 1200

def test_helper_deletes_download_and_has_no_cloud_model_path():
    h=H.read_text()
    assert "model.unlink()" in h
    assert "openai" not in h.lower()
    assert "anthropic" not in h.lower()
