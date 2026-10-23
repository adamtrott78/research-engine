from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _nearest_rank(samples: list[int], p: float) -> int:
    ordered=sorted(samples)
    return ordered[max(1,math.ceil(p*len(ordered)))-1]


def _setup_store(repo_root: Path):
    dsn=os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        raise RuntimeError("RESEARCH_ENGINE_DATABASE_URL is required")
    from research_engine.corpus_loader import PostgresCorpusLoader
    from research_engine.postgres import PostgresStore
    store=PostgresStore(dsn)
    store.initialize(repo_root/"sql"/"schema.sql")
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                TRUNCATE knowledge_checkpoints, verification_events,
                         evidence_relations, evidence_objects, source_locators,
                         extraction_runs, acquisition_attempts, discovery_runs,
                         external_identifiers, source_artifacts, work_versions, works
                CASCADE
            """)
        conn.commit()
    fixture=_load(repo_root/"tests"/"data"/"rf_osr_five_work.json")
    PostgresCorpusLoader(store).load(fixture)
    return store


def _candidate(action_id: str, workload_class: str, rationale: str) -> dict[str, Any]:
    from prototypes.semantic_control.decision import make_candidate_action
    return make_candidate_action(
        candidate_action_id=action_id,
        action_type="RUN_WORKLOAD",
        target={"project_id":"research-engine","workload_class":workload_class},
        rationale_observation_ids=[rationale],
        preconditions=[{"precondition_id":"p:scope","description":"Frozen integration scope precondition."}],
        required_capabilities=[workload_class],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["root-owned workload policy"],
        provenance=["wc-workload-policy"],
    )


def run_pre_extract_integration(repo_root: Path) -> dict[str, Any]:
    base=repo_root/"benchmarks"/"semantic_control_integration"
    baseline_contract=_load(base/"relational-baseline-contract.v1.json")
    baseline_result=_load(base/"relational-baseline-result.v1.json")
    adapter_overhead=_load(base/"research-engine-adapter-overhead-result.v1.json")
    wc_inputs=_load(base/"workspace-control-snapshot-inputs.v1.json")
    wc_sources=_load(base/"workspace-control-source-manifest.v1.json")
    extract_gate=_load(base/"extract-dependency-gate-current.v1.json")

    from prototypes.semantic_control_integration.research_engine_adapter import ReadOnlyResearchEngineAdapter
    from prototypes.semantic_control_integration.workspace_control_adapter import build_workspace_semantic_manifest,query_extract_dependency,query_workload_authority
    from prototypes.semantic_control.decision import evaluate_candidate_action

    store=_setup_store(repo_root)
    research_adapter=ReadOnlyResearchEngineAdapter(
        store=store,
        baseline_contract=baseline_contract,
        baseline_result=baseline_result,
        runtime_repository_commit="8bf738f2c6722cda60f98190b58892e8e7979918",
        baseline_contract_blob_sha="1c97ab527db5d375ff7bb068719330a4acc0458c",
    )
    research_manifest=research_adapter.build_manifest()
    research_manifest_second=research_adapter.build_manifest()

    workspace_manifest=build_workspace_semantic_manifest(wc_inputs,wc_sources)
    reversed_inputs=deepcopy(wc_inputs)
    reversed_inputs["inputs"]=list(reversed(reversed_inputs["inputs"]))
    workspace_manifest_second=build_workspace_semantic_manifest(reversed_inputs,wc_sources)

    expected_hashes={x["query_id"]:x["canonical_result_sha256"] for x in baseline_result["query_results"]}
    actual_hashes={x["query_id"]:x["relational_result_sha256"] for x in research_manifest["query_records"]}
    reconstructed_hashes={x["query_id"]:x["semantic_reconstruction_sha256"] for x in research_manifest["query_records"]}
    relational_semantic_equivalence=(actual_hashes==expected_hashes==reconstructed_hashes)

    small=query_workload_authority(workspace_manifest,"gpu-small")
    heavy=query_workload_authority(workspace_manifest,"gpu-heavy")
    precondition=[{"precondition_id":"p:scope","result":"PASS","evidence_ids":["wc:authority-principle"]}]
    small_eval=evaluate_candidate_action(
        candidate=_candidate("action:integration-gpu-small","gpu-small","wc:route:gpu-small"),
        observations=workspace_manifest["observations"],
        semantic_manifest_id=workspace_manifest["semantic_manifest_sha256"],
        knowledge_checkpoint_ids=["workspace-control@91459f5f6afa0d71394d63155c2fe8dfdc65a4b7"],
        precondition_results=precondition,
        proposed_executor="lambda-research-engine-gpu",
    )
    heavy_eval=evaluate_candidate_action(
        candidate=_candidate("action:integration-gpu-heavy","gpu-heavy","wc:route:model-eval-local-extract-wave7a3"),
        observations=workspace_manifest["observations"],
        semantic_manifest_id=workspace_manifest["semantic_manifest_sha256"],
        knowledge_checkpoint_ids=["workspace-control@91459f5f6afa0d71394d63155c2fe8dfdc65a4b7"],
        precondition_results=precondition,
        proposed_executor="lambda-research-engine-gpu",
    )

    fake=deepcopy(next(o for o in workspace_manifest["observations"] if o["observation_id"]=="wc:policy:gpu-heavy"))
    fake["observation_id"]="wc:fake-epistemic-gpu-heavy-allow"
    fake["layer"]="EPISTEMIC"
    fake["object"]["value"]="ALLOW"
    fake_eval=evaluate_candidate_action(
        candidate=_candidate("action:fake-gpu-heavy","gpu-heavy","wc:route:model-eval-local-extract-wave7a3"),
        observations=[fake],
        semantic_manifest_id="fake-only",
        knowledge_checkpoint_ids=[],
        precondition_results=precondition,
        proposed_executor="lambda-research-engine-gpu",
    )
    missing_eval=evaluate_candidate_action(
        candidate=_candidate("action:missing-authority","gpu-small","wc:route:gpu-small"),
        observations=[],
        semantic_manifest_id="missing-authority",
        knowledge_checkpoint_ids=[],
        precondition_results=precondition,
        proposed_executor="lambda-research-engine-gpu",
    )
    authorization_escapes=sum(x["decision"]=="ALLOW" for x in [heavy_eval,fake_eval,missing_eval])

    wc_drift_failed=False
    drift=deepcopy(wc_inputs)
    drift["inputs"][0]["git_blob_sha"]="0"*40
    try:
        build_workspace_semantic_manifest(drift,wc_sources)
    except ValueError:
        wc_drift_failed=True

    re_drift_failed=False
    bad_baseline=deepcopy(baseline_result)
    bad_baseline["query_results"][0]["canonical_result_sha256"]="0"*64
    bad_adapter=ReadOnlyResearchEngineAdapter(
        store=store,
        baseline_contract=baseline_contract,
        baseline_result=bad_baseline,
        runtime_repository_commit="8bf738f2c6722cda60f98190b58892e8e7979918",
        baseline_contract_blob_sha="1c97ab527db5d375ff7bb068719330a4acc0458c",
    )
    try:
        bad_adapter.read_and_project_query(baseline_contract["queries"][0])
    except ValueError:
        re_drift_failed=True

    for _ in range(20):
        build_workspace_semantic_manifest(wc_inputs,wc_sources)
    wc_samples=[]
    for _ in range(1000):
        started=time.perf_counter_ns()
        build_workspace_semantic_manifest(wc_inputs,wc_sources)
        wc_samples.append(time.perf_counter_ns()-started)

    baseline_by={x["query_id"]:x for x in baseline_result["query_results"]}
    overhead_by={x["query_id"]:x for x in adapter_overhead["query_results"]}
    latency_comparison=[]
    for query_id in sorted(baseline_by):
        relational=baseline_by[query_id]["latency_ns"]["p50"]
        projection=overhead_by[query_id]["latency_ns"]["p50"]
        latency_comparison.append({
            "query_id":query_id,
            "relational_p50_ns":relational,
            "semantic_projection_p50_ns":projection,
            "projection_over_relational_percent":projection/relational*100,
            "estimated_read_plus_projection_p50_ns":relational+projection,
        })

    extract_dependency=query_extract_dependency(workspace_manifest)
    extract_blocked=(
        extract_gate["gate_status"]=="BLOCKED_EVIDENCE"
        and extract_gate["accepted_output_checkpoint_present"] is False
        and extract_gate["scientific_result"]["semantic_generalization_execution_authorized"] is False
        and extract_gate["scientific_result"]["production_promotion_authorized"] is False
    )

    gates={
        "relational_semantic_equivalence":relational_semantic_equivalence,
        "research_manifest_deterministic":research_manifest==research_manifest_second,
        "workspace_manifest_deterministic":workspace_manifest==workspace_manifest_second,
        "gpu_small_authority_allow":small["decision"]=="ALLOW" and small_eval["decision"]=="ALLOW",
        "generic_gpu_heavy_authority_deny":heavy["decision"]=="DENY" and heavy_eval["decision"]=="DENY",
        "fake_epistemic_authority_abstains":fake_eval["decision"]=="ABSTAIN",
        "missing_authority_abstains":missing_eval["decision"]=="ABSTAIN",
        "authorization_escapes_zero":authorization_escapes==0,
        "workspace_blob_drift_fails_closed":wc_drift_failed,
        "research_answer_drift_fails_closed":re_drift_failed,
        "extract_binding_blocked_evidence":extract_blocked,
        "pinned_extract_dependency_non_authoritative":extract_dependency["extract_binding_authority"] is False,
    }
    available_pass=all(gates.values())

    wc_sorted=sorted(wc_samples)
    return {
        "schema_version":1,
        "benchmark_id":"semantic-control-pre-extract-integration-v1",
        "status":"PRE_EXTRACT_INTEGRATION_PASS" if available_pass else "REVISE",
        "extract_binding":"BLOCKED_EVIDENCE" if extract_blocked else "UNEXPECTEDLY_RESOLVED",
        "final_production_verdict_issued":False,
        "production_promotion_authorized":False,
        "authorization_escape_count":authorization_escapes,
        "gates":gates,
        "research_semantic_manifest_sha256":research_manifest["semantic_manifest_sha256"],
        "workspace_semantic_manifest_sha256":workspace_manifest["semantic_manifest_sha256"],
        "authority_results":{
            "gpu_small_query":small,
            "gpu_heavy_query":heavy,
            "gpu_small_decision":small_eval["decision"],
            "gpu_heavy_decision":heavy_eval["decision"],
            "fake_epistemic_decision":fake_eval["decision"],
            "missing_authority_decision":missing_eval["decision"],
        },
        "latency_comparison":latency_comparison,
        "workspace_projection_overhead":{
            "warmup_repetitions":20,
            "measured_repetitions":1000,
            "latency_ns":{
                "min":min(wc_samples),
                "mean":int(round(statistics.fmean(wc_samples))),
                "p50":wc_sorted[499],
                "p95":wc_sorted[949],
                "max":max(wc_samples),
            },
        },
        "extract_dependency_gate":extract_gate,
        "performance_promotion_claim":False,
        "latency_threshold_applied":False,
        "production_database_access":False,
        "production_mutation":False,
    }
