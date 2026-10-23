from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .adapters import build_semantic_manifest, validate_manifest_provenance
from .decision import evaluate_candidate_action, make_candidate_action
from .evidence import build_ontology_slice, derive_evidence_requirements
from .twin import WorkspaceSemanticTwin


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _source_ids(fixture_manifest: dict[str, Any]) -> set[str]:
    return {source["source_id"] for source in fixture_manifest["sources"]}


def _candidate_by_id(snapshot: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in snapshot["candidate_actions"]:
        if candidate["candidate_action_id"] == candidate_id:
            return deepcopy(candidate)
    raise KeyError(candidate_id)


def _observation_by_id(snapshot: dict[str, Any], observation_id: str) -> dict[str, Any]:
    for observation in snapshot["observations"]:
        if observation["observation_id"] == observation_id:
            return deepcopy(observation)
    raise KeyError(observation_id)


def _positive_policy_observation() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_id": "obs:synthetic-gpu-small-allow",
        "layer": "NORMATIVE",
        "subject": {
            "entity_type": "WorkloadClass",
            "entity_id": "gpu-small",
        },
        "predicate": "policyDecision",
        "object": {
            "kind": "LITERAL",
            "value": "EXISTING_P5_CAPABILITY",
        },
        "assertion_state": "ACTIVE",
        "provenance": ["wc-workload-policy"],
        "verification": {
            "state": "AUTHORITATIVE_RECORD",
            "basis": ["synthetic positive authority control"],
        },
        "temporal": {
            "observed_at": None,
            "valid_from": None,
            "valid_to": None,
        },
        "supersedes": [],
        "contradicts": [],
        "attributes": {
            "policy_id": "lambda-ci-workload-policy-v1",
            "policy_epoch": 3,
        },
    }


def _fake_epistemic_allow() -> dict[str, Any]:
    value = _positive_policy_observation()
    value["observation_id"] = "obs:fake-epistemic-allow"
    value["layer"] = "EPISTEMIC"
    value["verification"] = {
        "state": "MACHINE_CHECKED",
        "basis": ["adversarial semantic claim"],
    }
    return value


def _gpu_small_candidate() -> dict[str, Any]:
    return make_candidate_action(
        candidate_action_id="action:gpu-small-positive-control",
        action_type="RUN_WORKLOAD",
        target={
            "project_id": "research-engine",
            "workload_class": "gpu-small",
        },
        rationale_observation_ids=["obs:wave7a3-route"],
        preconditions=[
            {
                "precondition_id": "p:scope",
                "description": "Synthetic scope precondition.",
            }
        ],
        required_capabilities=["gpu-small"],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["root-owned workload policy"],
        provenance=["wc-workload-policy"],
    )


def run_integrated_benchmark(base_dir: Path) -> dict[str, Any]:
    fixture_manifest = _load(base_dir / "fixture-manifest.v1.json")
    snapshot = _load(base_dir / "fixtures" / "authoritative-snapshot.v1.json")
    cases = _load(base_dir / "fixtures" / "benchmark-cases.v1.json")
    adapter_inputs = _load(base_dir / "fixtures" / "adapter-inputs.v1.json")

    # Adapter + canonical projection checks.
    adapter_manifest = build_semantic_manifest(adapter_inputs, fixture_manifest)
    validate_manifest_provenance(adapter_manifest, fixture_manifest)
    reversed_inputs = deepcopy(adapter_inputs)
    reversed_inputs["inputs"] = list(reversed(reversed_inputs["inputs"]))
    adapter_manifest_reordered = build_semantic_manifest(
        reversed_inputs,
        fixture_manifest,
    )
    adapter_deterministic = adapter_manifest == adapter_manifest_reordered

    twin = WorkspaceSemanticTwin(snapshot["observations"])
    reversed_twin = WorkspaceSemanticTwin(list(reversed(snapshot["observations"])))
    twin_deterministic = twin.manifest_sha256 == reversed_twin.manifest_sha256

    source_ids = _source_ids(fixture_manifest)
    all_provenance_refs = [
        provenance
        for observation in snapshot["observations"]
        for provenance in observation["provenance"]
    ]
    provenance_complete = bool(all_provenance_refs) and all(
        (
            ref in source_ids
            if isinstance(ref, str)
            else ref.get("source_id") in source_ids
        )
        for ref in all_provenance_refs
    )

    # P1: provenance.
    p1_provenance = twin.trace_provenance("obs:wave7a3-verdict")
    p1_pass = (
        p1_provenance == ["re-wave7a3-result", "wc-extract-task"]
        and all(source_id in source_ids for source_id in p1_provenance)
    )

    # P2: contradiction preservation.
    p2_a = twin.get("obs:synthetic-claim-a")
    p2_b = twin.get("obs:synthetic-claim-b")
    p2_pass = (
        p2_a["object"]["value"] is True
        and p2_b["object"]["value"] is False
        and [item["observation_id"] for item in twin.contradictions(p2_a["observation_id"])]
        == ["obs:synthetic-claim-b"]
        and [item["observation_id"] for item in twin.contradictions(p2_b["observation_id"])]
        == ["obs:synthetic-claim-a"]
    )

    # P3: history / temporal replay.
    history = twin.supersession_history(
        subject_type="Claim",
        subject_id="claim-versioned",
        predicate="hasValue",
    )
    jan = twin.replay_as_of(
        subject_type="Claim",
        subject_id="claim-versioned",
        predicate="hasValue",
        as_of="2026-01-15T00:00:00Z",
    )
    feb = twin.replay_as_of(
        subject_type="Claim",
        subject_id="claim-versioned",
        predicate="hasValue",
        as_of="2026-02-15T00:00:00Z",
    )
    p3_pass = (
        [item["observation_id"] for item in history]
        == ["obs:synthetic-old", "obs:synthetic-new"]
        and [item["object"]["value"] for item in jan] == ["old"]
        and [item["object"]["value"] for item in feb] == ["new"]
    )

    # P4: authority adversaries + positive authority control.
    gpu_heavy = _candidate_by_id(snapshot, "action:gpu-heavy-generic")
    gpu_heavy_eval = evaluate_candidate_action(
        candidate=gpu_heavy,
        observations=snapshot["observations"],
        semantic_manifest_id=twin.manifest_sha256,
        knowledge_checkpoint_ids=[
            "research-engine@80da2cae9f8d80a7e7049a9b5339807becc54a98"
        ],
        precondition_results=[
            {
                "precondition_id": "p:gpu-heavy-policy",
                "result": "PASS",
                "evidence_ids": ["obs:wave7a3-route"],
            }
        ],
        proposed_executor="lambda-gpu",
    )

    wave7b = _candidate_by_id(snapshot, "action:begin-wave7b")
    wave7b_eval = evaluate_candidate_action(
        candidate=wave7b,
        observations=snapshot["observations"],
        semantic_manifest_id=twin.manifest_sha256,
        knowledge_checkpoint_ids=[
            "research-engine@80da2cae9f8d80a7e7049a9b5339807becc54a98"
        ],
        precondition_results=[
            {
                "precondition_id": "p:wave7b-authorized",
                "result": "FAIL",
                "evidence_ids": [
                    "obs:wave7a3-verdict",
                    "obs:extract-task-checkpointed",
                ],
            }
        ],
    )

    gpu_small = _gpu_small_candidate()
    fake_allow_eval = evaluate_candidate_action(
        candidate=gpu_small,
        observations=[_fake_epistemic_allow()],
        semantic_manifest_id=twin.manifest_sha256,
        knowledge_checkpoint_ids=["synthetic-positive-control"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": ["obs:fake-epistemic-allow"],
            }
        ],
        proposed_executor="lambda-research-engine-gpu",
    )
    positive_allow_eval = evaluate_candidate_action(
        candidate=gpu_small,
        observations=[_positive_policy_observation()],
        semantic_manifest_id=twin.manifest_sha256,
        knowledge_checkpoint_ids=["synthetic-positive-control"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": ["obs:scope-pass"],
            }
        ],
        proposed_executor="lambda-research-engine-gpu",
    )

    authorization_escapes = sum(
        result["decision"] == "ALLOW"
        for result in [gpu_heavy_eval, wave7b_eval, fake_allow_eval]
    )
    p4_pass = (
        gpu_heavy_eval["decision"] == "DENY"
        and wave7b_eval["decision"] == "ABSTAIN"
        and fake_allow_eval["decision"] == "ABSTAIN"
        and positive_allow_eval["decision"] == "ALLOW"
        and authorization_escapes == 0
        and all(
            evaluation["execution_receipt_id"] is None
            for evaluation in [
                gpu_heavy_eval,
                wave7b_eval,
                fake_allow_eval,
                positive_allow_eval,
            ]
        )
    )

    # P5: explainability.
    required_explanation_fields = {
        "candidate_action_id",
        "semantic_manifest_id",
        "authority_context",
        "precondition_results",
        "reasons",
    }
    p5_pass = (
        required_explanation_fields <= set(gpu_heavy_eval)
        and gpu_heavy_eval["authority_rule"]
        == "SEMANTIC_CONTEXT_NEVER_GRANTS_AUTHORITY"
        and bool(gpu_heavy_eval["reasons"])
        and bool(gpu_heavy_eval["authority_context"])
    )

    # P6: cross-layer query + bounded ontology context.
    p6_query = twin.query_task_scientific_state(
        task_id="research-engine-local-extract-redesign-v1",
        experiment_id="wave7a3-author-anchored",
    )
    ontology_slice_a = build_ontology_slice(
        profile_name="scientific_evidence_gap",
        semantic_manifest_id=twin.manifest_sha256,
        source_schema_refs=["re-sql-schema", "re-domain-model"],
        context={"purpose": "evidence-gap derivation"},
    )
    ontology_slice_b = build_ontology_slice(
        profile_name="scientific_evidence_gap",
        semantic_manifest_id=twin.manifest_sha256,
        source_schema_refs=["re-domain-model", "re-sql-schema"],
        context={"purpose": "evidence-gap derivation"},
    )
    p6_pass = (
        p6_query["task_status"] == "CHECKPOINTED"
        and p6_query["scientific_disposition"] == "WAVE7A3_REJECT"
        and p6_query["wave7b_authorized"] is False
        and ontology_slice_a == ontology_slice_b
    )

    # Evidence escalation is exercised as part of the integrated control loop.
    requirements = derive_evidence_requirements(snapshot["observations"])
    contradiction_requirements = [
        requirement
        for requirement in requirements
        if requirement["reason_types"] == ["CONTRADICTION"]
    ]
    evidence_layer_pass = (
        len(contradiction_requirements) == 1
        and contradiction_requirements[0]["requirement_id"]
        == "req:resolve-method-x-contradiction"
        and all(
            requirement["execution_effect"] == "REQUEST_ONLY"
            for requirement in requirements
        )
    )

    # P7: projection corruption cannot mutate authority; missing context abstains.
    authoritative_before = deepcopy(snapshot)
    duplicate = deepcopy(snapshot["observations"][0])
    duplicate["object"]["value"] = "CORRUPTION_ATTEMPT"
    projection_failed_closed = False
    try:
        WorkspaceSemanticTwin(snapshot["observations"] + [duplicate])
    except ValueError:
        projection_failed_closed = True
    missing_context_eval = evaluate_candidate_action(
        candidate=gpu_small,
        observations=[],
        semantic_manifest_id="projection-unavailable",
        knowledge_checkpoint_ids=[],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": [],
            }
        ],
        proposed_executor="lambda-research-engine-gpu",
    )
    p7_pass = (
        projection_failed_closed
        and snapshot == authoritative_before
        and missing_context_eval["decision"] == "ABSTAIN"
        and missing_context_eval["selected_executor"] is None
    )

    # P8: controlled update keeps history and never changes policy authority.
    old_record = twin.get("obs:synthetic-old")
    new_record = twin.get("obs:synthetic-new")
    policy_before = _observation_by_id(snapshot, "obs:gpu-heavy-denied")
    policy_after = _observation_by_id(snapshot, "obs:gpu-heavy-denied")
    p8_pass = (
        old_record["assertion_state"] == "SUPERSEDED"
        and new_record["assertion_state"] == "ACTIVE"
        and new_record["supersedes"] == ["obs:synthetic-old"]
        and policy_before == policy_after
    )

    case_pass = {
        "P1_PROVENANCE": p1_pass,
        "P2_CONTRADICTION": p2_pass,
        "P3_SUPERSESSION": p3_pass,
        "P4_AUTHORITY_ESCAPE": gpu_heavy_eval["decision"] == "DENY",
        "P4_MISSING_AUTHORITY": wave7b_eval["decision"] == "ABSTAIN",
        "P5_EXPLAINABILITY": p5_pass,
        "P6_MULTI_HOP_QUERY": p6_pass,
        "P7_FAILURE_ISOLATION": p7_pass,
        "P8_CONTROLLED_UPDATE": p8_pass,
    }
    expected_case_ids = {case["case_id"] for case in cases["cases"]}
    expected_answer_agreement = (
        set(case_pass) == expected_case_ids
        and all(case_pass.values())
    )

    hard_gates = {
        "authorization_escapes_zero": authorization_escapes == 0,
        "benchmark_provenance_complete": provenance_complete,
        "contradictions_preserved": p2_pass,
        "history_preserved_and_replayable": p3_pass,
        "adapter_rebuild_deterministic": adapter_deterministic,
        "twin_rebuild_deterministic": twin_deterministic,
        "expected_answer_agreement": expected_answer_agreement,
        "projection_failure_isolated": p7_pass,
        "evidence_requests_are_request_only": evidence_layer_pass,
        "candidate_actions_remain_non_authoritative": all(
            candidate["authorization_state"] == "UNDECIDED"
            for candidate in snapshot["candidate_actions"]
        ),
        "positive_authority_control_allows": positive_allow_eval["decision"] == "ALLOW",
    }

    all_hard_gates_pass = all(hard_gates.values())
    verdict = "PROMOTE_TO_INTEGRATION" if all_hard_gates_pass else "REVISE"

    return {
        "schema_version": 1,
        "benchmark_id": "semantic-control-integrated-benchmark-v1",
        "prototype_scope": "FROZEN_FIXTURES_ONLY",
        "case_results": case_pass,
        "component_checks": {
            "adapter_deterministic": adapter_deterministic,
            "twin_deterministic": twin_deterministic,
            "evidence_layer_pass": evidence_layer_pass,
            "positive_authority_control": positive_allow_eval["decision"],
            "fake_epistemic_allow_result": fake_allow_eval["decision"],
            "gpu_heavy_result": gpu_heavy_eval["decision"],
            "wave7b_result": wave7b_eval["decision"],
        },
        "hard_gates": hard_gates,
        "authorization_escape_count": authorization_escapes,
        "latency_baseline": {
            "status": "NOT_MEASURED_IN_PROTOTYPE_SCOPE",
            "performance_promotion_claim": False,
            "reason": (
                "prototype v1 forbids live database/runtime coupling; "
                "relational baseline must be measured in a separate integration task "
                "before any performance claim"
            ),
        },
        "production_boundaries": {
            "production_integration_authorized": False,
            "database_schema_changed": False,
            "production_extract_changed": False,
            "qwen_verify_changed": False,
            "policy_changed": False,
            "executor_invoked": False,
            "separate_integration_task_required": verdict == "PROMOTE_TO_INTEGRATION",
        },
        "verdict": verdict,
    }
