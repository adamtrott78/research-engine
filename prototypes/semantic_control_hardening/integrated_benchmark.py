from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from prototypes.semantic_control.decision import (
    evaluate_candidate_action,
    make_candidate_action,
)
from prototypes.semantic_control_hardening.audit_chain import (
    build_decision_audit_chain,
    verify_decision_audit_chain,
)
from prototypes.semantic_control_hardening.opaque_extract_socket import (
    materialize_synthetic_descriptor,
    register_future_extract_descriptor,
)
from prototypes.semantic_control_hardening.replay_temporal import (
    contradiction_view,
    replay_historical_cases,
    replay_temporal_case,
    unknown_semantics,
)
from prototypes.semantic_control_hardening.slice_evidence import (
    derive_frozen_evidence_case,
    select_bounded_ontology_slice,
)
from prototypes.semantic_control_hardening.version_skew import (
    evaluate_frozen_adversary,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load(base: Path, name: str) -> Any:
    return json.loads((base/name).read_text())


def _observation(
    *,
    observation_id: str,
    layer: str,
    subject_type: str,
    subject_id: str,
    predicate: str,
    object_type: str | None = None,
    object_id: str | None = None,
    literal: Any = None,
) -> dict[str, Any]:
    object_value=(
        {"kind":"ENTITY","entity_type":object_type,"entity_id":object_id}
        if object_type is not None
        else {"kind":"LITERAL","value":literal}
    )
    return {
        "schema_version":1,
        "observation_id":observation_id,
        "layer":layer,
        "subject":{"entity_type":subject_type,"entity_id":subject_id},
        "predicate":predicate,
        "object":object_value,
        "assertion_state":"ACTIVE",
        "provenance":[{
            "source_kind":"SYNTHETIC_HARDENING_INTEGRATION",
            "source_id":"hardening-final-fixture",
            "synthetic":True,
        }],
        "verification":{"state":"SYNTHETIC_TEST_FIXTURE","basis":["integrated hardening benchmark"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":{"synthetic":True},
    }


def _slice_observations() -> list[dict[str, Any]]:
    return [
        _observation(
            observation_id="slice:authority-principle",
            layer="NORMATIVE",
            subject_type="AuthorityPolicy",
            subject_id="policy:root",
            predicate="authorityPrinciple",
            literal="ROOT_POLICY_IS_AUTHORITY",
        ),
        _observation(
            observation_id="slice:workload-policy",
            layer="NORMATIVE",
            subject_type="WorkloadClass",
            subject_id="gpu-heavy",
            predicate="policyDecision",
            literal="DENY",
        ),
        _observation(
            observation_id="slice:route",
            layer="EXECUTION",
            subject_type="ProjectRoute",
            subject_id="route:gpu-heavy",
            predicate="bindsExecutionContract",
            object_type="ExecutionContract",
            object_id="contract:gpu-heavy",
        ),
        _observation(
            observation_id="slice:contract",
            layer="EXECUTION",
            subject_type="ExecutionContract",
            subject_id="contract:gpu-heavy",
            predicate="stateEffectClass",
            literal="CI_VALIDATION_ONLY",
        ),
        _observation(
            observation_id="slice:candidate",
            layer="EXECUTION",
            subject_type="CandidateAction",
            subject_id="candidate:gpu-heavy",
            predicate="hasDecision",
            object_type="DecisionEvaluation",
            object_id="eval:gpu-heavy",
        ),
        _observation(
            observation_id="slice:evaluation",
            layer="EXECUTION",
            subject_type="DecisionEvaluation",
            subject_id="eval:gpu-heavy",
            predicate="boundToCheckpoint",
            object_type="KnowledgeCheckpoint",
            object_id="checkpoint:historical",
        ),
        _observation(
            observation_id="slice:claim-contradiction",
            layer="EPISTEMIC",
            subject_type="Claim",
            subject_id="claim:contradicted",
            predicate="contradicts",
            object_type="Claim",
            object_id="claim:counter",
        ),
        _observation(
            observation_id="slice:evidence-requirement",
            layer="EPISTEMIC",
            subject_type="EvidenceRequirement",
            subject_id="req:claim",
            predicate="requiresEvidence",
            object_type="Method",
            object_id="method:independent-eval",
        ),
        _observation(
            observation_id="slice:method",
            layer="EPISTEMIC",
            subject_type="Method",
            subject_id="method:independent-eval",
            predicate="requiresEvidence",
            object_type="EvidenceRequirement",
            object_id="req:claim",
        ),
        _observation(
            observation_id="slice:knowledge-policy",
            layer="NORMATIVE",
            subject_type="KnowledgeCheckpoint",
            subject_id="checkpoint:historical",
            predicate="policyDecision",
            object_type="AuthorityPolicy",
            object_id="policy:root",
        ),
    ]


def _audit_case() -> tuple[dict[str, Any],dict[str, Any],dict[str, Any]]:
    candidate=make_candidate_action(
        candidate_action_id="hardening:final:audit",
        action_type="RUN_WORKLOAD",
        target={"workload_class":"gpu-small"},
        rationale_observation_ids=["audit:rationale"],
        preconditions=[{"precondition_id":"p:scope","description":"Frozen hardening scope."}],
        required_capabilities=["gpu-small"],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["root-owned workload policy"],
        provenance=["hardening-final-fixture"],
    )
    authority={
        "schema_version":1,
        "observation_id":"audit:policy:gpu-small",
        "layer":"NORMATIVE",
        "subject":{"entity_type":"WorkloadClass","entity_id":"gpu-small"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":"ALLOW"},
        "assertion_state":"ACTIVE",
        "provenance":[{"source_id":"audit-policy-source"}],
        "verification":{"state":"AUTHORITATIVE_RECORD","basis":["synthetic hardening fixture"]},
        "temporal":{"observed_at":None,"valid_from":None,"valid_to":None},
        "supersedes":[],
        "contradicts":[],
        "attributes":{"policy_id":"hardening-policy","policy_epoch":1},
    }
    evaluation=evaluate_candidate_action(
        candidate=candidate,
        observations=[authority],
        semantic_manifest_id="hardening-final-manifest",
        knowledge_checkpoint_ids=["checkpoint:hardening-final"],
        precondition_results=[{
            "precondition_id":"p:scope",
            "result":"PASS",
            "evidence_ids":["precondition:scope:pass"],
        }],
        proposed_executor="synthetic-executor",
    )
    receipt={
        "receipt_id":"receipt:hardening-final",
        "evaluation_id":evaluation["evaluation_id"],
        "candidate_action_id":candidate["candidate_action_id"],
        "status":"COMPLETED",
    }
    return candidate,evaluation,receipt


def run_integrated_hardening_benchmark(repo_root: Path) -> dict[str, Any]:
    base=repo_root/"benchmarks"/"semantic_control_hardening"
    contract=_load(base,"hardening-contract.v1.json")
    replay_fixture=_load(base,"historical-replay-cases.v1.json")
    temporal_fixture=_load(base,"temporal-version-skew-adversaries.v1.json")
    ontology_fixture=_load(base,"ontology-evidence-cases.v1.json")
    scale_result=_load(base,"scale-result.v1.json")
    scale_evidence=_load(base,"scale-evidence.v1.json")
    socket_contract=_load(base,"opaque-extract-socket.v1.json")

    replay_results=replay_historical_cases(replay_fixture["replay_cases"])
    replay_exact=all(
        result["matches_expected"]
        and result["current_state_used_for_decision"] is False
        for result in replay_results
    )

    temporal_cases=temporal_fixture["temporal_cases"]
    supersession=replay_temporal_case(temporal_cases[0])
    contradiction=contradiction_view(temporal_cases[1])
    unknown=unknown_semantics(temporal_cases[2])
    temporal_semantics=(
        supersession["matches_expected"]
        and contradiction["contradictions_preserved"]
        and contradiction["global_truth_resolution"]=="NONE"
        and unknown["assertion_state"]=="UNKNOWN"
        and unknown["coerced_to_false"] is False
    )

    adversary_results=[
        evaluate_frozen_adversary(case)
        for case in temporal_fixture["version_skew_cases"]
    ]
    expected_by_id={
        case["case_id"]:case
        for case in temporal_fixture["version_skew_cases"]
    }
    adversary_exact=True
    authorization_escape_count=0
    for result in adversary_results:
        expected=expected_by_id[result["case_id"]]
        if "expected_decision" in expected:
            adversary_exact=adversary_exact and (
                result.get("decision")==expected["expected_decision"]
                and result.get("reason_code")==expected["expected_reason_code"]
            )
            if expected["expected_decision"] in {"DENY","ABSTAIN"} and result.get("decision")=="ALLOW":
                authorization_escape_count+=1
        elif "expected_receipt_status" in expected:
            adversary_exact=adversary_exact and (
                result.get("receipt_status")==expected["expected_receipt_status"]
                and result.get("authority_effect")==expected["authority_effect"]
            )
        elif "expected_projection_status" in expected:
            adversary_exact=adversary_exact and (
                result.get("projection_status")==expected["expected_projection_status"]
                and result.get("authority_effect")==expected["authority_effect"]
            )

    observations=_slice_observations()
    slice_results=[]
    slice_exact=True
    for case in ontology_fixture["ontology_slice_cases"]:
        result=select_bounded_ontology_slice(
            observations=list(reversed(observations)),
            case=case,
            semantic_manifest_id="hardening-final-slice-manifest",
        )
        second=select_bounded_ontology_slice(
            observations=observations,
            case=case,
            semantic_manifest_id="hardening-final-slice-manifest",
        )
        required_types=set(case["required_entity_types"])
        required_predicates=set(case["required_predicates"])
        case_ok=(
            result==second
            and len(result["observations"])<=case["max_observations"]
            and required_types.issubset(set(result["included_entity_types"]))
            and required_predicates.issubset(set(result["included_predicates"]))
            and result["authority_widening_allowed"] is False
            and bool(result["source_schema_refs"])
        )
        slice_exact=slice_exact and case_ok
        slice_results.append({
            "case_id":case["case_id"],
            "ok":case_ok,
            "observation_count":len(result["observations"]),
            "deterministic_key":result["deterministic_key"],
        })

    evidence_results=[]
    evidence_exact=True
    for case in ontology_fixture["evidence_requirement_cases"]:
        result=derive_frozen_evidence_case(case)
        if case["case_id"]=="E7_MODEL_CONFIDENCE_ONLY":
            case_ok=result is None and case["expected_requirements"]==[]
        else:
            case_ok=(
                result is not None
                and set(result["reason_types"])==set(case["expected_reason_types"])
                and result["execution_effect"]==case["expected_execution_effect"]=="REQUEST_ONLY"
            )
        evidence_exact=evidence_exact and case_ok
        evidence_results.append({
            "case_id":case["case_id"],
            "ok":case_ok,
            "requirement_id":None if result is None else result["requirement_id"],
        })

    candidate,evaluation,receipt=_audit_case()
    chain=build_decision_audit_chain(
        candidate=candidate,
        evaluation=evaluation,
        receipt=receipt,
    )
    verify_valid=verify_decision_audit_chain(
        chain=chain,
        candidate=candidate,
        evaluation=evaluation,
        receipt=receipt,
    )
    tampered=deepcopy(chain)
    tampered["decision"]="DENY"
    verify_tampered=verify_decision_audit_chain(
        chain=tampered,
        candidate=candidate,
        evaluation=evaluation,
        receipt=receipt,
    )
    audit_exact=(
        evaluation["decision"]=="ALLOW"
        and verify_valid["valid"] is True
        and verify_tampered["valid"] is False
        and chain["receipt_binding"]["status"]=="VALID_BINDING"
        and chain["receipt_binding"]["authority_effect"]=="NONE"
        and verify_valid["receipt_authority_effect"]=="NONE"
    )

    scale_integrity=(
        scale_result["all_correct"] is True
        and [item["size"] for item in scale_result["results"]]==[1000,10000,100000]
        and all(item["authorization_escape_count"]==0 for item in scale_result["results"])
        and all(item["deterministic_rebuild"] for item in scale_result["results"])
        and all(item["slice_bounded"] for item in scale_result["results"])
        and all(item["provenance_exact_generated_sources"] for item in scale_result["results"])
        and scale_result["pass_fail_latency_threshold"] is None
        and scale_result["performance_promotion_claim"] is False
        and scale_result["production_mutation"] is False
        and scale_result["scientific_evidence"] is False
        and scale_evidence["ci_run_id"]==40047074263
        and scale_evidence["ci_job_id"]==120731149942
        and scale_evidence["all_correct"] is True
    )

    unaccepted_fixture=next(
        item for item in socket_contract["synthetic_test_descriptors"]
        if item["descriptor_id"]=="synthetic:unaccepted"
    )
    accepted_fixture=next(
        item for item in socket_contract["synthetic_test_descriptors"]
        if item["descriptor_id"]=="synthetic:accepted"
    )
    unaccepted=register_future_extract_descriptor(
        descriptor=materialize_synthetic_descriptor(
            contract=socket_contract,
            fixture=unaccepted_fixture,
        ),
        contract=socket_contract,
    )
    accepted=register_future_extract_descriptor(
        descriptor=materialize_synthetic_descriptor(
            contract=socket_contract,
            fixture=accepted_fixture,
        ),
        contract=socket_contract,
    )
    socket_exact=(
        unaccepted["registration_status"]=="REJECT_BINDING"
        and unaccepted["binding_state"]=="UNBOUND"
        and accepted["registration_status"]=="REGISTER_DESCRIPTOR_ONLY"
        and accepted["binding_state"]=="UNBOUND"
        and accepted["payload_accessed"] is False
        and accepted["payload_schema_defined"] is False
        and accepted["extract_mutation"] is False
        and accepted["production_mutation"] is False
    )

    gates={
        "historical_replay_exact":replay_exact,
        "temporal_semantics_exact":temporal_semantics,
        "version_skew_adversaries_exact":adversary_exact,
        "authorization_escape_count_zero":authorization_escape_count==0,
        "bounded_ontology_slices_exact":slice_exact,
        "evidence_requirements_request_only":evidence_exact,
        "decision_audit_chain_exact":audit_exact,
        "persisted_scale_evidence_integrity":scale_integrity,
        "opaque_extract_socket_exact":socket_exact,
    }

    safety_violation=(
        authorization_escape_count>0
        or accepted["payload_accessed"] is not False
        or accepted["payload_schema_defined"] is not False
        or accepted["extract_mutation"] is not False
        or accepted["production_mutation"] is not False
        or scale_result["production_mutation"] is not False
    )
    if safety_violation:
        verdict="REJECT"
    elif all(gates.values()):
        verdict="HARDENING_PASS"
    else:
        verdict="REVISE"

    body={
        "schema_version":1,
        "benchmark_id":"semantic-control-integrated-hardening-v1",
        "parent_contract_id":contract["contract_id"],
        "verdict":verdict,
        "gates":gates,
        "authorization_escape_count":authorization_escape_count,
        "replay_results":replay_results,
        "temporal_summary":{
            "supersession_matches_expected":supersession["matches_expected"],
            "contradictions_preserved":contradiction["contradictions_preserved"],
            "global_truth_resolution":contradiction["global_truth_resolution"],
            "unknown_coerced_to_false":unknown["coerced_to_false"],
        },
        "adversary_results":adversary_results,
        "slice_results":slice_results,
        "evidence_results":evidence_results,
        "audit_chain_sha256":chain["audit_chain_sha256"],
        "audit_valid":verify_valid["valid"],
        "audit_tamper_detected":verify_tampered["valid"] is False,
        "scale_result_identity":{
            "ci_run_id":scale_evidence["ci_run_id"],
            "ci_job_id":scale_evidence["ci_job_id"],
            "sizes":[item["size"] for item in scale_result["results"]],
            "manifest_hashes":[item["semantic_manifest_sha256"] for item in scale_result["results"]],
        },
        "opaque_extract_socket":{
            "unaccepted_registration":unaccepted["registration_status"],
            "accepted_registration":accepted["registration_status"],
            "accepted_binding_state":accepted["binding_state"],
            "payload_accessed":accepted["payload_accessed"],
            "payload_schema_defined":accepted["payload_schema_defined"],
            "extract_mutation":accepted["extract_mutation"],
            "production_mutation":accepted["production_mutation"],
        },
        "hardening_authorizes_production":False,
        "hardening_binds_extract":False,
        "production_mutation":False,
        "extract_mutation":False,
    }
    return {
        **body,
        "result_sha256":_sha256(body),
    }
