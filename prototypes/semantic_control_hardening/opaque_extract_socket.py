from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any


_HEX40=re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nested_keys(value: Any) -> set[str]:
    keys:set[str]=set()
    if isinstance(value,dict):
        for key,item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value,list):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


def materialize_synthetic_descriptor(
    *,
    contract: dict[str,Any],
    fixture: dict[str,Any],
) -> dict[str,Any]:
    descriptor=deepcopy(fixture)
    descriptor["schema_version"]=contract["schema_version"]
    descriptor.pop("expected_registration",None)
    return descriptor


def validate_future_extract_descriptor(
    *,
    descriptor: dict[str,Any],
    contract: dict[str,Any],
) -> dict[str,Any]:
    missing=[
        field for field in contract["descriptor_required_fields"]
        if field not in descriptor
    ]
    if missing:
        raise ValueError(f"missing required descriptor fields: {sorted(missing)}")

    forbidden=set(contract["forbidden_descriptor_fields"])
    present_forbidden=sorted(forbidden & _nested_keys(descriptor))
    if present_forbidden:
        raise ValueError(
            "forbidden EXTRACT implementation/payload fields: "
            + ",".join(present_forbidden)
        )

    if descriptor["schema_version"]!=contract["schema_version"]:
        raise ValueError("descriptor schema_version mismatch")
    if descriptor["payload_schema_opaque"] is not True:
        raise ValueError("payload_schema_opaque must be true")
    if descriptor["adapter_target_contract"]!="SemanticObservationV1":
        raise ValueError("adapter_target_contract must be SemanticObservationV1")

    commit_sha=descriptor["external_commit_sha"]
    if not isinstance(commit_sha,str) or _HEX40.fullmatch(commit_sha) is None:
        raise ValueError("external_commit_sha must be exact 40-character lowercase hex")

    artifact=descriptor["external_contract_artifact"]
    if not isinstance(artifact,dict):
        raise ValueError("external_contract_artifact must be an object")
    artifact_missing=[
        field for field in contract["external_contract_artifact_required_fields"]
        if field not in artifact
    ]
    if artifact_missing:
        raise ValueError(
            f"missing external contract artifact fields: {sorted(artifact_missing)}"
        )
    blob_sha=artifact["git_blob_sha"]
    if not isinstance(blob_sha,str) or _HEX40.fullmatch(blob_sha) is None:
        raise ValueError("external contract git_blob_sha must be exact 40-character lowercase hex")
    if not artifact["path"] or not artifact["schema_id"]:
        raise ValueError("external contract path/schema_id must be non-empty")

    if descriptor["binding_state"] not in {"UNBOUND","BOUND"}:
        raise ValueError("binding_state must be UNBOUND or BOUND")

    body={
        "schema_version":descriptor["schema_version"],
        "descriptor_id":descriptor["descriptor_id"],
        "external_owner_task_id":descriptor["external_owner_task_id"],
        "external_repository":descriptor["external_repository"],
        "external_commit_sha":descriptor["external_commit_sha"],
        "external_contract_artifact":deepcopy(artifact),
        "scientific_disposition":descriptor["scientific_disposition"],
        "acceptance_evidence":deepcopy(descriptor["acceptance_evidence"]),
        "payload_schema_opaque":True,
        "adapter_target_contract":"SemanticObservationV1",
        "binding_state":descriptor["binding_state"],
    }
    return {
        **body,
        "descriptor_sha256":_sha256(body),
        "payload_accessed":False,
        "payload_schema_defined":False,
        "extract_mutation":False,
        "production_mutation":False,
    }


def register_future_extract_descriptor(
    *,
    descriptor: dict[str,Any],
    contract: dict[str,Any],
) -> dict[str,Any]:
    validated=validate_future_extract_descriptor(
        descriptor=descriptor,
        contract=contract,
    )

    accepted=(
        validated["scientific_disposition"]=="ACCEPTED_OUTPUT_CONTRACT"
        and bool(validated["acceptance_evidence"])
    )
    if not accepted:
        return {
            "schema_version":1,
            "descriptor_id":validated["descriptor_id"],
            "registration_status":"REJECT_BINDING",
            "reason_code":"SCIENTIFIC_ACCEPTANCE_NOT_ESTABLISHED",
            "binding_state":"UNBOUND",
            "registered_descriptor_sha256":validated["descriptor_sha256"],
            "payload_accessed":False,
            "payload_schema_defined":False,
            "adapter_target_contract":"SemanticObservationV1",
            "extract_mutation":False,
            "production_mutation":False,
        }

    if validated["binding_state"]!="UNBOUND":
        return {
            "schema_version":1,
            "descriptor_id":validated["descriptor_id"],
            "registration_status":"REJECT_BINDING",
            "reason_code":"CONCRETE_ADAPTER_REQUIRED_BEFORE_BINDING",
            "binding_state":"UNBOUND",
            "registered_descriptor_sha256":validated["descriptor_sha256"],
            "payload_accessed":False,
            "payload_schema_defined":False,
            "adapter_target_contract":"SemanticObservationV1",
            "extract_mutation":False,
            "production_mutation":False,
        }

    registration_body={
        "schema_version":1,
        "descriptor_id":validated["descriptor_id"],
        "registration_status":"REGISTER_DESCRIPTOR_ONLY",
        "reason_code":"ACCEPTED_EXTERNAL_CONTRACT_METADATA_REGISTERED",
        "binding_state":"UNBOUND",
        "external_owner_task_id":validated["external_owner_task_id"],
        "external_repository":validated["external_repository"],
        "external_commit_sha":validated["external_commit_sha"],
        "external_contract_artifact":deepcopy(validated["external_contract_artifact"]),
        "scientific_disposition":validated["scientific_disposition"],
        "acceptance_evidence":deepcopy(validated["acceptance_evidence"]),
        "registered_descriptor_sha256":validated["descriptor_sha256"],
        "payload_accessed":False,
        "payload_schema_defined":False,
        "adapter_target_contract":"SemanticObservationV1",
        "concrete_adapter_present":False,
        "extract_mutation":False,
        "production_mutation":False,
    }
    return {
        **registration_body,
        "registration_sha256":_sha256(registration_body),
    }
