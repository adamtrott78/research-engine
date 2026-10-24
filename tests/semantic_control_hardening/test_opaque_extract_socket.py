import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
CONTRACT_PATH=ROOT/"benchmarks"/"semantic_control_hardening"/"opaque-extract-socket.v1.json"


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


def materialized_fixture(descriptor_id):
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        materialize_synthetic_descriptor,
    )
    contract=load_contract()
    fixture=next(
        item for item in contract["synthetic_test_descriptors"]
        if item["descriptor_id"]==descriptor_id
    )
    return contract,materialize_synthetic_descriptor(contract=contract,fixture=fixture)


def test_unaccepted_descriptor_is_rejected_without_payload_access():
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        register_future_extract_descriptor,
    )
    contract,descriptor=materialized_fixture("synthetic:unaccepted")
    result=register_future_extract_descriptor(descriptor=descriptor,contract=contract)
    assert result["registration_status"]=="REJECT_BINDING"
    assert result["reason_code"]=="SCIENTIFIC_ACCEPTANCE_NOT_ESTABLISHED"
    assert result["binding_state"]=="UNBOUND"
    assert result["payload_accessed"] is False
    assert result["payload_schema_defined"] is False
    assert result["extract_mutation"] is False
    assert result["production_mutation"] is False


def test_accepted_descriptor_registers_metadata_only_and_stays_unbound():
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        register_future_extract_descriptor,
    )
    contract,descriptor=materialized_fixture("synthetic:accepted")
    first=register_future_extract_descriptor(descriptor=descriptor,contract=contract)
    second=register_future_extract_descriptor(descriptor=deepcopy(descriptor),contract=contract)
    assert first==second
    assert first["registration_status"]=="REGISTER_DESCRIPTOR_ONLY"
    assert first["binding_state"]=="UNBOUND"
    assert first["adapter_target_contract"]=="SemanticObservationV1"
    assert first["concrete_adapter_present"] is False
    assert first["payload_accessed"] is False
    assert first["payload_schema_defined"] is False
    assert first["extract_mutation"] is False
    assert first["production_mutation"] is False
    assert len(first["registration_sha256"])==64


def test_real_descriptor_requires_explicit_schema_version():
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        validate_future_extract_descriptor,
    )
    contract=load_contract()
    raw=deepcopy(contract["synthetic_test_descriptors"][1])
    raw.pop("expected_registration")
    with pytest.raises(ValueError,match="missing required descriptor fields"):
        validate_future_extract_descriptor(descriptor=raw,contract=contract)


@pytest.mark.parametrize("forbidden_field",[
    "payload_fields","prompt","retrieval_method","model",
    "qwen","codex","ontology_required_output","rdf_required_output",
])
def test_forbidden_extract_implementation_or_payload_fields_fail_closed(forbidden_field):
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        register_future_extract_descriptor,
    )
    contract,descriptor=materialized_fixture("synthetic:accepted")
    descriptor["extra_metadata"]={forbidden_field:"forbidden"}
    with pytest.raises(ValueError,match="forbidden EXTRACT"):
        register_future_extract_descriptor(descriptor=descriptor,contract=contract)


def test_invalid_identity_and_nonopaque_schema_fail_closed():
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        register_future_extract_descriptor,
    )
    contract,descriptor=materialized_fixture("synthetic:accepted")

    bad_commit=deepcopy(descriptor)
    bad_commit["external_commit_sha"]="not-a-sha"
    with pytest.raises(ValueError,match="external_commit_sha"):
        register_future_extract_descriptor(descriptor=bad_commit,contract=contract)

    bad_blob=deepcopy(descriptor)
    bad_blob["external_contract_artifact"]["git_blob_sha"]="not-a-blob"
    with pytest.raises(ValueError,match="git_blob_sha"):
        register_future_extract_descriptor(descriptor=bad_blob,contract=contract)

    nonopaque=deepcopy(descriptor)
    nonopaque["payload_schema_opaque"]=False
    with pytest.raises(ValueError,match="payload_schema_opaque"):
        register_future_extract_descriptor(descriptor=nonopaque,contract=contract)


def test_accepted_descriptor_cannot_self_declare_bound_without_concrete_adapter():
    from prototypes.semantic_control_hardening.opaque_extract_socket import (
        register_future_extract_descriptor,
    )
    contract,descriptor=materialized_fixture("synthetic:accepted")
    descriptor["binding_state"]="BOUND"
    result=register_future_extract_descriptor(descriptor=descriptor,contract=contract)
    assert result["registration_status"]=="REJECT_BINDING"
    assert result["reason_code"]=="CONCRETE_ADAPTER_REQUIRED_BEFORE_BINDING"
    assert result["binding_state"]=="UNBOUND"
    assert result["payload_accessed"] is False


def test_socket_source_has_no_external_runtime_or_extract_science_dependencies():
    text=(ROOT/"prototypes"/"semantic_control_hardening"/"opaque_extract_socket.py").read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "research_engine.postgres","workspace_control_adapter",
    ]:
        assert token not in text
