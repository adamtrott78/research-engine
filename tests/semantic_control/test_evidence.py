import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control"
MODULE_PATH = ROOT / "prototypes" / "semantic_control" / "evidence.py"

spec = importlib.util.spec_from_file_location("semantic_control_evidence", MODULE_PATH)
evidence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evidence)


def snapshot_observations():
    data = json.loads(
        (BASE / "fixtures" / "authoritative-snapshot.v1.json").read_text()
    )
    return data["observations"]


def synthetic_observation(
    observation_id,
    *,
    attributes=None,
    verification_state="MACHINE_CHECKED",
    valid_to=None,
):
    return {
        "schema_version": 1,
        "observation_id": observation_id,
        "layer": "EPISTEMIC",
        "subject": {"entity_type": "Claim", "entity_id": observation_id},
        "predicate": "hasEvidenceState",
        "object": {"kind": "LITERAL", "value": True},
        "assertion_state": "ACTIVE",
        "provenance": ["re-sql-schema"],
        "verification": {
            "state": verification_state,
            "basis": ["synthetic evidence-rule test"],
        },
        "temporal": {
            "observed_at": "2026-01-01T00:00:00Z",
            "valid_from": None,
            "valid_to": valid_to,
        },
        "supersedes": [],
        "contradicts": [],
        "attributes": dict(attributes or {}),
    }


def test_frozen_contradiction_derives_one_request_only_requirement():
    requirements = evidence.derive_evidence_requirements(snapshot_observations())
    contradiction = [
        req for req in requirements if req["reason_types"] == ["CONTRADICTION"]
    ]
    assert len(contradiction) == 1
    req = contradiction[0]
    assert req["requirement_id"] == "req:resolve-method-x-contradiction"
    assert req["target_ids"] == [
        "obs:synthetic-claim-a",
        "obs:synthetic-claim-b",
    ]
    assert req["requested_evidence_kinds"] == ["INDEPENDENT_EVALUATION"]
    assert req["execution_effect"] == "REQUEST_ONLY"


def test_all_six_durable_reason_types_are_supported():
    observations = [
        *snapshot_observations(),
        synthetic_observation(
            "obs:missing",
            attributes={"missing_facets": ["hardware_shift_evaluation"]},
        ),
        synthetic_observation(
            "obs:source",
            attributes={
                "source_level": "ABSTRACT",
                "required_source_level": "FULLTEXT",
            },
        ),
        synthetic_observation(
            "obs:verify",
            verification_state="UNVERIFIED",
        ),
        synthetic_observation(
            "obs:stale",
            valid_to="2026-02-01T00:00:00Z",
        ),
        synthetic_observation(
            "obs:benchmark",
            attributes={"benchmark_sufficient": False},
        ),
    ]
    requirements = evidence.derive_evidence_requirements(
        observations,
        as_of="2026-03-01T00:00:00Z",
    )
    reasons = {
        reason
        for requirement in requirements
        for reason in requirement["reason_types"]
    }
    assert reasons == {
        "CONTRADICTION",
        "MISSING_FACET",
        "SOURCE_LEVEL_INSUFFICIENT",
        "VERIFICATION_INSUFFICIENT",
        "STALE",
        "BENCHMARK_CALIBRATED_INSUFFICIENCY",
    }
    assert all(req["execution_effect"] == "REQUEST_ONLY" for req in requirements)


def test_raw_model_confidence_alone_never_triggers_escalation():
    observation = synthetic_observation(
        "obs:confidence-only",
        attributes={"model_confidence": 0.0001},
    )
    assert evidence.derive_evidence_requirements([observation]) == []


def test_requirement_derivation_is_deterministic_under_reordering():
    observations = snapshot_observations()
    first = evidence.derive_evidence_requirements(observations)
    second = evidence.derive_evidence_requirements(list(reversed(observations)))
    assert first == second


def test_ontology_slice_is_bounded_and_deterministic():
    first = evidence.build_ontology_slice(
        profile_name="scientific_evidence_gap",
        semantic_manifest_id="manifest:test",
        source_schema_refs=["re-sql-schema", "re-domain-model"],
        context={"purpose": "evidence-gap derivation"},
    )
    second = evidence.build_ontology_slice(
        profile_name="scientific_evidence_gap",
        semantic_manifest_id="manifest:test",
        source_schema_refs=["re-domain-model", "re-sql-schema"],
        context={"purpose": "evidence-gap derivation"},
    )
    assert first == second
    assert first["slice_id"] == "slice:scientific-gap-v1"
    assert first["included_entity_types"] == [
        "Claim",
        "EvidenceRequirement",
        "Method",
    ]
    assert first["included_predicates"] == [
        "contradicts",
        "improvesHardwareShiftRobustness",
        "requiresEvidence",
    ]


def test_unknown_slice_profile_fails_closed():
    with pytest.raises(ValueError, match="unknown ontology slice profile"):
        evidence.build_ontology_slice(
            profile_name="not-real",
            semantic_manifest_id="manifest:test",
            source_schema_refs=["re-sql-schema"],
        )


def test_unknown_source_level_fails_closed():
    observation = synthetic_observation(
        "obs:unknown-source",
        attributes={
            "source_level": "MAGIC",
            "required_source_level": "FULLTEXT",
        },
    )
    with pytest.raises(ValueError, match="unknown source level"):
        evidence.derive_evidence_requirements([observation])


def test_evidence_module_has_no_external_or_production_runtime_imports():
    text = MODULE_PATH.read_text().lower()
    forbidden_import_fragments = [
        "import psycopg",
        "from psycopg",
        "import sqlalchemy",
        "from sqlalchemy",
        "import requests",
        "from requests",
        "import urllib",
        "from urllib",
        "import socket",
        "from socket",
        "import subprocess",
        "from subprocess",
        "import openai",
        "from openai",
        "import research_engine",
        "from research_engine",
    ]
    for fragment in forbidden_import_fragments:
        assert fragment not in text
