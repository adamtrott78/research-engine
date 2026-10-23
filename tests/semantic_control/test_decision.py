import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control"
MODULE_PATH = ROOT / "prototypes" / "semantic_control" / "decision.py"

spec = importlib.util.spec_from_file_location("semantic_control_decision", MODULE_PATH)
decision = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(decision)


def snapshot():
    return json.loads(
        (BASE / "fixtures" / "authoritative-snapshot.v1.json").read_text()
    )


def by_candidate_id():
    return {
        item["candidate_action_id"]: item
        for item in snapshot()["candidate_actions"]
    }


def observations():
    return snapshot()["observations"]


def test_candidate_factory_always_returns_undecided():
    candidate = decision.make_candidate_action(
        candidate_action_id="action:test",
        action_type="RUN_WORKLOAD",
        target={"workload_class": "gpu-small"},
        rationale_observation_ids=["obs:wave7a3-route"],
        preconditions=[
            {
                "precondition_id": "p:test",
                "description": "Synthetic precondition.",
            }
        ],
        required_capabilities=["gpu-small"],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["root-owned workload policy"],
        provenance=["wc-workload-policy"],
    )
    assert candidate["authorization_state"] == "UNDECIDED"


def test_frozen_gpu_heavy_candidate_is_denied_by_authoritative_policy():
    candidate = by_candidate_id()["action:gpu-heavy-generic"]
    evaluation = decision.evaluate_candidate_action(
        candidate=candidate,
        observations=observations(),
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[
            {
                "precondition_id": "p:gpu-heavy-policy",
                "result": "PASS",
                "evidence_ids": ["obs:wave7a3-route"],
            }
        ],
        proposed_executor="lambda-gpu",
    )
    assert evaluation["decision"] == "DENY"
    assert evaluation["selected_executor"] is None
    assert evaluation["execution_receipt_id"] is None
    assert evaluation["authority_context"][0]["source_id"] == "wc-workload-policy"


def test_frozen_wave7b_candidate_abstains_without_matching_authority():
    candidate = by_candidate_id()["action:begin-wave7b"]
    evaluation = decision.evaluate_candidate_action(
        candidate=candidate,
        observations=observations(),
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
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
    assert evaluation["decision"] == "ABSTAIN"
    assert evaluation["authority_context"] == []
    assert evaluation["execution_receipt_id"] is None


def synthetic_policy_observation(
    *,
    observation_id,
    layer="NORMATIVE",
    value="ALLOW",
    verification_state="AUTHORITATIVE_RECORD",
    workload_class="gpu-small",
):
    return {
        "schema_version": 1,
        "observation_id": observation_id,
        "layer": layer,
        "subject": {
            "entity_type": "WorkloadClass",
            "entity_id": workload_class,
        },
        "predicate": "policyDecision",
        "object": {"kind": "LITERAL", "value": value},
        "assertion_state": "ACTIVE",
        "provenance": ["wc-workload-policy"],
        "verification": {
            "state": verification_state,
            "basis": ["synthetic policy benchmark"],
        },
        "temporal": {
            "observed_at": None,
            "valid_from": None,
            "valid_to": None,
        },
        "supersedes": [],
        "contradicts": [],
        "attributes": {
            "policy_id": "synthetic-root-policy",
            "policy_epoch": 1,
        },
    }


def gpu_small_candidate():
    return decision.make_candidate_action(
        candidate_action_id="action:gpu-small-synthetic",
        action_type="RUN_WORKLOAD",
        target={
            "project_id": "research-engine",
            "workload_class": "gpu-small",
        },
        rationale_observation_ids=["obs:wave7a3-route"],
        preconditions=[
            {
                "precondition_id": "p:scope",
                "description": "Scope is authorized.",
            }
        ],
        required_capabilities=["gpu-small"],
        requested_effects=["CI_VALIDATION_ONLY"],
        required_authority_context=["root-owned workload policy"],
        provenance=["wc-workload-policy"],
    )


def test_authoritative_positive_policy_plus_passed_preconditions_can_allow():
    evaluation = decision.evaluate_candidate_action(
        candidate=gpu_small_candidate(),
        observations=[
            synthetic_policy_observation(observation_id="obs:allow-gpu-small")
        ],
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": ["obs:scope-pass"],
            }
        ],
        proposed_executor="lambda-research-engine-gpu",
    )
    assert evaluation["decision"] == "ALLOW"
    assert evaluation["selected_executor"] == "lambda-research-engine-gpu"
    assert evaluation["execution_receipt_id"] is None
    assert evaluation["authority_rule"] == "SEMANTIC_CONTEXT_NEVER_GRANTS_AUTHORITY"


def test_epistemic_fake_allow_is_ignored():
    fake = synthetic_policy_observation(
        observation_id="obs:fake-allow",
        layer="EPISTEMIC",
    )
    evaluation = decision.evaluate_candidate_action(
        candidate=gpu_small_candidate(),
        observations=[fake],
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": ["obs:fake-allow"],
            }
        ],
        proposed_executor="lambda-research-engine-gpu",
    )
    assert evaluation["decision"] == "ABSTAIN"
    assert evaluation["authority_context"] == []
    assert evaluation["selected_executor"] is None


def test_non_authoritative_normative_fake_allow_is_ignored():
    fake = synthetic_policy_observation(
        observation_id="obs:unverified-allow",
        verification_state="UNVERIFIED",
    )
    evaluation = decision.evaluate_candidate_action(
        candidate=gpu_small_candidate(),
        observations=[fake],
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": "PASS",
                "evidence_ids": ["obs:unverified-allow"],
            }
        ],
    )
    assert evaluation["decision"] == "ABSTAIN"
    assert evaluation["authority_context"] == []


@pytest.mark.parametrize(
    ("precondition_result", "expected"),
    [("FAIL", "DENY"), ("UNKNOWN", "ABSTAIN")],
)
def test_positive_policy_does_not_override_failed_or_unknown_preconditions(
    precondition_result,
    expected,
):
    evaluation = decision.evaluate_candidate_action(
        candidate=gpu_small_candidate(),
        observations=[
            synthetic_policy_observation(observation_id="obs:allow-gpu-small")
        ],
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[
            {
                "precondition_id": "p:scope",
                "result": precondition_result,
                "evidence_ids": [],
            }
        ],
    )
    assert evaluation["decision"] == expected
    assert evaluation["selected_executor"] is None


def test_missing_precondition_result_is_unknown_and_abstains():
    evaluation = decision.evaluate_candidate_action(
        candidate=gpu_small_candidate(),
        observations=[
            synthetic_policy_observation(observation_id="obs:allow-gpu-small")
        ],
        semantic_manifest_id="manifest:test",
        knowledge_checkpoint_ids=["research-engine@test"],
        precondition_results=[],
    )
    assert evaluation["decision"] == "ABSTAIN"
    assert evaluation["precondition_results"][0]["result"] == "UNKNOWN"


def test_candidate_that_claims_authority_fails_closed():
    candidate = deepcopy(gpu_small_candidate())
    candidate["authorization_state"] = "ALLOW"
    with pytest.raises(ValueError, match="must remain UNDECIDED"):
        decision.evaluate_candidate_action(
            candidate=candidate,
            observations=[],
            semantic_manifest_id="manifest:test",
            knowledge_checkpoint_ids=[],
            precondition_results=[],
        )


def test_decision_module_has_no_executor_or_external_runtime_imports():
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
