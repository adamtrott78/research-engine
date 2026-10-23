import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control"
MODULE_PATH = ROOT / "prototypes" / "semantic_control" / "twin.py"

spec = importlib.util.spec_from_file_location("semantic_control_twin", MODULE_PATH)
twin_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(twin_module)
WorkspaceSemanticTwin = twin_module.WorkspaceSemanticTwin


def load_snapshot():
    return json.loads(
        (BASE / "fixtures" / "authoritative-snapshot.v1.json").read_text()
    )


def build_twin():
    return WorkspaceSemanticTwin(load_snapshot()["observations"])


def test_rebuild_is_deterministic_under_input_reordering():
    observations = load_snapshot()["observations"]
    first = WorkspaceSemanticTwin(observations)
    second = WorkspaceSemanticTwin(list(reversed(observations)))
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.all_observations() == second.all_observations()


def test_p1_provenance_traversal_reaches_frozen_sources():
    twin = build_twin()
    provenance = twin.trace_provenance("obs:wave7a3-verdict")
    assert provenance == ["re-wave7a3-result", "wc-extract-task"]


def test_p2_contradiction_preserves_both_claims_without_resolution():
    twin = build_twin()
    a = twin.get("obs:synthetic-claim-a")
    b = twin.get("obs:synthetic-claim-b")
    assert a["object"]["value"] is True
    assert b["object"]["value"] is False
    assert [x["observation_id"] for x in twin.contradictions(a["observation_id"])] == [
        "obs:synthetic-claim-b"
    ]
    assert [x["observation_id"] for x in twin.contradictions(b["observation_id"])] == [
        "obs:synthetic-claim-a"
    ]


def test_p3_supersession_retains_history_and_replays_time():
    twin = build_twin()
    history = twin.supersession_history(
        subject_type="Claim",
        subject_id="claim-versioned",
        predicate="hasValue",
    )
    assert [x["observation_id"] for x in history] == [
        "obs:synthetic-old",
        "obs:synthetic-new",
    ]
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
    assert [x["object"]["value"] for x in jan] == ["old"]
    assert [x["object"]["value"] for x in feb] == ["new"]


def test_p6_task_scientific_state_query_matches_frozen_expected_answer():
    twin = build_twin()
    result = twin.query_task_scientific_state(
        task_id="research-engine-local-extract-redesign-v1",
        experiment_id="wave7a3-author-anchored",
    )
    assert result["task_status"] == "CHECKPOINTED"
    assert result["scientific_disposition"] == "WAVE7A3_REJECT"
    assert result["wave7b_authorized"] is False
    assert result["evidence_observation_ids"] == [
        "obs:extract-task-checkpointed",
        "obs:wave7a3-verdict",
    ]


def test_projection_is_read_only_with_respect_to_input_and_returned_records():
    observations = load_snapshot()["observations"]
    original = deepcopy(observations)
    twin = WorkspaceSemanticTwin(observations)
    returned = twin.get("obs:wave7a3-verdict")
    returned["attributes"]["wave7b_authorized"] = True
    assert observations == original
    assert twin.get("obs:wave7a3-verdict")["attributes"]["wave7b_authorized"] is False


def test_duplicate_observation_identity_fails_closed_without_mutating_input():
    observations = load_snapshot()["observations"]
    duplicate = deepcopy(observations[0])
    duplicate["object"]["value"] = "CONFLICTING_DUPLICATE"
    candidate = observations + [duplicate]
    original = deepcopy(candidate)
    with pytest.raises(ValueError, match="duplicate observation_id"):
        WorkspaceSemanticTwin(candidate)
    assert candidate == original


def test_missing_contradiction_target_fails_closed():
    observations = load_snapshot()["observations"]
    changed = deepcopy(observations)
    record = next(x for x in changed if x["observation_id"] == "obs:synthetic-claim-a")
    record["contradicts"] = ["obs:missing"]
    twin = WorkspaceSemanticTwin(changed)
    with pytest.raises(ValueError, match="missing contradiction"):
        twin.contradictions("obs:synthetic-claim-a")


def test_twin_module_has_no_external_or_production_runtime_imports():
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
