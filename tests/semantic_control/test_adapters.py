import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control"
MODULE_PATH = ROOT / "prototypes" / "semantic_control" / "adapters.py"

spec = importlib.util.spec_from_file_location("semantic_control_adapters", MODULE_PATH)
adapters = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(adapters)


def load(path):
    return json.loads((BASE / path).read_text())


def build():
    return adapters.build_semantic_manifest(
        load("fixtures/adapter-inputs.v1.json"),
        load("fixture-manifest.v1.json"),
    )


def test_adapter_manifest_is_deterministic_under_input_reordering():
    fixture_inputs = load("fixtures/adapter-inputs.v1.json")
    fixture_manifest = load("fixture-manifest.v1.json")
    first = adapters.build_semantic_manifest(fixture_inputs, fixture_manifest)
    fixture_inputs["inputs"] = list(reversed(fixture_inputs["inputs"]))
    second = adapters.build_semantic_manifest(fixture_inputs, fixture_manifest)
    assert first == second


def test_canonical_hash_is_64_hex():
    manifest = build()
    digest = manifest["semantic_manifest_sha256"]
    assert len(digest) == 64
    int(digest, 16)


def test_expected_four_authoritative_observations_are_projected():
    manifest = build()
    by_id = {obs["observation_id"]: obs for obs in manifest["observations"]}
    assert set(by_id) == {
        "obs:wave7a3-verdict",
        "obs:extract-task-checkpointed",
        "obs:gpu-heavy-denied",
        "obs:wave7a3-route",
    }
    assert by_id["obs:wave7a3-verdict"]["object"]["value"] == "WAVE7A3_REJECT"
    assert by_id["obs:gpu-heavy-denied"]["layer"] == "NORMATIVE"


def test_provenance_expands_to_exact_frozen_source_identity():
    manifest = build()
    source_manifest = load("fixture-manifest.v1.json")
    adapters.validate_manifest_provenance(manifest, source_manifest)
    expected = {
        source["source_id"]: source
        for source in source_manifest["sources"]
    }
    for observation in manifest["observations"]:
        for provenance in observation["provenance"]:
            source = expected[provenance["source_id"]]
            assert provenance["repository"] == source["repository"]
            assert provenance["commit_sha"] == source["commit_sha"]
            assert provenance["path"] == source["path"]
            assert provenance["git_blob_sha"] == source["git_blob_sha"]


def test_unknown_adapter_fails_closed():
    fixture_inputs = load("fixtures/adapter-inputs.v1.json")
    fixture_manifest = load("fixture-manifest.v1.json")
    fixture_inputs["inputs"][0]["adapter"] = "not_real"
    with pytest.raises(ValueError, match="unknown adapter"):
        adapters.build_semantic_manifest(fixture_inputs, fixture_manifest)


def test_unknown_source_fails_closed():
    fixture_inputs = load("fixtures/adapter-inputs.v1.json")
    fixture_manifest = load("fixture-manifest.v1.json")
    fixture_inputs["inputs"][0]["source_id"] = "missing-source"
    with pytest.raises(ValueError, match="unknown source_id"):
        adapters.build_semantic_manifest(fixture_inputs, fixture_manifest)


def test_adapter_module_has_no_production_or_external_runtime_imports():
    text = MODULE_PATH.read_text()
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
    lowered = text.lower()
    for fragment in forbidden_import_fragments:
        assert fragment not in lowered
