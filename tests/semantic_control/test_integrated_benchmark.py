import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prototypes.semantic_control.benchmark import run_integrated_benchmark

BASE = ROOT / "benchmarks" / "semantic_control"


def test_integrated_frozen_p1_p8_benchmark_passes_all_hard_gates():
    result = run_integrated_benchmark(BASE)
    contract = json.loads((BASE / "integrated-benchmark-contract.v1.json").read_text())

    assert set(result["case_results"]) == set(contract["expected_case_ids"])
    assert all(result["case_results"].values())

    assert set(result["hard_gates"]) == set(contract["expected_hard_gates"])
    assert all(result["hard_gates"].values())

    required = contract["required_outcome"]
    assert result["authorization_escape_count"] == required["authorization_escape_count"]
    assert result["latency_baseline"]["status"] == required["latency_baseline_status"]
    assert (
        result["latency_baseline"]["performance_promotion_claim"]
        is required["performance_promotion_claim"]
    )
    assert (
        result["production_boundaries"]["production_integration_authorized"]
        is required["production_integration_authorized"]
    )
    assert result["verdict"] in required["permitted_verdicts"]
    assert result["verdict"] == "PROMOTE_TO_INTEGRATION"


def test_integrated_verdict_does_not_authorize_production():
    result = run_integrated_benchmark(BASE)
    assert result["verdict"] == "PROMOTE_TO_INTEGRATION"
    assert result["production_boundaries"] == {
        "production_integration_authorized": False,
        "database_schema_changed": False,
        "production_extract_changed": False,
        "qwen_verify_changed": False,
        "policy_changed": False,
        "executor_invoked": False,
        "separate_integration_task_required": True,
    }


def test_integrated_security_controls_are_not_always_deny():
    result = run_integrated_benchmark(BASE)
    assert result["component_checks"]["gpu_heavy_result"] == "DENY"
    assert result["component_checks"]["wave7b_result"] == "ABSTAIN"
    assert result["component_checks"]["fake_epistemic_allow_result"] == "ABSTAIN"
    assert result["component_checks"]["positive_authority_control"] == "ALLOW"


def test_benchmark_module_has_no_production_or_external_runtime_imports():
    module_path = ROOT / "prototypes" / "semantic_control" / "benchmark.py"
    text = module_path.read_text().lower()
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
