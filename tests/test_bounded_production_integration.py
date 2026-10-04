import inspect
import json
import subprocess
from pathlib import Path

from research_engine.bounded_integration import (
    build_integration_report,
    compare_fingerprints,
    validate_sidecar_smoke,
)
from research_engine.semantic_pipeline import ResearchContextHydrator


ROOT = Path(__file__).parents[1]


def test_sidecar_smoke_validation_preserves_authoritative_semantic_baseline():
    result = validate_sidecar_smoke(
        {
            "artifact_id": "source-artifact:test",
            "cache_miss_then_hit": True,
            "first": {
                "status": "COMPLETED",
                "authority": "DERIVED_SIDECAR_ONLY",
                "semantic_baseline": "AUTHORITATIVE_PDF_HYDRATOR",
                "routing_summary": {},
            },
            "second": {"status": "CACHE_HIT"},
            "broker_final_health": {"enabled": True},
        }
    )
    assert result["pass"] is True
    assert result["authority"] == "DERIVED_SIDECAR_ONLY"
    assert result["semantic_baseline"] == "AUTHORITATIVE_PDF_HYDRATOR"


def test_scientific_fingerprint_comparison_detects_any_change():
    before = {
        "tables": {
            "works": {"rows": 1, "sha256": "a"},
            "evidence_objects": {"rows": 2, "sha256": "b"},
        }
    }
    same = {
        "tables": {
            "works": {"rows": 1, "sha256": "a"},
            "evidence_objects": {"rows": 2, "sha256": "b"},
        }
    }
    changed = {
        "tables": {
            "works": {"rows": 1, "sha256": "a"},
            "evidence_objects": {"rows": 3, "sha256": "c"},
        }
    }
    assert compare_fingerprints(before, same)["unchanged"] is True
    result = compare_fingerprints(before, changed)
    assert result["unchanged"] is False
    assert result["changed_tables"] == ["evidence_objects"]


def test_bounded_report_requires_every_gate():
    common = {
        "research_engine_head": "abc",
        "preflight": {"safe_for_idle_daemon_smoke": True},
        "sidecar": {"pass": True},
        "workers": {
            "daemon_pid": 123,
            "workers": [],
            "worker_registration_pass": True,
            "quiescence": {"safe_for_idle_daemon_smoke": True},
        },
        "fingerprint_comparison": {"unchanged": True},
        "telemetry": {"telemetry_pass": True},
    }
    report = build_integration_report(**common)
    assert report["decision"] == "PASS"
    assert report["semantic_calls_performed"] == 0
    assert report["scientific_state_materialized"] is False
    assert report["semantic_baseline"] == "AUTHORITATIVE_PDF_HYDRATOR"

    common["telemetry"] = {"telemetry_pass": False}
    assert build_integration_report(**common)["decision"] == "FAIL"


def test_semantic_hydrator_has_no_sidecar_substitution():
    source = inspect.getsource(ResearchContextHydrator)
    assert "document_sidecar" not in source
    assert "paper_package" not in source


def test_production_env_manages_sidecar_feature_gate_and_timeout():
    example = (ROOT / "deploy" / "research-engine.env.example").read_text()
    configure = (
        ROOT / "deploy" / "configure-production-local-inference.sh"
    ).read_text()

    for text in (example, configure):
        assert "RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED=1" in text
        assert "RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED=1" in text
        assert "RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX=0" in text
        assert "RESEARCH_ENGINE_DOCUMENT_SIDECAR_TIMEOUT_SECONDS=900" in text


def test_bounded_integration_contract_locks_authority_and_cascade():
    data = json.loads(
        (
            ROOT
            / "benchmarks"
            / "document_intelligence"
            / "bounded-production-integration-contract.v1.json"
        ).read_text()
    )
    authority = "\n".join(data["authority_invariants"])
    routing = "\n".join(data["routing_contract"])
    daemon = "\n".join(data["bounded_daemon_gate"])

    assert "before any sidecar compile" in authority
    assert "fail-open" in authority
    assert "DERIVED_SIDECAR_ONLY" in authority
    assert "authoritative-PDF ResearchContextHydrator" in authority
    assert "FAST empty table grids" in routing
    assert "TableFormer ACCURATE" in routing
    assert "Paddle table crop" in routing
    assert "Scientific tables are fingerprinted" in daemon
    assert "Daemon is stopped unconditionally" in daemon


def test_bounded_integration_runner_is_finite_and_rollback_safe():
    script = (
        ROOT / "deploy" / "run-bounded-production-integration.sh"
    ).read_text()

    assert "trap cleanup EXIT INT TERM" in script
    assert "bounded-integration-preflight" in script
    assert "document-sidecar-smoke" in script
    assert "bounded-integration-validate-sidecar" in script
    assert "bounded-integration-fingerprint" in script
    assert "bounded-integration-workers" in script
    assert "bounded-integration-compare-fingerprints" in script
    assert "BOUNDED_PRODUCTION_INTEGRATION=PASS" in script
    assert "SCIENTIFIC_FINGERPRINT=UNCHANGED" in script
    assert "systemctl --user start" in script
    assert "systemctl --user stop" in script
    assert "sleep 6" in script

    assert "submit-run" not in script
    assert "autonomous-run" not in script
    assert "paper-package-ab-replay" not in script
    assert "OPENAI_API_KEY" not in script


def test_bounded_integration_shell_parses():
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "deploy" / "run-bounded-production-integration.sh"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
