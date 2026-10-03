import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_production_env_enables_validated_shared_local_inference():
    text = (
        ROOT / "deploy" / "research-engine.env.example"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1" in text
    assert "RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING=8" in text
    assert "RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS=900" in text
    assert "RESEARCH_ENGINE_FORMULA_GPU_INDEX=0" in text
    assert "RESEARCH_ENGINE_TABLE_GPU_INDEX=1" in text
    assert "RESEARCH_ENGINE_VERIFY_PAGE_RADIUS=0" in text


def test_codeformula_license_review_allows_internal_deployment_with_share_obligation():
    data = json.loads(
        (
            ROOT
            / "benchmarks"
            / "document_intelligence"
            / "codeformulav2-license-review.v1.json"
        ).read_text(encoding="utf-8")
    )

    assert data["model_repo"] == "docling-project/CodeFormulaV2"
    assert data["pinned_revision"] == (
        "0284bc97b73c88776957d00d6695dd63a5a8784e"
    )
    assert data["declared_license"] == "cdla-permissive-2.0"
    assert data["decision"] == (
        "PASS_INTERNAL_DEPLOYMENT_WITH_SHARING_OBLIGATION"
    )
    assert any(
        "redistribute" in item.lower()
        for item in data["production_requirements"]
    )


def test_lambda_production_acceptance_contract_is_bounded_and_fail_closed():
    contract = json.loads(
        (
            ROOT
            / "benchmarks"
            / "document_intelligence"
            / "lambda-production-acceptance-contract.v1.json"
        ).read_text(encoding="utf-8")
    )

    required = contract["required_state"]
    assert required["research_daemon"] == "inactive/dead PID 0 before and after"
    assert required["local_inference_enabled"] is True
    assert required["formula_gpu_index"] == 0
    assert required["table_gpu_index"] == 1

    safety = "\n".join(contract["safety_checks"])
    assert "no active RUNNING research run" in safety
    assert "no QUEUED/RUNNING non-HUMAN jobs" in safety
    assert "never starts research-engine-daemon.service" in safety
    assert "no semantic model call" in safety


def test_production_env_sync_is_backup_first_and_requires_stopped_daemon():
    script = (
        ROOT / "deploy" / "configure-production-local-inference.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_BE_STOPPED_BEFORE_ENV_UPDATE" in script
    assert "cp -a \"$envf\" \"$backup\"" in script
    assert "BEGIN MANAGED LOCAL DOCUMENT INFERENCE" in script
    assert "RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED=1" in script
    assert "RESEARCH_ENGINE_FORMULA_GPU_INDEX=0" in script
    assert "RESEARCH_ENGINE_TABLE_GPU_INDEX=1" in script
    assert "ROLLBACK=" in script
    assert "systemctl --user start" not in script


def test_lambda_production_acceptance_never_starts_research_or_models():
    script = (
        ROOT / "deploy" / "run-lambda-production-acceptance.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in script
    assert "ACTIVE_RUNNING_RESEARCH_RUN_PRESENT" in script
    assert "QUEUED_OR_RUNNING_NONHUMAN_WORK_PRESENT" in script
    assert "SYSTEMD_UNIT_DRIFT_FROM_REPOSITORY" in script
    assert "OPENAI_API_KEY_FORBIDDEN_IN_PRODUCTION_ENV" in script
    assert "LOCAL_INFERENCE_NOT_ENABLED" in script
    assert "AUTHORITATIVE PDF HYDRATION SMOKE" in script
    assert "broker_constructed_without_model_process_spawn" in script
    assert "semantic_calls_performed" in script
    assert "model_server_processes_spawned" in script
    assert "LAMBDA_PRODUCTION_ACCEPTANCE=PASS" in script
    assert "workspace-control-tmp" in script

    assert "systemctl --user start" not in script
    assert "paper-package-ab-replay" not in script
    assert "local_inference_benchmark" not in script
    assert ".formula_page(" not in script
    assert ".table_crop(" not in script
    assert "pip install" not in script
    assert "python3 -m venv" not in script


def test_acceptance_broker_bootstrap_is_zero_spawn():
    script = (
        ROOT / "deploy" / "run-lambda-production-acceptance.sh"
    ).read_text(encoding="utf-8")

    assert "local_inference_service_from_env(require_enabled=True)" in script
    assert 'health["formula"]["spawn_count"] != 0' in script
    assert 'health["table"]["spawn_count"] != 0' in script
    assert "service.close()" in script



def test_lambda_production_shell_scripts_parse():
    for name in (
        "configure-production-local-inference.sh",
        "run-lambda-production-acceptance.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "deploy" / name)],
            check=True,
            text=True,
            capture_output=True,
        )
