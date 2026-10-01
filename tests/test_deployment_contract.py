from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_container_deployment_is_explicit_api_development_only():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "deploy" / "docker-compose.daemon.yml").read_text()

    assert 'ENTRYPOINT ["research-engine"]' in dockerfile
    assert "USER research-engine" in dockerfile
    assert "openai-api-development" in compose
    assert "RESEARCH_ENGINE_SEMANTIC_BACKEND: openai_api" in compose
    assert "OPENAI_API_KEY" in compose
    assert "restart: unless-stopped" in compose
    assert "condition: service_healthy" in compose
    assert "research_engine_artifacts:/var/lib/research-engine/artifacts" in compose
    assert "/tmp" not in compose


def test_production_env_contract_is_codex_plan_only():
    env = (ROOT / "deploy" / "research-engine.env.example").read_text()

    assert "RESEARCH_ENGINE_SEMANTIC_BACKEND=codex" in env
    assert "RESEARCH_ENGINE_CODEX_BIN=" in env
    assert "RESEARCH_ENGINE_CODEX_CHEAP_MODEL=" in env
    assert "RESEARCH_ENGINE_CODEX_STRONG_MODEL=" in env
    assert "OPENAI_API_KEY=" not in env
    assert "sk-" not in env


def test_systemd_contract_uses_native_venv_codex_and_persistent_workspace():
    service = (ROOT / "deploy" / "systemd" / "research-engine-daemon.service").read_text()

    assert "Restart=always" in service
    assert "%h/adamArchives/Adam/research-engine" in service
    assert "%h/adamArchives/Adam/.research-engine/research-engine.env" in service
    assert "%h/adamArchives/Adam/.research-engine/venv/bin/research-engine daemon" in service
    assert "RESEARCH_ENGINE_SEMANTIC_BACKEND=codex" in service
    assert "RESEARCH_ENGINE_DATABASE_URL=postgresql://research_engine:research_engine@localhost:5432/research_engine" in service
    assert "RESEARCH_ENGINE_ARTIFACT_ROOT=%h/adamArchives/Adam/.research-engine/artifacts" in service
    assert "/tmp" not in service
