from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_container_deployment_contract():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "deploy" / "docker-compose.daemon.yml").read_text()
    env = (ROOT / "deploy" / "research-engine.env.example").read_text()

    assert 'ENTRYPOINT ["research-engine"]' in dockerfile
    assert "USER research-engine" in dockerfile
    assert "restart: unless-stopped" in compose
    assert "condition: service_healthy" in compose
    assert "research_engine_artifacts:/var/lib/research-engine/artifacts" in compose
    assert "RESEARCH_ENGINE_CHEAP_WORKERS" in compose
    assert "RESEARCH_ENGINE_STRONG_WORKERS" in compose
    assert "OPENAI_API_KEY=" in env
    assert "sk-" not in env
    assert "/tmp" not in compose


def test_systemd_contract_uses_persistent_workspace_and_restart():
    service = (ROOT / "deploy" / "systemd" / "research-engine-daemon.service").read_text()
    assert "Restart=always" in service
    assert "%h/adamArchives/Adam/research-engine" in service
    assert "%h/adamArchives/Adam/.research-engine/research-engine.env" in service
    assert "/tmp" not in service
