import json
from types import SimpleNamespace

import pytest

from research_engine.codex_executor import CodexExecExecutor, CodexModelProfile
from research_engine.orchestration import ExecutorTier, JobType, ResearchJob


def _job(job_type=JobType.DISCOVERY):
    return ResearchJob(
        id="job:test",
        run_id="run:test",
        job_type=job_type,
        executor_tier=ExecutorTier.CHEAP,
        status="RUNNING",
        priority=1,
        idempotency_key="test",
        payload={"topic": "open-set RF classification"},
        result=None,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker:test",
        lease_expires_at=None,
    )


def test_codex_exec_parses_structured_result_usage_and_search(monkeypatch):
    events = [
        {"type": "thread.started", "thread_id": "thread:1"},
        {
            "type": "item.completed",
            "item": {
                "type": "web_search",
                "query": "open set RF classification",
                "action": {"type": "search"},
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "paper:1",
                                "title": "Paper",
                                "source_uri": "https://example.org/paper",
                                "rationale": "relevant",
                            }
                        ]
                    }
                ),
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reasoning_output_tokens": 3,
            },
        },
    ]

    def fake_run(cmd, **kwargs):
        assert "--search" in cmd
        assert "--output-schema" in cmd
        assert "gpt-5.6-luna" in cmd
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(x) for x in events),
            stderr="",
        )

    monkeypatch.setattr("research_engine.codex_executor.subprocess.run", fake_run)
    executor = CodexExecExecutor(
        tier=ExecutorTier.CHEAP,
        profiles={
            ExecutorTier.CHEAP: CodexModelProfile("gpt-5.6-luna", "low")
        },
    )
    result = executor.execute(_job())
    assert result["candidates"][0]["source_uri"] == "https://example.org/paper"
    assert result["_execution"]["provider"] == "codex-chatgpt-plan"
    assert result["_execution"]["usage"]["input_tokens"] == 100
    assert result["_execution"]["web_search"]["events"][0]["query"]


def test_codex_exec_reports_plan_limit(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Usage limit reached. Try again later.",
        )

    monkeypatch.setattr("research_engine.codex_executor.subprocess.run", fake_run)
    executor = CodexExecExecutor(tier=ExecutorTier.CHEAP)
    with pytest.raises(Exception) as exc:
        executor.execute(_job(JobType.SCREEN))
    assert getattr(exc.value, "code", None) == "CODEX_PLAN_LIMIT"
