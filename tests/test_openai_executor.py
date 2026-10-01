import json
from types import SimpleNamespace

import pytest

from research_engine.openai_executor import (
    OpenAIResponsesExecutor,
    profiles_from_env,
    result_schema,
)
from research_engine.orchestration import (
    ExecutorTier,
    JobExecutionError,
    JobType,
    ResearchJob,
)


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def job(job_type, tier, payload=None):
    return ResearchJob(
        id="job:test",
        run_id="run:test",
        job_type=job_type,
        executor_tier=tier,
        status="RUNNING",
        priority=0,
        idempotency_key="test",
        payload=payload or {},
        result=None,
        depends_on=[],
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker:test",
        lease_expires_at=None,
    )


def response(output, output_items=None):
    return SimpleNamespace(
        id="resp_123",
        model="model-used",
        status="completed",
        output_text=json.dumps(output),
        output=output_items or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_default_profiles_route_cost_and_reasoning_tiers(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("RESEARCH_ENGINE_OPENAI_"):
            monkeypatch.delenv(name, raising=False)
    profiles = profiles_from_env()
    assert profiles[ExecutorTier.CHEAP].model == "gpt-5.6-luna"
    assert profiles[ExecutorTier.CHEAP].reasoning_effort == "low"
    assert profiles[ExecutorTier.STANDARD].model == "gpt-5.6-terra"
    assert profiles[ExecutorTier.STANDARD].reasoning_effort == "medium"
    assert profiles[ExecutorTier.STRONG].model == "gpt-5.6-sol"
    assert profiles[ExecutorTier.STRONG].reasoning_effort == "high"


def test_discovery_uses_structured_output_and_records_execution_metadata():
    fake = FakeClient(
        response(
            {
                "candidates": [
                    {
                        "id": "paper:1",
                        "title": "Paper",
                        "source_uri": "https://example.test/paper",
                        "rationale": "relevant",
                    }
                ]
            }
        )
    )
    executor = OpenAIResponsesExecutor(tier=ExecutorTier.CHEAP, client=fake)
    result = executor.execute(job(JobType.DISCOVERY, ExecutorTier.CHEAP, {"topic": "RF"}))
    assert result["candidates"][0]["id"] == "paper:1"
    assert result["_execution"]["provider"] == "openai"
    assert result["_execution"]["response_id"] == "resp_123"
    call = fake.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["reasoning"] == {"effort": "low"}
    assert call["store"] is False
    assert call["tools"] == [{"type": "web_search"}]
    assert call["tool_choice"] == "required"
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True


def test_novelty_arbitration_schema_cannot_emit_supported_as_open():
    values = result_schema(JobType.NOVELTY_ARBITRATE)["properties"]["status"]["enum"]
    assert "SUPPORTED_AS_OPEN" not in values
    assert set(values) == {
        "REJECTED_PRIOR_WORK_FOUND",
        "NARROWED",
        "NEEDS_HUMAN_REVIEW",
    }


def test_refusal_is_non_retryable():
    refusal = SimpleNamespace(
        id="resp_refusal",
        model="gpt-5.6-luna",
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(type="refusal", refusal="cannot perform job")]
            )
        ],
        usage=None,
    )
    executor = OpenAIResponsesExecutor(
        tier=ExecutorTier.CHEAP,
        client=FakeClient(refusal),
    )
    with pytest.raises(JobExecutionError) as exc:
        executor.execute(job(JobType.DISCOVERY, ExecutorTier.CHEAP))
    assert exc.value.code == "MODEL_REFUSAL"
    assert exc.value.retryable is False


def test_incomplete_response_is_retryable():
    incomplete = SimpleNamespace(
        id="resp_incomplete",
        model="gpt-5.6-terra",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        output_text="",
        output=[],
        usage=None,
    )
    executor = OpenAIResponsesExecutor(
        tier=ExecutorTier.STANDARD,
        client=FakeClient(incomplete),
    )
    with pytest.raises(JobExecutionError) as exc:
        executor.execute(job(JobType.EXTRACT, ExecutorTier.STANDARD))
    assert exc.value.code == "MODEL_INCOMPLETE"
    assert exc.value.retryable is True


def test_model_executor_rejects_deterministic_and_human_tiers():
    fake = FakeClient(response({}))
    with pytest.raises(ValueError):
        OpenAIResponsesExecutor(tier=ExecutorTier.DETERMINISTIC, client=fake)
    with pytest.raises(ValueError):
        OpenAIResponsesExecutor(tier=ExecutorTier.HUMAN, client=fake)


def test_search_jobs_capture_web_call_and_url_source_provenance():
    search_call = SimpleNamespace(
        type="web_search_call",
        id="ws_123",
        action={"type": "search", "queries": ["open set RF fingerprinting"]},
    )
    citation = SimpleNamespace(
        type="url_citation",
        url="https://arxiv.org/abs/2605.15593",
        title="Dynamic and Open-Set RF Fingerprinting",
        start_index=1,
        end_index=20,
    )
    message = SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(
                type="output_text",
                annotations=[citation],
            )
        ],
    )
    fake = FakeClient(
        response(
            {
                "candidates": [
                    {
                        "id": "paper:1",
                        "title": "Paper",
                        "source_uri": "https://arxiv.org/abs/2605.15593",
                        "rationale": "counterevidence",
                    }
                ]
            },
            [search_call, message],
        )
    )
    result = OpenAIResponsesExecutor(
        tier=ExecutorTier.CHEAP,
        client=fake,
    ).execute(job(JobType.DISCOVERY, ExecutorTier.CHEAP, {"topic": "RF"}))

    provenance = result["_execution"]["web_search"]
    assert provenance["mode"] == "live"
    assert provenance["calls"][0]["id"] == "ws_123"
    assert provenance["calls"][0]["action"]["queries"] == [
        "open set RF fingerprinting"
    ]
    assert provenance["sources"] == [
        {
            "url": "https://arxiv.org/abs/2605.15593",
            "title": "Dynamic and Open-Set RF Fingerprinting",
            "start_index": 1,
            "end_index": 20,
        }
    ]


def test_non_search_semantic_jobs_do_not_receive_web_tool():
    fake = FakeClient(
        response(
            {
                "retained": [],
            }
        )
    )
    OpenAIResponsesExecutor(
        tier=ExecutorTier.CHEAP,
        client=fake,
    ).execute(job(JobType.SCREEN, ExecutorTier.CHEAP, {"candidates": []}))
    call = fake.responses.calls[0]
    assert "tools" not in call
    assert "tool_choice" not in call
