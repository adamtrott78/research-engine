from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .orchestration import (
    ExecutorTier,
    JobExecutionError,
    JobType,
    ResearchJob,
)


SEMANTIC_JOB_TYPES = {
    JobType.DISCOVERY,
    JobType.SCREEN,
    JobType.EXTRACT,
    JobType.VERIFY,
    JobType.SYNTHESIZE,
    JobType.GAP_GENERATE,
    JobType.NOVELTY_SEARCH,
    JobType.NOVELTY_ARBITRATE,
}


@dataclass(frozen=True)
class OpenAIModelProfile:
    model: str
    reasoning_effort: str
    max_output_tokens: int


def profiles_from_env() -> dict[ExecutorTier, OpenAIModelProfile]:
    return {
        ExecutorTier.CHEAP: OpenAIModelProfile(
            model=os.getenv("RESEARCH_ENGINE_OPENAI_CHEAP_MODEL", "gpt-5.6-luna"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_OPENAI_CHEAP_EFFORT", "low"),
            max_output_tokens=int(os.getenv("RESEARCH_ENGINE_OPENAI_CHEAP_MAX_OUTPUT", "6000")),
        ),
        ExecutorTier.STANDARD: OpenAIModelProfile(
            model=os.getenv("RESEARCH_ENGINE_OPENAI_STANDARD_MODEL", "gpt-5.6-terra"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_OPENAI_STANDARD_EFFORT", "medium"),
            max_output_tokens=int(os.getenv("RESEARCH_ENGINE_OPENAI_STANDARD_MAX_OUTPUT", "12000")),
        ),
        ExecutorTier.STRONG: OpenAIModelProfile(
            model=os.getenv("RESEARCH_ENGINE_OPENAI_STRONG_MODEL", "gpt-5.6-sol"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_OPENAI_STRONG_EFFORT", "high"),
            max_output_tokens=int(os.getenv("RESEARCH_ENGINE_OPENAI_STRONG_MAX_OUTPUT", "20000")),
        ),
    }


def _candidate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "source_uri": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["id", "title", "source_uri", "rationale"],
    }


def result_schema(job_type: JobType) -> dict[str, Any]:
    candidate = _candidate_schema()
    schemas: dict[JobType, dict[str, Any]] = {
        JobType.DISCOVERY: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidates": {"type": "array", "items": candidate},
            },
            "required": ["candidates"],
        },
        JobType.SCREEN: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "retained": {"type": "array", "items": candidate},
            },
            "required": ["retained"],
        },
        JobType.EXTRACT: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "normalized_form": {"type": "string"},
                            "source_locator": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "id",
                            "type",
                            "normalized_form",
                            "source_locator",
                            "confidence",
                        ],
                    },
                },
            },
            "required": ["evidence"],
        },
        JobType.VERIFY: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verified": {"type": "boolean"},
                "item_id": {"type": "string"},
                "accepted_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["verified", "item_id", "accepted_evidence_ids", "issues"],
        },
        JobType.SYNTHESIZE: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "themes": {"type": "array", "items": {"type": "string"}},
                "statements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "qualifiers": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text", "evidence_ids", "qualifiers"],
                    },
                },
            },
            "required": ["themes", "statements"],
        },
        JobType.GAP_GENERATE: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "statement": {"type": "string"},
                            "gap_type": {"type": "string"},
                            "rationale": {"type": "string"},
                            "search_queries": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "id",
                            "statement",
                            "gap_type",
                            "rationale",
                            "search_queries",
                        ],
                    },
                },
            },
            "required": ["gaps"],
        },
        JobType.NOVELTY_SEARCH: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string"},
                "counterevidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "identifier": {"type": "string"},
                            "source_uri": {"type": "string"},
                            "relevance": {"type": "string"},
                        },
                        "required": ["title", "identifier", "source_uri", "relevance"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["query", "counterevidence", "notes"],
        },
        JobType.NOVELTY_ARBITRATE: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "REJECTED_PRIOR_WORK_FOUND",
                        "NARROWED",
                        "NEEDS_HUMAN_REVIEW",
                    ],
                },
                "reasoning": {"type": "string"},
                "surviving_statement": {"type": "string"},
                "search_count": {"type": "integer"},
            },
            "required": ["status", "reasoning", "surviving_statement", "search_count"],
        },
    }
    if job_type not in schemas:
        raise ValueError(f"no semantic result schema for {job_type}")
    return schemas[job_type]


COMMON_INSTRUCTIONS = """You are one bounded worker inside a provenance-first automated research engine.
Operate only on the supplied job payload and the tools/context explicitly available to this worker.
Do not invent missing source facts, identifiers, measurements, citations, or provenance.
Preserve uncertainty and return empty arrays rather than fabricating evidence.
Your output is machine-consumed structured data, not prose for a user.
Scientific novelty is challenged, never asserted automatically.
"""


JOB_INSTRUCTIONS: dict[JobType, str] = {
    JobType.DISCOVERY: "Identify candidate scholarly works relevant to the supplied research topic or search task.",
    JobType.SCREEN: "Screen supplied candidates for relevance. Retain only candidates supported by the supplied metadata/context.",
    JobType.EXTRACT: "Extract source-grounded evidence objects. Each evidence item must retain a source locator string supplied or derivable from the payload.",
    JobType.VERIFY: "Verify the supplied extraction against the supplied source context. Report issues explicitly.",
    JobType.SYNTHESIZE: "Synthesize only across supplied verified evidence/checkpoint context. Every statement must name its evidence IDs and qualifiers.",
    JobType.GAP_GENERATE: "Generate provisional gap candidates from supplied synthesis. Each gap must include adversarial search queries designed to disprove it.",
    JobType.NOVELTY_SEARCH: "Search or assess the supplied query for counterevidence to the candidate gap. Prefer evidence that would reject or narrow the gap.",
    JobType.NOVELTY_ARBITRATE: "Adjudicate the supplied novelty-search results conservatively. You may reject, narrow, or escalate to human review. You may not mark a gap supported as open.",
}


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {
            key: _as_jsonable(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _find_refusal(response: Any) -> str | None:
    for item in _attr(response, "output", []) or []:
        for content in _attr(item, "content", []) or []:
            if _attr(content, "type") == "refusal":
                return str(_attr(content, "refusal", "model refused the job"))
    return None


class OpenAIResponsesExecutor:
    """Semantic JobExecutor backed by OpenAI Responses structured outputs."""

    def __init__(
        self,
        *,
        tier: ExecutorTier,
        client: Any | None = None,
        profiles: dict[ExecutorTier, OpenAIModelProfile] | None = None,
    ):
        if tier not in {ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG}:
            raise ValueError(f"OpenAI semantic executor cannot serve tier {tier}")
        self.tier = tier
        self.profiles = profiles or profiles_from_env()
        self.profile = self.profiles[tier]
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI executor requires optional dependency: pip install 'research-engine[openai]'"
                ) from exc
            client = OpenAI()
        self.client = client

    def execute(self, job: ResearchJob) -> dict[str, Any]:
        if job.executor_tier != self.tier:
            raise JobExecutionError(
                f"job tier {job.executor_tier} does not match executor tier {self.tier}",
                code="TIER_MISMATCH",
                retryable=False,
            )
        if job.job_type not in SEMANTIC_JOB_TYPES:
            raise JobExecutionError(
                f"job type {job.job_type} must not be executed by a semantic model",
                code="UNSUPPORTED_JOB_TYPE",
                retryable=False,
            )

        schema = result_schema(job.job_type)
        payload = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": str(job.job_type),
            "payload": job.payload,
        }

        try:
            response = self.client.responses.create(
                model=self.profile.model,
                store=False,
                reasoning={"effort": self.profile.reasoning_effort},
                instructions=COMMON_INSTRUCTIONS + "\n" + JOB_INSTRUCTIONS[job.job_type],
                input=json.dumps(payload, sort_keys=True),
                max_output_tokens=self.profile.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": f"research_engine_{str(job.job_type).lower()}_result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except Exception as exc:
            raise JobExecutionError(
                f"OpenAI request failed: {exc}",
                code="OPENAI_REQUEST_FAILED",
                retryable=True,
                detail={"model": self.profile.model},
            ) from exc

        refusal = _find_refusal(response)
        if refusal:
            raise JobExecutionError(
                refusal,
                code="MODEL_REFUSAL",
                retryable=False,
                detail={"model": _attr(response, "model", self.profile.model)},
            )

        status = str(_attr(response, "status", "completed"))
        if status != "completed":
            raise JobExecutionError(
                f"model response status was {status}",
                code="MODEL_INCOMPLETE",
                retryable=True,
                detail={
                    "model": _attr(response, "model", self.profile.model),
                    "response_id": _attr(response, "id"),
                    "incomplete_details": _as_jsonable(
                        _attr(response, "incomplete_details")
                    ),
                },
            )

        output_text = _attr(response, "output_text", "")
        if not output_text:
            raise JobExecutionError(
                "model returned no structured output text",
                code="EMPTY_MODEL_OUTPUT",
                retryable=True,
                detail={"response_id": _attr(response, "id")},
            )
        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise JobExecutionError(
                "structured model output was not valid JSON",
                code="INVALID_MODEL_JSON",
                retryable=True,
                detail={"response_id": _attr(response, "id")},
            ) from exc

        if not isinstance(result, dict):
            raise JobExecutionError(
                "structured model output root was not an object",
                code="INVALID_MODEL_RESULT",
                retryable=True,
            )

        result["_execution"] = {
            "provider": "openai",
            "model": _attr(response, "model", self.profile.model),
            "response_id": _attr(response, "id"),
            "executor_tier": str(self.tier),
            "reasoning_effort": self.profile.reasoning_effort,
            "usage": _as_jsonable(_attr(response, "usage")),
        }
        return result
