from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openai_executor import (
    COMMON_INSTRUCTIONS,
    JOB_INSTRUCTIONS,
    SEARCH_JOB_TYPES,
    SEMANTIC_JOB_TYPES,
    result_schema,
)
from .orchestration import ExecutorTier, JobExecutionError, ResearchJob


@dataclass(frozen=True)
class CodexModelProfile:
    model: str
    reasoning_effort: str


def profiles_from_env() -> dict[ExecutorTier, CodexModelProfile]:
    return {
        ExecutorTier.CHEAP: CodexModelProfile(
            model=os.getenv("RESEARCH_ENGINE_CODEX_CHEAP_MODEL", "gpt-5.6-luna"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_CODEX_CHEAP_EFFORT", "low"),
        ),
        ExecutorTier.STANDARD: CodexModelProfile(
            model=os.getenv("RESEARCH_ENGINE_CODEX_STANDARD_MODEL", "gpt-5.6-terra"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_CODEX_STANDARD_EFFORT", "medium"),
        ),
        ExecutorTier.STRONG: CodexModelProfile(
            model=os.getenv("RESEARCH_ENGINE_CODEX_STRONG_MODEL", "gpt-5.6-sol"),
            reasoning_effort=os.getenv("RESEARCH_ENGINE_CODEX_STRONG_EFFORT", "high"),
        ),
    }


class CodexExecExecutor:
    """Semantic executor backed by ChatGPT-authenticated Codex CLI plan usage."""

    def __init__(
        self,
        *,
        tier: ExecutorTier,
        binary: str | None = None,
        profiles: dict[ExecutorTier, CodexModelProfile] | None = None,
        timeout_seconds: int | None = None,
    ):
        if tier not in {ExecutorTier.CHEAP, ExecutorTier.STANDARD, ExecutorTier.STRONG}:
            raise ValueError(f"Codex semantic executor cannot serve tier {tier}")
        self.tier = tier
        self.profile = (profiles or profiles_from_env())[tier]
        self.binary = binary or os.getenv("RESEARCH_ENGINE_CODEX_BIN", "codex")
        self.timeout_seconds = timeout_seconds or int(
            os.getenv("RESEARCH_ENGINE_CODEX_TIMEOUT_SECONDS", "900")
        )

    @staticmethod
    def _prompt(job: ResearchJob) -> str:
        payload = {
            "job_id": job.id,
            "run_id": job.run_id,
            "job_type": str(job.job_type),
            "payload": job.payload,
        }
        return (
            COMMON_INSTRUCTIONS
            + "\n"
            + JOB_INSTRUCTIONS[job.job_type]
            + "\nReturn only the JSON object required by the supplied output schema. "
              "Do not modify local files or run shell commands. "
              "For discovery/novelty search, use Codex native web search and preserve source URLs."
            + "\n\nJOB:\n"
            + json.dumps(payload, sort_keys=True)
        )

    @staticmethod
    def _parse_events(stdout: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        final_text: str | None = None
        usage: dict[str, Any] = {}
        search_events: list[dict[str, Any]] = []

        for raw in stdout.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise JobExecutionError(
                    "Codex exec emitted invalid JSONL",
                    code="CODEX_INVALID_JSONL",
                    retryable=True,
                    detail={"line": raw[:500]},
                ) from exc

            etype = event.get("type")
            if etype == "turn.failed":
                raise JobExecutionError(
                    str((event.get("error") or {}).get("message") or "Codex turn failed"),
                    code="CODEX_TURN_FAILED",
                    retryable=True,
                )
            if etype == "error":
                raise JobExecutionError(
                    str(event.get("message") or "Codex exec error"),
                    code="CODEX_STREAM_ERROR",
                    retryable=True,
                )
            if etype == "turn.completed":
                usage = dict(event.get("usage") or {})
                continue
            if etype != "item.completed":
                continue
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                final_text = item.get("text")
            elif item.get("type") == "web_search":
                search_events.append(
                    {
                        "query": item.get("query"),
                        "action": item.get("action"),
                    }
                )

        if not final_text:
            raise JobExecutionError(
                "Codex exec produced no final agent message",
                code="CODEX_EMPTY_OUTPUT",
                retryable=True,
            )
        try:
            result = json.loads(final_text)
        except json.JSONDecodeError as exc:
            raise JobExecutionError(
                "Codex structured result was not valid JSON",
                code="CODEX_INVALID_RESULT",
                retryable=True,
                detail={"output": final_text[:1000]},
            ) from exc
        if not isinstance(result, dict):
            raise JobExecutionError(
                "Codex structured result root was not an object",
                code="CODEX_INVALID_RESULT",
                retryable=True,
            )
        return result, usage, search_events

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

        with tempfile.TemporaryDirectory(prefix="research-engine-codex-") as scratch:
            schema_path = Path(scratch) / "schema.json"
            schema_path.write_text(json.dumps(result_schema(job.job_type)))

            cmd = [
                self.binary,
                "--model",
                self.profile.model,
                "--ask-for-approval",
                "never",
                "--sandbox",
                "read-only",
                "-c",
                f'model_reasoning_effort="{self.profile.reasoning_effort}"',
            ]
            if job.job_type in SEARCH_JOB_TYPES:
                cmd.append("--search")
            cmd.extend(
                [
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--json",
                    "--output-schema",
                    str(schema_path),
                    self._prompt(job),
                ]
            )

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=os.getenv("RESEARCH_ENGINE_CODEX_WORKDIR") or None,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise JobExecutionError(
                    "Codex CLI is not installed or not on PATH",
                    code="CODEX_NOT_INSTALLED",
                    retryable=False,
                    detail={"binary": self.binary},
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise JobExecutionError(
                    "Codex exec timed out",
                    code="CODEX_TIMEOUT",
                    retryable=True,
                    detail={"timeout_seconds": self.timeout_seconds},
                ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            lowered = stderr.lower()
            if "usage limit" in lowered or "rate limit" in lowered or "limit reached" in lowered:
                code = "CODEX_PLAN_LIMIT"
            elif "login" in lowered or "authentication" in lowered or "unauthorized" in lowered:
                code = "CODEX_AUTH_REQUIRED"
            else:
                code = "CODEX_EXEC_FAILED"
            raise JobExecutionError(
                stderr[-2000:] or f"Codex exec exited {proc.returncode}",
                code=code,
                retryable=code not in {"CODEX_AUTH_REQUIRED", "CODEX_NOT_INSTALLED"},
                detail={"returncode": proc.returncode},
            )

        result, usage, search_events = self._parse_events(proc.stdout)
        execution: dict[str, Any] = {
            "provider": "codex-chatgpt-plan",
            "model": self.profile.model,
            "model_identity_source": "requested_config",
            "executor_tier": str(self.tier),
            "reasoning_effort": self.profile.reasoning_effort,
            "usage": usage,
        }
        if job.job_type in SEARCH_JOB_TYPES:
            execution["web_search"] = {
                "mode": "codex-live",
                "events": search_events,
            }
        result["_execution"] = execution
        return result
