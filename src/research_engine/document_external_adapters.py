from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REMOTE_SECRET_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "MATHPIX_APP_ID",
        "MATHPIX_APP_KEY",
        "MINERU_API_KEY",
    }
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_registry_path() -> Path:
    return (
        repository_root()
        / "benchmarks"
        / "document_intelligence"
        / "candidate-runtime-contracts.v1.json"
    )


@dataclass(frozen=True)
class RuntimeProbe:
    adapter_id: str
    available: bool
    executable: str | None
    actual_version: str | None
    expected_version: str
    version_matches_contract: bool | None
    reason: str | None = None


@dataclass(frozen=True)
class ExternalRunReceipt:
    adapter_id: str
    mode: str
    command: tuple[str, ...]
    output_dir: str
    markdown_path: str
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    actual_version: str | None


def load_candidate_registry(
    path: str | Path | None = None,
) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else default_registry_path()
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported candidate runtime registry schema")
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate runtime registry must contain candidates")
    return data


def candidate_spec(
    adapter_id: str,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    registry = load_candidate_registry(registry_path)
    for candidate in registry["candidates"]:
        if candidate.get("adapter_id") == adapter_id:
            return dict(candidate)
    raise KeyError(f"unknown document adapter: {adapter_id}")


def _candidate_executable(spec: Mapping[str, Any]) -> str | None:
    override_name = spec.get("executable_env")
    if override_name:
        override = os.getenv(str(override_name))
        if override:
            path = Path(override).expanduser()
            return str(path) if path.is_file() else None
    executable = str(spec["executable"])
    return shutil.which(executable)


def _python_for_executable(executable: str) -> str | None:
    path = Path(executable)
    sibling = path.parent / "python"
    if sibling.is_file():
        return str(sibling)
    sibling3 = path.parent / "python3"
    if sibling3.is_file():
        return str(sibling3)
    return shutil.which("python3") or shutil.which("python")


def _distribution_version(
    executable: str,
    distribution: str,
) -> str | None:
    python = (
        executable
        if Path(executable).name.startswith("python")
        else _python_for_executable(executable)
    )
    if python is None:
        return None
    code = (
        "import importlib.metadata as m; "
        f"print(m.version({distribution!r}))"
    )
    proc = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _version_matches(actual: str | None, pattern: str) -> bool | None:
    if actual is None:
        return None
    return re.fullmatch(pattern, actual) is not None


def probe_candidate(
    adapter_id: str,
    *,
    registry_path: str | Path | None = None,
) -> RuntimeProbe:
    spec = candidate_spec(adapter_id, registry_path=registry_path)
    executable = _candidate_executable(spec)
    expected = str(spec["current_version"])
    if executable is None:
        return RuntimeProbe(
            adapter_id=adapter_id,
            available=False,
            executable=None,
            actual_version=None,
            expected_version=expected,
            version_matches_contract=None,
            reason=(
                f"{spec['executable']} unavailable; set "
                f"{spec.get('executable_env')} for an isolated runtime"
            ),
        )

    actual = _distribution_version(
        executable,
        str(spec["distribution"]),
    )
    matches = _version_matches(actual, str(spec["version_pattern"]))
    reason = None
    available = actual is not None
    if actual is None:
        reason = (
            "runtime executable exists but the pinned package distribution "
            "could not be verified in that runtime"
        )
    elif matches is False:
        reason = (
            f"installed version {actual!r} does not match pinned "
            f"contract {spec['version_pattern']!r}"
        )
    return RuntimeProbe(
        adapter_id=adapter_id,
        available=available,
        executable=executable,
        actual_version=actual,
        expected_version=expected,
        version_matches_contract=matches,
        reason=reason,
    )


def _format_command(
    template: list[str],
    *,
    executable: str,
    source: Path,
    output_dir: Path,
) -> list[str]:
    python = _python_for_executable(executable)
    values = {
        "bin": executable,
        "python": python or executable,
        "source": str(source),
        "source_stem": source.stem,
        "output_dir": str(output_dir),
        "output_file": str(output_dir / "document.md"),
        "shim": str(
            repository_root()
            / "benchmarks"
            / "document_intelligence"
            / "runtimes"
            / "paddle_ppstructurev3.py"
        ),
        "docling_native_shim": str(
            repository_root()
            / "benchmarks"
            / "document_intelligence"
            / "runtimes"
            / "docling_native.py"
        ),
    }
    return [str(part).format(**values) for part in template]


def build_candidate_command(
    adapter_id: str,
    mode: str,
    *,
    source_pdf: str | Path,
    output_dir: str | Path,
    registry_path: str | Path | None = None,
) -> tuple[str, ...]:
    spec = candidate_spec(adapter_id, registry_path=registry_path)
    modes = spec.get("modes", {})
    if mode not in modes:
        raise KeyError(f"adapter {adapter_id!r} has no mode {mode!r}")
    probe = probe_candidate(adapter_id, registry_path=registry_path)
    if not probe.available or probe.executable is None:
        raise RuntimeError(probe.reason or f"{adapter_id} runtime unavailable")
    if probe.version_matches_contract is not True:
        raise RuntimeError(
            probe.reason
            or f"{adapter_id} runtime version is not pinned/verified"
        )
    source = Path(source_pdf).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir).resolve()
    return tuple(
        _format_command(
            list(modes[mode]["command"]),
            executable=probe.executable,
            source=source,
            output_dir=destination,
        )
    )


def local_only_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    for name in REMOTE_SECRET_NAMES:
        env.pop(name, None)
    env["RESEARCH_ENGINE_DOCUMENT_BENCHMARK_LOCAL_ONLY"] = "1"
    return env


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def discover_markdown(
    output_dir: str | Path,
    *,
    source_stem: str,
) -> Path:
    root = Path(output_dir)
    preferred = [
        root / "document.md",
        root / f"{source_stem}.md",
    ]
    for path in preferred:
        if path.is_file():
            return path
    matches = sorted(root.rglob("*.md"))
    exact = [path for path in matches if path.stem == source_stem]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"candidate runtime produced no Markdown under {root}"
        )
    raise RuntimeError(
        f"ambiguous Markdown outputs for {source_stem}: "
        + ", ".join(str(path.relative_to(root)) for path in matches[:20])
    )


def run_candidate(
    adapter_id: str,
    mode: str,
    *,
    source_pdf: str | Path,
    output_dir: str | Path,
    registry_path: str | Path | None = None,
    timeout_seconds: int = 1800,
) -> ExternalRunReceipt:
    source = Path(source_pdf).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = build_candidate_command(
        adapter_id,
        mode,
        source_pdf=source,
        output_dir=destination,
        registry_path=registry_path,
    )
    probe = probe_candidate(adapter_id, registry_path=registry_path)

    started = time.perf_counter()
    proc = subprocess.run(
        list(command),
        cwd=str(repository_root()),
        env=local_only_environment(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{adapter_id}/{mode} failed rc={proc.returncode}; "
            f"stderr={_tail(proc.stderr)!r}"
        )
    markdown = discover_markdown(
        destination,
        source_stem=source.stem,
    )
    return ExternalRunReceipt(
        adapter_id=adapter_id,
        mode=mode,
        command=command,
        output_dir=str(destination),
        markdown_path=str(markdown),
        elapsed_seconds=elapsed,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
        actual_version=probe.actual_version,
    )


def candidate_status(
    *,
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    registry = load_candidate_registry(registry_path)
    rows = []
    for spec in registry["candidates"]:
        probe = probe_candidate(
            str(spec["adapter_id"]),
            registry_path=registry_path,
        )
        rows.append(
            {
                "adapter_id": probe.adapter_id,
                "available": probe.available,
                "executable": probe.executable,
                "actual_version": probe.actual_version,
                "expected_version": probe.expected_version,
                "version_matches_contract": probe.version_matches_contract,
                "reason": probe.reason,
                "modes": sorted(spec.get("modes", {}).keys()),
                "license": spec.get("license"),
                "capabilities": spec.get("capabilities", []),
            }
        )
    return rows
