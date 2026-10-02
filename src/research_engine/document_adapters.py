from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ADAPTER_IDS = (
    "docling",
    "mineru",
    "paddle-pp-structure-v3",
    "marker",
)


@dataclass(frozen=True)
class AdapterAvailability:
    adapter_id: str
    available: bool
    executable: str | None
    reason: str | None = None


@dataclass(frozen=True)
class AdapterPlan:
    adapter_id: str
    executable: str
    command: tuple[str, ...]
    output_directory: str
    structured_provenance_expected: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterRunReceipt:
    adapter_id: str
    command: tuple[str, ...]
    elapsed_seconds: float
    returncode: int
    stdout_tail: str
    stderr_tail: str
    output_files: tuple[dict[str, Any], ...]
    structured_provenance_expected: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_executable(env_name: str, default: str) -> tuple[str | None, str | None]:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path), None
        found = shutil.which(configured)
        if found:
            return found, None
        return None, f"{env_name}={configured!r} is not executable"
    found = shutil.which(default)
    if found:
        return found, None
    return None, f"{default!r} not found; set {env_name} to the isolated executable"


def _clean_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"adapter output directory must be empty to avoid stale results: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _tail(value: str, limit: int = 12_000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _inventory_output(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        rows.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    return tuple(rows)


def _markdown_outputs(path: Path) -> list[Path]:
    return sorted(
        [
            *path.rglob("*.md"),
            *path.rglob("*.mmd"),
            *path.rglob("*.markdown"),
        ]
    )


class ExternalDocumentAdapter:
    adapter_id: str
    env_var: str
    default_executable: str
    structured_provenance_expected: bool = False
    notes: tuple[str, ...] = ()

    def availability(self) -> AdapterAvailability:
        executable, reason = _resolve_executable(
            self.env_var,
            self.default_executable,
        )
        return AdapterAvailability(
            adapter_id=self.adapter_id,
            available=executable is not None,
            executable=executable,
            reason=reason,
        )

    def build_command(
        self,
        *,
        executable: str,
        source_pdf: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def plan(
        self,
        *,
        source_pdf: str | Path,
        output_directory: str | Path,
    ) -> AdapterPlan:
        availability = self.availability()
        if not availability.available or not availability.executable:
            raise RuntimeError(
                availability.reason or f"{self.adapter_id} is unavailable"
            )
        source = Path(source_pdf)
        if not source.is_file():
            raise FileNotFoundError(source)
        output = Path(output_directory)
        command = self.build_command(
            executable=availability.executable,
            source_pdf=source,
            output_directory=output,
        )
        return AdapterPlan(
            adapter_id=self.adapter_id,
            executable=availability.executable,
            command=command,
            output_directory=str(output),
            structured_provenance_expected=self.structured_provenance_expected,
            notes=self.notes,
        )

    def run_raw(
        self,
        *,
        source_pdf: str | Path,
        output_directory: str | Path,
        timeout_seconds: float = 1800.0,
    ) -> AdapterRunReceipt:
        plan = self.plan(
            source_pdf=source_pdf,
            output_directory=output_directory,
        )
        output = Path(plan.output_directory)
        _clean_output_directory(output)

        started = time.perf_counter()
        completed = subprocess.run(
            list(plan.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        receipt = AdapterRunReceipt(
            adapter_id=plan.adapter_id,
            command=plan.command,
            elapsed_seconds=elapsed,
            returncode=completed.returncode,
            stdout_tail=_tail(completed.stdout or ""),
            stderr_tail=_tail(completed.stderr or ""),
            output_files=_inventory_output(output),
            structured_provenance_expected=plan.structured_provenance_expected,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.adapter_id} failed with return code "
                f"{completed.returncode}: {_tail(completed.stderr or '', 4000)}"
            )
        if not _markdown_outputs(output):
            raise RuntimeError(
                f"{self.adapter_id} completed without a Markdown output in {output}"
            )
        return receipt


class DoclingAdapter(ExternalDocumentAdapter):
    adapter_id = "docling"
    env_var = "RESEARCH_ENGINE_DOCLING_BIN"
    default_executable = "docling"
    structured_provenance_expected = True
    notes = (
        "Runs local Docling conversion only.",
        "Requests Markdown plus Docling JSON; paper-package conversion waits for "
        "parser-specific locator validation.",
    )

    def build_command(
        self,
        *,
        executable: str,
        source_pdf: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        return (
            executable,
            "convert",
            str(source_pdf),
            "--to",
            "md",
            "--to",
            "json",
            "--output",
            str(output_directory),
            "--abort-on-error",
        )


class MinerUAdapter(ExternalDocumentAdapter):
    adapter_id = "mineru"
    env_var = "RESEARCH_ENGINE_MINERU_BIN"
    default_executable = "mineru-kit"
    structured_provenance_expected = False
    notes = (
        "Uses the stateless MinerU 4.x mineru-kit parser rather than the "
        "document-library service.",
        "Default tier is standard; override later only in benchmark-specific config.",
    )

    def build_command(
        self,
        *,
        executable: str,
        source_pdf: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        return (
            executable,
            "parse",
            str(source_pdf),
            "-o",
            str(output_directory / "document.md"),
            "--tier",
            "standard",
        )


class MarkerAdapter(ExternalDocumentAdapter):
    adapter_id = "marker"
    env_var = "RESEARCH_ENGINE_MARKER_BIN"
    default_executable = "marker_single"
    structured_provenance_expected = False
    notes = (
        "Runs Marker locally with Markdown output and pagination enabled.",
        "Does not enable remote LLM enhancement.",
    )

    def build_command(
        self,
        *,
        executable: str,
        source_pdf: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        return (
            executable,
            str(source_pdf),
            "--output_format",
            "markdown",
            "--output_dir",
            str(output_directory),
            "--paginate_output",
        )


class PaddlePPStructureV3Adapter(ExternalDocumentAdapter):
    adapter_id = "paddle-pp-structure-v3"
    env_var = "RESEARCH_ENGINE_PADDLE_PYTHON"
    default_executable = "__paddle_python_must_be_configured__"
    structured_provenance_expected = True
    notes = (
        "Requires RESEARCH_ENGINE_PADDLE_PYTHON to point at an isolated Python "
        "environment containing PaddlePaddle + PaddleOCR.",
        "Uses the repository helper to preserve per-page JSON/Markdown/image output.",
    )

    def build_command(
        self,
        *,
        executable: str,
        source_pdf: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        helper = (
            Path(__file__).resolve().parent
            / "adapter_helpers"
            / "paddle_ppstructure_v3.py"
        )
        return (
            executable,
            str(helper),
            str(source_pdf),
            str(output_directory),
        )


def adapters() -> dict[str, ExternalDocumentAdapter]:
    rows: tuple[ExternalDocumentAdapter, ...] = (
        DoclingAdapter(),
        MinerUAdapter(),
        PaddlePPStructureV3Adapter(),
        MarkerAdapter(),
    )
    return {adapter.adapter_id: adapter for adapter in rows}


def adapter_status() -> list[dict[str, Any]]:
    return [
        {
            "adapter_id": status.adapter_id,
            "available": status.available,
            "executable": status.executable,
            "reason": status.reason,
        }
        for status in (adapter.availability() for adapter in adapters().values())
    ]


def render_plan(plan: AdapterPlan) -> dict[str, Any]:
    return {
        "adapter_id": plan.adapter_id,
        "executable": plan.executable,
        "command": list(plan.command),
        "command_shell": shlex.join(plan.command),
        "output_directory": plan.output_directory,
        "structured_provenance_expected": plan.structured_provenance_expected,
        "notes": list(plan.notes),
    }


def write_receipt(path: str | Path, receipt: AdapterRunReceipt) -> None:
    payload = {
        "adapter_id": receipt.adapter_id,
        "command": list(receipt.command),
        "elapsed_seconds": receipt.elapsed_seconds,
        "returncode": receipt.returncode,
        "stdout_tail": receipt.stdout_tail,
        "stderr_tail": receipt.stderr_tail,
        "output_files": list(receipt.output_files),
        "structured_provenance_expected": receipt.structured_provenance_expected,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
