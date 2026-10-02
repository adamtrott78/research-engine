import os
import stat
import sys
from pathlib import Path

import pytest

from research_engine.document_adapters import (
    DoclingAdapter,
    ExternalDocumentAdapter,
    MarkerAdapter,
    MinerUAdapter,
    PaddlePPStructureV3Adapter,
    adapter_status,
    adapters,
    render_plan,
)


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% benchmark placeholder\n")
    return path


def _make_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_registry_contains_all_candidate_adapters():
    assert set(adapters()) == {
        "docling",
        "mineru",
        "paddle-pp-structure-v3",
        "marker",
    }


@pytest.mark.parametrize(
    ("adapter_cls", "env_name", "default_name"),
    [
        (DoclingAdapter, "RESEARCH_ENGINE_DOCLING_BIN", "docling"),
        (MinerUAdapter, "RESEARCH_ENGINE_MINERU_BIN", "mineru-kit"),
        (MarkerAdapter, "RESEARCH_ENGINE_MARKER_BIN", "marker_single"),
    ],
)
def test_adapter_availability_fails_closed_when_executable_is_missing(
    monkeypatch,
    adapter_cls,
    env_name,
    default_name,
):
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    status = adapter_cls().availability()
    assert status.available is False
    assert status.executable is None
    assert default_name in status.reason


def test_paddle_requires_explicit_isolated_python(monkeypatch):
    monkeypatch.delenv("RESEARCH_ENGINE_PADDLE_PYTHON", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    status = PaddlePPStructureV3Adapter().availability()
    assert status.available is False
    assert "RESEARCH_ENGINE_PADDLE_PYTHON" in status.reason


def test_docling_plan_requests_markdown_and_structured_json(
    tmp_path: Path,
    monkeypatch,
):
    source = _pdf(tmp_path / "paper.pdf")
    fake = _make_executable(tmp_path / "docling", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("RESEARCH_ENGINE_DOCLING_BIN", str(fake))

    plan = DoclingAdapter().plan(
        source_pdf=source,
        output_directory=tmp_path / "out",
    )
    assert plan.command[:3] == (str(fake), "convert", str(source))
    assert plan.command.count("--to") == 2
    assert "md" in plan.command
    assert "json" in plan.command
    assert "--abort-on-error" in plan.command
    assert plan.structured_provenance_expected is True


def test_mineru_plan_uses_stateless_full_document_output(
    tmp_path: Path,
    monkeypatch,
):
    source = _pdf(tmp_path / "paper.pdf")
    fake = _make_executable(tmp_path / "mineru-kit", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("RESEARCH_ENGINE_MINERU_BIN", str(fake))

    plan = MinerUAdapter().plan(
        source_pdf=source,
        output_directory=tmp_path / "out",
    )
    assert plan.command[:3] == (str(fake), "parse", str(source))
    assert "--tier" in plan.command
    assert "standard" in plan.command
    assert str(tmp_path / "out" / "document.md") in plan.command


def test_marker_plan_never_enables_remote_llm(
    tmp_path: Path,
    monkeypatch,
):
    source = _pdf(tmp_path / "paper.pdf")
    fake = _make_executable(tmp_path / "marker_single", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("RESEARCH_ENGINE_MARKER_BIN", str(fake))

    plan = MarkerAdapter().plan(
        source_pdf=source,
        output_directory=tmp_path / "out",
    )
    assert plan.command[0] == str(fake)
    assert "--output_format" in plan.command
    assert "markdown" in plan.command
    assert "--paginate_output" in plan.command
    assert "--use_llm" not in plan.command


def test_paddle_plan_invokes_repo_helper_through_isolated_python(
    tmp_path: Path,
    monkeypatch,
):
    source = _pdf(tmp_path / "paper.pdf")
    monkeypatch.setenv("RESEARCH_ENGINE_PADDLE_PYTHON", sys.executable)

    plan = PaddlePPStructureV3Adapter().plan(
        source_pdf=source,
        output_directory=tmp_path / "out",
    )
    assert plan.command[0] == sys.executable
    assert plan.command[1].endswith(
        "research_engine/adapter_helpers/paddle_ppstructure_v3.py"
    )
    assert plan.command[2] == str(source)
    assert plan.structured_provenance_expected is True


class _FakeAdapter(ExternalDocumentAdapter):
    adapter_id = "fake"
    env_var = "RESEARCH_ENGINE_FAKE_BIN"
    default_executable = "fake-document-adapter"

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
            str(output_directory / "document.md"),
        )


def test_raw_adapter_run_records_hashed_outputs_and_rejects_stale_dir(
    tmp_path: Path,
    monkeypatch,
):
    source = _pdf(tmp_path / "paper.pdf")
    fake = _make_executable(
        tmp_path / "fake-document-adapter",
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_text('# converted\\n', encoding='utf-8')\n",
    )
    monkeypatch.setenv("RESEARCH_ENGINE_FAKE_BIN", str(fake))

    output = tmp_path / "out"
    receipt = _FakeAdapter().run_raw(
        source_pdf=source,
        output_directory=output,
        timeout_seconds=5,
    )
    assert receipt.returncode == 0
    assert receipt.output_files[0]["path"] == "document.md"
    assert len(receipt.output_files[0]["sha256"]) == 64

    with pytest.raises(ValueError, match="must be empty"):
        _FakeAdapter().run_raw(
            source_pdf=source,
            output_directory=output,
            timeout_seconds=5,
        )


def test_plan_rendering_is_json_safe(tmp_path: Path, monkeypatch):
    source = _pdf(tmp_path / "paper.pdf")
    fake = _make_executable(tmp_path / "docling", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("RESEARCH_ENGINE_DOCLING_BIN", str(fake))

    rendered = render_plan(
        DoclingAdapter().plan(
            source_pdf=source,
            output_directory=tmp_path / "out",
        )
    )
    assert rendered["adapter_id"] == "docling"
    assert isinstance(rendered["command"], list)
    assert isinstance(rendered["command_shell"], str)


def test_status_does_not_import_heavy_candidate_packages(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    rows = adapter_status()
    assert len(rows) == 4
    assert all("adapter_id" in row for row in rows)
