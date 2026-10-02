import ast
import json
import os
import sys
from pathlib import Path

import pytest

import research_engine.document_external_adapters as ext
from research_engine.document_external_adapters import (
    ExternalRunReceipt,
    RuntimeProbe,
    build_candidate_command,
    candidate_status,
    local_only_environment,
    run_candidate,
)


def _ready_probe(adapter_id: str, executable: str) -> RuntimeProbe:
    expected = {
        "docling": "2.131.0",
        "mineru": "4.0.x",
        "paddle-pp-structure-v3": "3.7.x",
        "marker": "2.0.0",
    }[adapter_id]
    actual = {
        "docling": "2.131.0",
        "mineru": "4.0.0",
        "paddle-pp-structure-v3": "3.7.1",
        "marker": "2.0.0",
    }[adapter_id]
    return RuntimeProbe(
        adapter_id=adapter_id,
        available=True,
        executable=executable,
        actual_version=actual,
        expected_version=expected,
        version_matches_contract=True,
    )


def test_candidate_registry_pins_versions_licenses_and_modes():
    registry = ext.load_candidate_registry()
    rows = {row["adapter_id"]: row for row in registry["candidates"]}

    assert set(rows) == {
        "docling",
        "mineru",
        "paddle-pp-structure-v3",
        "marker",
    }
    assert rows["docling"]["current_version"] == "2.131.0"
    assert rows["docling"]["license"]["code"] == "MIT"
    assert "convert-core" in rows["docling"]["install_profiles"]["native"]
    assert "format-pdf" in rows["docling"]["install_profiles"]["native"]
    assert rows["mineru"]["version_pattern"] == r"4\..+"
    assert "additional terms" in rows["mineru"]["license"]["code"]
    assert rows["paddle-pp-structure-v3"]["version_pattern"] == r"3\.7\..+"
    assert rows["paddle-pp-structure-v3"]["license"]["code"] == "Apache-2.0"
    assert rows["marker"]["current_version"] == "2.0.0"
    assert rows["marker"]["license"]["code"] == "Apache-2.0"
    assert "Open Rail" in rows["marker"]["license"]["weights"]

    assert set(rows["docling"]["modes"]) == {
        "native",
        "standard_no_ocr",
        "standard",
        "vlm",
    }
    assert set(rows["mineru"]["modes"]) == {
        "flash_native",
        "basic",
        "standard",
        "advanced",
    }
    assert set(rows["marker"]["modes"]) == {
        "fast_no_ocr",
        "fast",
        "balanced",
    }


def test_probe_fails_closed_when_runtime_missing(monkeypatch):
    monkeypatch.delenv("RESEARCH_ENGINE_DOCLING_BIN", raising=False)
    monkeypatch.setattr(ext.shutil, "which", lambda _: None)

    probe = ext.probe_candidate("docling")
    assert probe.available is False
    assert probe.version_matches_contract is None
    assert "unavailable" in (probe.reason or "")


def test_build_commands_never_select_remote_or_llm_modes(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fake")
    out = tmp_path / "out"

    def fake_probe(adapter_id, **_):
        return _ready_probe(adapter_id, sys.executable)

    monkeypatch.setattr(ext, "probe_candidate", fake_probe)

    docling_native = build_candidate_command(
        "docling",
        "native",
        source_pdf=source,
        output_dir=out,
    )
    assert Path(docling_native[0]).name.startswith("python")
    assert docling_native[1].endswith("document_intelligence/runtimes/docling_native.py")
    assert docling_native[2] == str(source.resolve())
    assert docling_native[3] == str(out.resolve())
    assert "--pipeline" not in docling_native

    docling_standard_no_ocr = build_candidate_command(
        "docling",
        "standard_no_ocr",
        source_pdf=source,
        output_dir=out,
    )
    assert Path(docling_standard_no_ocr[0]).name.startswith("python")
    assert any(
        part.endswith(
            "document_intelligence/runtimes/docling_standard_no_ocr.py"
        )
        for part in docling_standard_no_ocr
    )
    assert "--pipeline" not in docling_standard_no_ocr

    docling = build_candidate_command(
        "docling",
        "standard",
        source_pdf=source,
        output_dir=out,
    )
    assert "--pipeline" in docling
    assert "standard" in docling
    assert "--to" in docling

    mineru = build_candidate_command(
        "mineru",
        "standard",
        source_pdf=source,
        output_dir=out,
    )
    assert "--tier" in mineru
    assert "standard" in mineru
    assert "--remote" not in mineru

    marker = build_candidate_command(
        "marker",
        "fast",
        source_pdf=source,
        output_dir=out,
    )
    assert "--mode" in marker
    assert "fast" in marker
    assert "--use_llm" not in marker

    paddle = build_candidate_command(
        "paddle-pp-structure-v3",
        "pp_structure_v3",
        source_pdf=source,
        output_dir=out,
    )
    assert any(part.endswith("paddle_ppstructurev3.py") for part in paddle)


def test_version_drift_blocks_execution(tmp_path: Path, monkeypatch):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fake")

    monkeypatch.setattr(
        ext,
        "probe_candidate",
        lambda *_args, **_kwargs: RuntimeProbe(
            adapter_id="docling",
            available=True,
            executable=sys.executable,
            actual_version="2.130.0",
            expected_version="2.131.0",
            version_matches_contract=False,
            reason="version drift",
        ),
    )
    with pytest.raises(RuntimeError, match="version drift"):
        build_candidate_command(
            "docling",
            "standard",
            source_pdf=source,
            output_dir=tmp_path / "out",
        )


def test_local_only_environment_strips_remote_provider_secrets():
    base = {
        "PATH": "/bin",
        "OPENAI_API_KEY": "secret",
        "GOOGLE_API_KEY": "secret",
        "GEMINI_API_KEY": "secret",
        "OPENROUTER_API_KEY": "secret",
        "MATHPIX_APP_KEY": "secret",
        "SAFE_VALUE": "keep",
    }
    env = local_only_environment(base)
    for key in ext.REMOTE_SECRET_NAMES:
        assert key not in env
    assert env["SAFE_VALUE"] == "keep"
    assert env["RESEARCH_ENGINE_DOCUMENT_BENCHMARK_LOCAL_ONLY"] == "1"


def test_run_candidate_executes_without_shell_and_discovers_markdown(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fake")
    output = tmp_path / "out"
    script = tmp_path / "fake_adapter.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'document.md').write_text('# fake output\\n')\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ext,
        "build_candidate_command",
        lambda *_args, **_kwargs: (
            sys.executable,
            str(script),
            str(output),
        ),
    )
    monkeypatch.setattr(
        ext,
        "probe_candidate",
        lambda *_args, **_kwargs: _ready_probe(
            "docling",
            sys.executable,
        ),
    )

    receipt = run_candidate(
        "docling",
        "standard",
        source_pdf=source,
        output_dir=output,
        timeout_seconds=30,
    )
    assert isinstance(receipt, ExternalRunReceipt)
    assert receipt.command[0] == sys.executable
    assert Path(receipt.markdown_path).read_text() == "# fake output\n"
    assert receipt.actual_version == "2.131.0"


def test_candidate_status_is_read_only_and_reports_all_candidates(monkeypatch):
    monkeypatch.setattr(
        ext,
        "probe_candidate",
        lambda adapter_id, **_: RuntimeProbe(
            adapter_id=adapter_id,
            available=False,
            executable=None,
            actual_version=None,
            expected_version="test",
            version_matches_contract=None,
            reason="not installed",
        ),
    )
    rows = candidate_status()
    assert {row["adapter_id"] for row in rows} == {
        "docling",
        "mineru",
        "paddle-pp-structure-v3",
        "marker",
    }
    assert all(row["available"] is False for row in rows)



def test_paddle_runtime_shim_is_standalone_valid_python():
    shim = (
        ext.repository_root()
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "paddle_ppstructurev3.py"
    )
    assert shim.is_file()
    ast.parse(shim.read_text(encoding="utf-8"))
    text = shim.read_text(encoding="utf-8")
    assert "from paddleocr import PPStructureV3" in text
    assert "research_engine" not in text


def test_docling_native_runtime_shim_is_model_free_python():
    shim = (
        ext.repository_root()
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_native.py"
    )
    assert shim.is_file()
    text = shim.read_text(encoding="utf-8")
    ast.parse(text)
    assert "NativePdfPipelineOptions" in text
    assert "NativePdfFormatOption" in text
    assert "generate_page_images=False" in text
    assert "generate_picture_images=False" in text
    assert "ocr" not in text.lower()
    assert "vlm" not in text.lower()
    assert "DOCLING_NATIVE_SDK_SELF_TEST=PASS" in text


def test_docling_standard_no_ocr_runtime_shim_is_bounded_local_structure_python():
    shim = (
        ext.repository_root()
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_standard_no_ocr.py"
    )
    assert shim.is_file()
    text = shim.read_text(encoding="utf-8")
    ast.parse(text)
    assert "PdfPipelineOptions" in text
    assert "do_ocr=False" in text
    assert "do_table_structure=True" in text
    assert "do_formula_enrichment=False" in text
    assert "do_code_enrichment=False" in text
    assert "generate_page_images=False" in text
    assert "generate_picture_images=False" in text
    assert "enable_remote_services=False" in text
    assert "TableFormerMode.FAST" in text
    assert "RESEARCH_ENGINE_DOCLING_ARTIFACTS" in text
    assert "prefetch_structure_models" in text
    assert "LayoutObjectDetectionOptions" in text
    assert "TableStructureModel.download_models" in text
    assert "docling.utils.model_downloader" not in text
    assert "easyocr" not in text.lower()
    assert "rapidocr" not in text.lower()
    assert "DOCLING_STANDARD_NO_OCR_PREFETCH=PASS" in text
    assert "DOCLING_STANDARD_NO_OCR_SDK_SELF_TEST=PASS" in text


def test_docling_standard_no_ocr_contract_includes_headless_opencv():
    registry = json.loads(
        (
            ext.repository_root()
            / "benchmarks"
            / "document_intelligence"
            / "candidate-runtime-contracts.v1.json"
        ).read_text(encoding="utf-8")
    )
    docling = next(
        item for item in registry["candidates"]
        if item["adapter_id"] == "docling"
    )
    profile = docling["install_profiles"]["standard_no_ocr"]
    assert "docling-slim[convert-core,format-pdf,models-local]==2.131.0" in profile
    assert "docling-ibm-models[opencv-python-headless]>=4.0.2,<5" in profile
    assert "opencv-python]" not in profile
