from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader, PdfWriter

from research_engine import residual_source_audit as audit
from research_engine.document_docling import normalize_docling_document
from research_engine.document_package import BlockKind


ROOT = Path(__file__).parents[1]


def _region(*, page_number=1, bbox=None):
    return SimpleNamespace(page_number=page_number, pdf_bbox=bbox)


def _block(
    text,
    bbox,
    *,
    block_id="b",
    kind=BlockKind.PARAGRAPH,
    parameters=None,
):
    derivation = ()
    if parameters is not None:
        derivation = (SimpleNamespace(parameters=parameters),)
    return SimpleNamespace(
        id=block_id,
        kind=kind,
        markdown=text,
        source_region=_region(bbox=bbox),
        derivation=derivation,
    )


def _asset(bbox, *, caption_block_id=None, asset_id="a"):
    return SimpleNamespace(
        id=asset_id,
        caption_block_id=caption_block_id,
        source_region=_region(bbox=bbox),
    )


def _package(*, blocks=(), figures=(), tables=(), equations=()):
    return SimpleNamespace(
        blocks=list(blocks),
        figures=list(figures),
        tables=list(tables),
        equations=list(equations),
    )


def _reader(tmp_path, *, width=600, height=800):
    path = tmp_path / "one-page.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=height)
    with path.open("wb") as handle:
        writer.write(handle)
    return PdfReader(str(path))


def test_scan_opener_fires_only_with_visible_nontext(tmp_path):
    reader = _reader(tmp_path)
    native = _package(
        blocks=[
            _block(
                "",
                (0, 0, 600, 500),
                block_id="picture-1",
                kind=BlockKind.FIGURE,
                parameters={"caption_text": ""},
            )
        ]
    )
    standard = _package()

    metrics = audit._page_metrics(
        native=native,
        standard=standard,
        reader=reader,
        page_number=1,
    )

    assert metrics["native_nonspace_characters"] == 0
    assert metrics["figure_bbox_page_coverage"] >= 0.50
    assert metrics["residual_openers"]["SCAN_IMAGE_ONLY_OR_TEXTLESS_PAGE"] is True


def test_low_confidence_text_opener_uses_intrinsic_text_signal(tmp_path):
    reader = _reader(tmp_path)
    numeric = " ".join(str(value) for value in range(30))
    native = _package(blocks=[_block(numeric, (40, 40, 560, 200))])
    standard = _package(blocks=[_block(numeric, (40, 40, 560, 200))])

    metrics = audit._page_metrics(
        native=native,
        standard=standard,
        reader=reader,
        page_number=1,
    )

    assert metrics["native_nonspace_characters"] >= 32
    assert metrics["alphabetic_token_ratio"] < 0.45
    assert (
        metrics["residual_openers"]["LOW_CONFIDENCE_TEXT_OR_OCR_RECOVERY"]
        is True
    )


def test_layout_opener_detects_interleaved_columns(tmp_path):
    reader = _reader(tmp_path)
    blocks = [
        _block("left one useful text", (40, 80, 250, 130), block_id="l1"),
        _block("right one useful text", (350, 80, 560, 130), block_id="r1"),
        _block("left two useful text", (40, 160, 250, 210), block_id="l2"),
        _block("right two useful text", (350, 160, 560, 210), block_id="r2"),
    ]
    native = _package(blocks=blocks)
    standard = _package(blocks=blocks)

    metrics = audit._page_metrics(
        native=native,
        standard=standard,
        reader=reader,
        page_number=1,
    )

    assert metrics["compressed_column_sequence"] == ["L", "R", "L", "R"]
    assert metrics["column_switches"] == 3
    assert metrics["interleaved_column_order"] is True
    assert (
        metrics["residual_openers"]["LAYOUT_OR_READING_ORDER_ANOMALY"]
        is True
    )


def test_semantic_region_locator_uses_normalized_figure_block_linkage():
    caption = _block(
        "Figure 6. Training error curves.",
        (50, 650, 550, 700),
        block_id="cap-6",
        kind=BlockKind.CAPTION,
    )
    figure = _block(
        "",
        (60, 300, 540, 640),
        block_id="fig-6",
        kind=BlockKind.FIGURE,
        parameters={
            "caption_text": "Figure 6. Training error curves.",
            "caption_refs": ["#/texts/42"],
        },
    )
    standard = _package(blocks=[caption, figure])

    locator = audit._semantic_region_locator(
        standard=standard,
        target={"page_index_zero_based": 0, "figure_label": "Figure 6"},
    )

    assert locator["status"] == "LINKED_FIGURE"
    assert locator["linked_figures"][0]["figure_block_id"] == "fig-6"
    assert locator["linked_figures"][0]["caption_refs"] == ["#/texts/42"]
    assert locator["linked_figures"][0]["pdf_bbox"] == [60, 300, 540, 640]


def test_real_docling_normalizer_exposes_semantic_figure_as_block(tmp_path):
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    with source.open("wb") as handle:
        writer.write(handle)

    raw = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "pages": {
            "1": {
                "page_no": 1,
                "size": {"width": 600.0, "height": 800.0},
            }
        },
        "body": {
            "children": [
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/0"},
            ]
        },
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 60.0,
                            "t": 300.0,
                            "r": 540.0,
                            "b": 640.0,
                            "coord_origin": "TOPLEFT",
                        },
                    }
                ],
                "captions": [{"$ref": "#/texts/0"}],
            }
        ],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "caption",
                "text": "Figure 6. Training error curves.",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 50.0,
                            "t": 650.0,
                            "r": 550.0,
                            "b": 700.0,
                            "coord_origin": "TOPLEFT",
                        },
                    }
                ],
            }
        ],
    }
    raw_path = tmp_path / "docling.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    package = normalize_docling_document(
        docling_json_path=raw_path,
        source_pdf_path=source,
        source_artifact_id="benchmark-source:test",
        source_pdf_sha256=audit.sha256_file(source),
        output_directory=tmp_path / "package",
        docling_mode="standard_no_ocr",
    )

    assert package.figures == ()
    assert any(block.kind == BlockKind.FIGURE for block in package.blocks)

    locator = audit._semantic_region_locator(
        standard=package,
        target={"page_index_zero_based": 0, "figure_label": "Figure 6"},
    )
    assert locator["status"] == "LINKED_FIGURE"
    assert locator["linked_figures"][0]["pdf_bbox"] == [60.0, 300.0, 540.0, 640.0]


def test_fetch_registered_sources_hashes_only_valid_pdfs(tmp_path, monkeypatch):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    payload = io.BytesIO()
    writer.write(payload)
    pdf_bytes = payload.getvalue()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_urlopen(request, timeout):
        assert timeout == 120
        assert request.full_url.startswith("https://example.invalid/")
        return Response(pdf_bytes)

    monkeypatch.setattr(audit.urllib.request, "urlopen", fake_urlopen)

    for case_id in audit.REQUIRED_CASES:
        (case_dir / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "source": {
                        "pdf_url": f"https://example.invalid/{case_id}.pdf",
                        "expected_page_count_from_public_source": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "fetch.json"
    report = audit.fetch_registered_sources(
        case_dir=case_dir,
        source_root=tmp_path / "sources",
        output_path=output,
    )

    assert report["decision"] == "PASS"
    assert report["candidate_model_downloads"] == 0
    assert len(report["sources"]) == 3
    assert all(len(row["sha256"]) == 64 for row in report["sources"])
    assert all(row["page_count"] == 1 for row in report["sources"])
    assert output.is_file()


def test_source_audit_runner_is_bounded_and_uses_existing_runtimes_only():
    script = (
        ROOT / "deploy" / "run-residual-source-prefetch-audit.sh"
    ).read_text(encoding="utf-8")

    assert "workspace-control-tmp" in script
    assert "research-engine-daemon.service" in script
    assert "require_stopped" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "residual_source_audit fetch" in script
    assert "residual_source_audit audit" in script
    assert "--fail-on-reject" in script
    assert "docling_native.py" in script
    assert "docling_standard_no_ocr.py" in script

    assert 'echo "STRUCTURE_GPU_PROBE_INDEX=$index" >&2' in script
    assert 'echo "STRUCTURE_GPU_PROBE_FREE_MIB=$free_mib" >&2' in script
    assert "file=sys.stderr" in script
    assert 'structure_gpu="$(select_structure_gpu)"' in script
    assert '[[ "$structure_gpu" =~ ^[0-9]+$ ]]' in script

    forbidden = [
        "pip install paddleocr",
        "pip install vllm",
        "PaddleOCR-VL-1.6",
        "Qwen3.5-4B",
        "PP-OCRv6_medium",
        "systemctl --user start",
    ]
    for value in forbidden:
        assert value not in script
