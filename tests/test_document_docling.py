import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from research_engine.document_docling import (
    docling_structure_inventory,
    normalize_docling_document,
)
from research_engine.document_package import BlockKind


def _write_pdf(path: Path, pages: int = 1) -> str:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prov(*, page=1, top=10, bottom=30, origin="TOPLEFT"):
    return [
        {
            "page_no": page,
            "bbox": {
                "l": 20,
                "t": top,
                "r": 200,
                "b": bottom,
                "coord_origin": origin,
            },
            "charspan": [0, 10],
        }
    ]


def _docling_document():
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "fixture",
        "pages": {
            "1": {
                "page_no": 1,
                "size": {"width": 612, "height": 792},
            }
        },
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/3"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/4"},
            ],
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "title",
                "orig": "Paper Title",
                "text": "Paper Title",
                "prov": _prov(top=10, bottom=25),
            },
            {
                "self_ref": "#/texts/1",
                "label": "section_header",
                "level": 1,
                "orig": "Methods",
                "text": "Methods",
                "prov": _prov(top=30, bottom=45),
            },
            {
                "self_ref": "#/texts/2",
                "label": "paragraph",
                "orig": "Body text.",
                "text": "Body text.",
                "prov": _prov(top=50, bottom=80),
            },
            {
                "self_ref": "#/texts/3",
                "label": "formula",
                "orig": "x^2 + y^2",
                "text": "x^2 + y^2",
                "prov": _prov(top=150, bottom=175),
            },
            {
                "self_ref": "#/texts/4",
                "label": "caption",
                "orig": "Figure 1. Architecture.",
                "text": "Figure 1. Architecture.",
                "prov": _prov(top=220, bottom=240),
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": _prov(top=90, bottom=145),
                "captions": [],
                "data": {
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "A",
                        },
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "text": "B",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "1",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "text": "2",
                        },
                    ]
                },
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": _prov(top=180, bottom=215),
                "captions": [{"$ref": "#/texts/4"}],
            }
        ],
    }


def test_structure_inventory_uses_ordered_body_items():
    doc = _docling_document()
    metrics = docling_structure_inventory(doc)
    assert metrics["ordered_body_items"] == 7
    assert metrics["tables"] == 1
    assert metrics["formulas"] == 1
    assert metrics["pictures"] == 1
    assert metrics["section_headers"] == 1
    assert metrics["picture_captioned"] == 1
    assert metrics["provenance_coverage"] == 1.0
    assert metrics["bbox_coverage"] == 1.0


def test_normalizer_preserves_order_provenance_and_structured_assets(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    sha = _write_pdf(pdf)
    raw = tmp_path / "docling.json"
    raw.write_text(json.dumps(_docling_document()), encoding="utf-8")

    output = tmp_path / "package"
    package = normalize_docling_document(
        docling_json_path=raw,
        source_pdf_path=pdf,
        source_artifact_id=f"source-artifact:sha256:{sha}",
        source_pdf_sha256=sha,
        output_directory=output,
    )

    assert package.manifest.source_page_count == 1
    assert package.manifest.blocks == 7
    assert package.manifest.tables == 1
    assert package.manifest.equations == 1
    assert package.manifest.figures == 0
    assert [block.kind for block in package.blocks] == [
        BlockKind.TITLE,
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.TABLE,
        BlockKind.EQUATION,
        BlockKind.FIGURE,
        BlockKind.CAPTION,
    ]
    assert package.blocks[2].source_region.section_path == ("Methods",)
    assert package.blocks[2].source_region.raw_anchor == "#/texts/2"
    assert package.blocks[2].source_region.pdf_bbox == (20.0, 50.0, 200.0, 80.0)
    assert package.blocks[3].asset_ids == ("table-0001",)
    assert package.blocks[4].asset_ids == ("equation-0001",)
    assert package.blocks[5].markdown == ""
    assert (
        package.blocks[5].derivation[0].parameters["caption_text"]
        == "Figure 1. Architecture."
    )
    assert "| A | B |" in package.blocks[3].markdown
    assert package.equations[0].latex == "x^2 + y^2"
    assert (output / "tables/table-0001.json").is_file()
    assert (output / "tables/table-0001.md").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "blocks.json").is_file()
    assert (output / "diagnostics.json").is_file()
    assert (output / "package.json").is_file()
    assert (output / "document.md").is_file()

    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["structure_inventory"]["pictures"] == 1
    assert any("image asset materialization is deferred" in x for x in diagnostics["warnings"])


def test_bottomleft_bbox_is_converted_with_page_height(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    sha = _write_pdf(pdf)
    doc = _docling_document()
    doc["texts"][0]["prov"] = _prov(
        top=782,
        bottom=762,
        origin="BOTTOMLEFT",
    )
    raw = tmp_path / "docling.json"
    raw.write_text(json.dumps(doc), encoding="utf-8")

    package = normalize_docling_document(
        docling_json_path=raw,
        source_pdf_path=pdf,
        source_artifact_id=f"source-artifact:sha256:{sha}",
        source_pdf_sha256=sha,
        output_directory=tmp_path / "package",
    )
    assert package.blocks[0].source_region.pdf_bbox == (20.0, 10.0, 200.0, 30.0)


def test_normalizer_fails_closed_on_source_hash_mismatch(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    _write_pdf(pdf)
    raw = tmp_path / "docling.json"
    raw.write_text(json.dumps(_docling_document()), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        normalize_docling_document(
            docling_json_path=raw,
            source_pdf_path=pdf,
            source_artifact_id="source-artifact:sha256:" + "a" * 64,
            source_pdf_sha256="a" * 64,
            output_directory=tmp_path / "package",
        )


def test_normalizer_fails_closed_on_missing_item_provenance(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    sha = _write_pdf(pdf)
    doc = _docling_document()
    doc["texts"][2]["prov"] = []
    raw = tmp_path / "docling.json"
    raw.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError, match="no valid page provenance"):
        normalize_docling_document(
            docling_json_path=raw,
            source_pdf_path=pdf,
            source_artifact_id=f"source-artifact:sha256:{sha}",
            source_pdf_sha256=sha,
            output_directory=tmp_path / "package",
        )


def test_multi_provenance_is_preserved_in_derivation_metadata(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    sha = _write_pdf(pdf)
    doc = _docling_document()
    doc["texts"][2]["prov"] = _prov(top=50, bottom=60) + _prov(
        top=65,
        bottom=80,
    )
    raw = tmp_path / "docling.json"
    raw.write_text(json.dumps(doc), encoding="utf-8")

    package = normalize_docling_document(
        docling_json_path=raw,
        source_pdf_path=pdf,
        source_artifact_id=f"source-artifact:sha256:{sha}",
        source_pdf_sha256=sha,
        output_directory=tmp_path / "package",
    )
    params = package.blocks[2].derivation[0].parameters
    assert params["provenance_count"] == 2
    assert len(params["additional_provenance"]) == 1
