import hashlib
import json
from pathlib import Path

from pypdf import PdfWriter

from research_engine.document_docling_benchmark import (
    run_docling_normalization_benchmark,
)


def _write_pdf(path: Path) -> str:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prov(top: float, bottom: float):
    return [
        {
            "page_no": 1,
            "bbox": {
                "l": 20,
                "t": top,
                "r": 200,
                "b": bottom,
                "coord_origin": "TOPLEFT",
            },
            "charspan": [0, 5],
        }
    ]


def test_docling_normalization_benchmark_runs_over_raw_output(tmp_path: Path):
    bundle = tmp_path / "bundle"
    paper_dir = bundle / "papers" / "case"
    paper_dir.mkdir(parents=True)
    pdf = paper_dir / "case.pdf"
    pdf_sha = _write_pdf(pdf)

    oracle = paper_dir / "case.mmd"
    oracle.write_text(
        "# Paper\n\n"
        "\\[x^2\\]\n\n"
        "\\begin{table}\\begin{tabular}{c}A\\end{tabular}\\end{table}\n\n"
        "\\begin{figure}F\\end{figure}\n",
        encoding="utf-8",
    )
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()

    manifest = {
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": pdf_sha,
                "source_pdf_byte_size": pdf.stat().st_size,
                "page_count": 1,
                "difficulty_tags": ["fixture"],
                "source_location": "papers/case/case.pdf",
                "mathpix_oracle_location": "papers/case/case.mmd",
                "mathpix_oracle_sha256": oracle_sha,
                "mathpix_image_mode": "NO_IMAGES",
                "expected_features": {},
            }
        ]
    }
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    raw_root = tmp_path / "raw"
    raw_case = raw_root / "case" / "docling-native"
    raw_case.mkdir(parents=True)
    raw_doc = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "case",
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
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/pictures/0"},
            ],
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "title",
                "orig": "Paper",
                "text": "Paper",
                "prov": _prov(10, 25),
            },
            {
                "self_ref": "#/texts/1",
                "label": "formula",
                "orig": "x^2",
                "text": "x^2",
                "prov": _prov(100, 120),
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": _prov(30, 90),
                "captions": [],
                "data": {
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "A",
                        }
                    ]
                },
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": _prov(130, 180),
                "captions": [],
            }
        ],
    }
    (raw_case / "case.json").write_text(
        json.dumps(raw_doc),
        encoding="utf-8",
    )

    output = tmp_path / "normalized"
    summary = run_docling_normalization_benchmark(
        case_manifest_path=manifest_path,
        bundle_root=bundle,
        raw_docling_root=raw_root,
        output_root=output,
    )

    assert summary["cases"] == 1
    assert summary["total_pages"] == 1
    assert summary["normalization_status"] == "PAPER_PACKAGE_V1"
    row = summary["results"][0]
    assert row["status"] == "PASS_NORMALIZED"
    assert row["tables_materialized"] == 1
    assert row["equations_materialized"] == 1
    assert row["figures_materialized"] == 0
    assert row["provenance_accuracy_proxy"]["body_item_provenance_coverage"] == 1.0
    assert row["structure_count_agreement"]["table_count_agreement"] == 1.0
    assert row["structure_count_agreement"]["formula_count_agreement"] == 1.0
    assert row["structure_count_agreement"]["picture_count_agreement"] == 1.0
    assert (output / "docling-normalization-summary.json").is_file()
    assert (
        output
        / "case"
        / "paper-package-v1"
        / "normalization-benchmark-result.json"
    ).is_file()

def test_standard_replay_evaluates_content_and_compares_native_packages(tmp_path: Path):
    bundle = tmp_path / "bundle"
    paper_dir = bundle / "papers" / "case"
    paper_dir.mkdir(parents=True)
    pdf = paper_dir / "case.pdf"
    pdf_sha = _write_pdf(pdf)

    oracle = paper_dir / "case.mmd"
    oracle.write_text(
        "# Paper\n\n"
        "## Methods\n\n"
        "\\[E = mc^2\\]\n\n"
        "\\begin{table}\\begin{tabular}{cc}"
        "Model & Score \\\\ A & 0.9"
        "\\end{tabular}\\end{table}\n\n"
        "\\begin{figure}\\caption{System architecture and signal flow.}"
        "\\end{figure}\n",
        encoding="utf-8",
    )
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()

    manifest = {
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": pdf_sha,
                "source_pdf_byte_size": pdf.stat().st_size,
                "page_count": 1,
                "difficulty_tags": ["fixture"],
                "source_location": "papers/case/case.pdf",
                "mathpix_oracle_location": "papers/case/case.mmd",
                "mathpix_oracle_sha256": oracle_sha,
                "mathpix_image_mode": "NO_IMAGES",
                "expected_features": {},
            }
        ]
    }
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    native_raw = tmp_path / "native-raw"
    native_case = native_raw / "case" / "docling-native"
    native_case.mkdir(parents=True)
    native_doc = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "case",
        "pages": {"1": {"page_no": 1, "size": {"width": 612, "height": 792}}},
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "title",
                "text": "Paper",
                "prov": _prov(10, 25),
            },
            {
                "self_ref": "#/texts/1",
                "label": "paragraph",
                "text": "Methods E = mc^2 Model Score A 0.9",
                "prov": _prov(30, 80),
            },
        ],
        "tables": [],
        "pictures": [],
    }
    (native_case / "case.json").write_text(
        json.dumps(native_doc),
        encoding="utf-8",
    )
    native_output = tmp_path / "native-normalized"
    run_docling_normalization_benchmark(
        case_manifest_path=manifest_path,
        bundle_root=bundle,
        raw_docling_root=native_raw,
        output_root=native_output,
    )

    standard_raw = tmp_path / "standard-raw"
    standard_case = standard_raw / "case" / "docling-standard_no_ocr"
    standard_case.mkdir(parents=True)
    standard_doc = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "case",
        "pages": {"1": {"page_no": 1, "size": {"width": 612, "height": 792}}},
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/3"},
            ],
        },
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "title",
                "text": "Paper",
                "prov": _prov(10, 25),
            },
            {
                "self_ref": "#/texts/1",
                "label": "section_header",
                "level": 1,
                "text": "Methods",
                "prov": _prov(30, 45),
            },
            {
                "self_ref": "#/texts/2",
                "label": "formula",
                "text": "E = mc^2",
                "prov": _prov(100, 120),
            },
            {
                "self_ref": "#/texts/3",
                "label": "caption",
                "text": "System architecture and signal flow.",
                "prov": _prov(160, 180),
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": _prov(50, 95),
                "captions": [],
                "data": {
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "Model",
                        },
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "text": "Score",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "A",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "text": "0.9",
                        },
                    ]
                },
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": _prov(125, 155),
                "captions": [{"$ref": "#/texts/3"}],
            }
        ],
    }
    (standard_case / "case.json").write_text(
        json.dumps(standard_doc),
        encoding="utf-8",
    )

    output = tmp_path / "standard-normalized"
    summary = run_docling_normalization_benchmark(
        case_manifest_path=manifest_path,
        bundle_root=bundle,
        raw_docling_root=standard_raw,
        output_root=output,
        docling_mode="standard_no_ocr",
        native_package_root=native_output,
    )

    assert summary["mode"] == "standard_no_ocr"
    assert summary["count_agreement_is_separate_from_content_fidelity"] is True
    row = summary["results"][0]
    assert row["native_package_baseline"] is not None
    assert row["content_fidelity"]["tables"]["content_f1_on_matched"] == 1.0
    assert row["content_fidelity"]["tables"]["aligned_cell_f1_on_matched"] == 1.0
    assert row["content_fidelity"]["equations"]["content_f1_on_matched"] == 1.0
    assert (
        row["content_fidelity"]["figure_captions"][
            "candidate_caption_association_rate"
        ]
        == 1.0
    )
    assert row["content_fidelity"]["headings"]["reference_coverage_at_f1_0_5"] == 1.0
    assert row["candidate_minus_native"]["table_content_f1_on_matched"] > 0
    assert row["candidate_minus_native"]["equation_content_f1_on_matched"] > 0
    assert row["candidate_minus_native"]["figure_caption_content_f1_on_matched"] > 0

    package_manifest = json.loads(
        (
            output
            / "case"
            / "paper-package-v1"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert package_manifest["compiler_identity"] == "docling-standard-no-ocr-structured"
    assert package_manifest["compiler_config"]["docling_mode"] == "standard_no_ocr"

