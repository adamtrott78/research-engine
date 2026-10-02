import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

import research_engine.document_benchmark as document_benchmark
from pydantic import ValidationError

from research_engine.document_benchmark import (
    NativePdfAdapter,
    detect_mathpix_image_mode,
    run_native_benchmark,
    sha256_file,
)
from research_engine.document_external_adapters import ExternalRunReceipt
from research_engine.document_package import (
    DocumentBenchmarkCase,
    OracleImageMode,
)


def _write_blank_pdf(path: Path, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def test_detect_mathpix_image_mode_from_content_not_extension():
    assert (
        detect_mathpix_image_mode(
            r"\includegraphics{https://cdn.mathpix.com/cropped/a.jpg}"
        )
        == OracleImageMode.REMOTE_IMAGES
    )
    assert (
        detect_mathpix_image_mode(r"\includegraphics{images/figure-1.png}")
        == OracleImageMode.LOCAL_IMAGES
    )
    assert (
        detect_mathpix_image_mode(
            r"\includegraphics{images/a.png}"
            "\n"
            r"\includegraphics{https://cdn.mathpix.com/b.png}"
        )
        == OracleImageMode.MIXED
    )
    assert detect_mathpix_image_mode("# no images") == OracleImageMode.NO_IMAGES


def test_benchmark_case_requires_oracle_hash_when_oracle_is_present():
    with pytest.raises(ValidationError, match="mathpix_oracle_sha256"):
        DocumentBenchmarkCase(
            case_id="case",
            source_pdf_sha256="a" * 64,
            source_pdf_byte_size=1,
            page_count=1,
            difficulty_tags=("simple",),
            source_location="paper.pdf",
            mathpix_oracle_location="paper.md",
            mathpix_image_mode=OracleImageMode.NO_IMAGES,
        )


def test_native_adapter_writes_page_grounded_package(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    _write_blank_pdf(pdf, pages=2)
    source_sha = sha256_file(pdf)

    output = tmp_path / "package"
    package = NativePdfAdapter("plain").compile(
        source_pdf_path=str(pdf),
        source_artifact_id=f"source-artifact:sha256:{source_sha}",
        source_pdf_sha256=source_sha,
        output_directory=str(output),
    )

    assert package.manifest.source_page_count == 2
    assert len(package.blocks) == 2
    assert [block.source_region.page_number for block in package.blocks] == [1, 2]
    assert package.manifest.authority == "DERIVED_FROM_AUTHORITATIVE_PDF"
    assert (output / "manifest.json").is_file()
    assert (output / "document.md").is_file()
    assert (output / "blocks.json").is_file()
    assert (output / "diagnostics.json").is_file()
    markdown = (output / "document.md").read_text()
    assert "<!-- source-page: 1 -->" in markdown
    assert "<!-- source-page: 2 -->" in markdown


def test_native_benchmark_validates_bundle_and_is_non_materializing(tmp_path: Path):
    bundle = tmp_path / "bundle"
    paper_dir = bundle / "papers" / "case"
    paper_dir.mkdir(parents=True)

    pdf = paper_dir / "case.pdf"
    oracle = paper_dir / "case.md"
    _write_blank_pdf(pdf)
    oracle.write_text(
        "# Oracle\n\nMathpix reference with no image assets.\n",
        encoding="utf-8",
    )

    source_sha = sha256_file(pdf)
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()
    cases = {
        "schema_version": 1,
        "corpus_id": "test",
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": source_sha,
                "source_pdf_byte_size": pdf.stat().st_size,
                "page_count": 1,
                "difficulty_tags": ["simple"],
                "source_location": "papers/case/case.pdf",
                "mathpix_oracle_location": "papers/case/case.md",
                "mathpix_oracle_sha256": oracle_sha,
                "mathpix_image_mode": "NO_IMAGES",
                "expected_features": {},
            }
        ],
    }
    case_manifest = tmp_path / "cases.json"
    case_manifest.write_text(json.dumps(cases), encoding="utf-8")
    output = tmp_path / "results"

    summary = run_native_benchmark(
        case_manifest_path=case_manifest,
        bundle_root=bundle,
        output_root=output,
    )

    assert summary["cases"] == 1
    assert summary["total_pages_per_adapter"] == 1
    assert summary["result_rows"] == 2
    assert summary["adapters"] == ["native-pypdf-plain", "native-pypdf-layout"]
    result = summary["results"][0]
    assert result["status"] == "PASS"
    assert result["adapter_id"] == "native-pypdf-plain"
    assert result["quality"]["provenance_accuracy"] == 1.0
    assert result["mathpix_image_mode"] == "NO_IMAGES"
    assert (output / "benchmark-summary.json").is_file()
    assert (
        output / "case" / "native-pypdf-plain" / "benchmark-result.json"
    ).is_file()


def test_native_benchmark_fails_closed_on_source_hash_mismatch(tmp_path: Path):
    bundle = tmp_path / "bundle"
    paper_dir = bundle / "papers" / "case"
    paper_dir.mkdir(parents=True)
    pdf = paper_dir / "case.pdf"
    _write_blank_pdf(pdf)

    manifest = {
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": "a" * 64,
                "source_pdf_byte_size": pdf.stat().st_size,
                "page_count": 1,
                "difficulty_tags": ["simple"],
                "source_location": "papers/case/case.pdf",
            }
        ]
    }
    case_manifest = tmp_path / "cases.json"
    case_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_native_benchmark(
            case_manifest_path=case_manifest,
            bundle_root=bundle,
            output_root=tmp_path / "results",
        )



def test_external_benchmark_records_raw_output_without_fake_provenance(
    tmp_path: Path,
    monkeypatch,
):
    bundle = tmp_path / "bundle"
    paper_dir = bundle / "papers" / "case"
    paper_dir.mkdir(parents=True)
    pdf = paper_dir / "case.pdf"
    oracle = paper_dir / "case.md"
    _write_blank_pdf(pdf)
    oracle.write_text(
        "# Oracle\n\nExample benchmark text in stable order.\n",
        encoding="utf-8",
    )

    source_sha = sha256_file(pdf)
    oracle_sha = hashlib.sha256(oracle.read_bytes()).hexdigest()
    manifest = {
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": source_sha,
                "source_pdf_byte_size": pdf.stat().st_size,
                "page_count": 1,
                "difficulty_tags": ["simple"],
                "source_location": "papers/case/case.pdf",
                "mathpix_oracle_location": "papers/case/case.md",
                "mathpix_oracle_sha256": oracle_sha,
                "mathpix_image_mode": "NO_IMAGES",
            }
        ]
    }
    case_manifest = tmp_path / "cases.json"
    case_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        document_benchmark,
        "candidate_spec",
        lambda *_args, **_kwargs: {
            "current_version": "2.131.0",
            "modes": {
                "standard": {
                    "class": "SPECIALIZED_LOCAL_MODEL",
                    "model_inference": True,
                }
            },
        },
    )

    def fake_run_candidate(
        adapter_id,
        mode,
        *,
        source_pdf,
        output_dir,
        **_kwargs,
    ):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        md = output / "document.md"
        md.write_text(
            "# Candidate\n\nExample benchmark text in stable order.\n",
            encoding="utf-8",
        )
        structured = {
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
                "children": [{"$ref": "#/texts/0"}],
            },
            "groups": [],
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "text",
                    "orig": "Example benchmark text in stable order.",
                    "text": "Example benchmark text in stable order.",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 10,
                                "t": 20,
                                "r": 300,
                                "b": 40,
                                "coord_origin": "TOPLEFT",
                            },
                            "charspan": [0, 39],
                        }
                    ],
                }
            ],
            "tables": [],
            "pictures": [],
        }
        (output / "case.json").write_text(
            json.dumps(structured),
            encoding="utf-8",
        )
        return ExternalRunReceipt(
            adapter_id=adapter_id,
            mode=mode,
            command=("fake", str(source_pdf)),
            output_dir=str(output),
            markdown_path=str(md),
            elapsed_seconds=0.25,
            stdout_tail="",
            stderr_tail="",
            actual_version="2.131.0",
        )

    monkeypatch.setattr(
        document_benchmark,
        "run_candidate",
        fake_run_candidate,
    )

    output = tmp_path / "results"
    summary = document_benchmark.run_external_benchmark(
        case_manifest_path=case_manifest,
        bundle_root=bundle,
        output_root=output,
        adapter_id="docling",
        mode="standard",
    )

    assert summary["normalization_status"] == "RAW_EXTERNAL_OUTPUT"
    result = summary["results"][0]
    assert result["package_id"] is None
    assert result["normalization_status"] == "RAW_EXTERNAL_OUTPUT"
    assert result["quality"]["provenance_accuracy"] is None
    assert result["specialized_model_pages"] == 1
    assert result["raw_output_paths"] == ["document.md"]
    assert result["status"] == "PASS_RAW"
    assert result["structured_json_path"] == "case.json"
    assert result["structured_inventory"]["ordered_body_items"] == 1
    assert result["structured_inventory"]["provenance_coverage"] == 1.0
