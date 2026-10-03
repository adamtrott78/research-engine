import json
from pathlib import Path

from research_engine.document_formula_eval import (
    build_formula_target_plan,
    evaluate_formula_enrichment,
)
from research_engine.document_package import (
    BlockKind,
    DerivationStep,
    DocumentBlock,
    EquationAsset,
    PaperPackage,
    PaperPackageManifest,
    PdfRegion,
    ProcessingMode,
    TableAsset,
    compiler_config_sha256,
    paper_package_id,
)
from research_engine.document_routing import build_hybrid_routing_report
from research_engine.document_structure_audit import audit_table_residuals


SHA = "a" * 64


def _region(page: int = 1, section=()):
    return PdfRegion(
        page_number=page,
        pdf_bbox=(10.0, 10.0, 100.0, 30.0),
        section_path=section,
    )


def _step(**parameters):
    return (
        DerivationStep(
            processor="fixture",
            processor_version="1",
            mode=ProcessingMode.DETERMINISTIC,
            parameters=parameters,
        ),
    )


def _package(*, standard: bool) -> PaperPackage:
    config = {"mode": "standard" if standard else "native"}
    config_sha = compiler_config_sha256(config)
    identity = "fixture-standard" if standard else "fixture-native"
    version = "1"

    if not standard:
        blocks = (
            DocumentBlock(
                id="native-text",
                kind=BlockKind.PARAGRAPH,
                markdown="Native lexical content.",
                source_region=_region(),
                derivation=_step(),
            ),
        )
        tables = ()
        equations = ()
    else:
        blocks = (
            DocumentBlock(
                id="heading",
                kind=BlockKind.HEADING,
                markdown="Methods",
                source_region=_region(section=("Methods",)),
                derivation=_step(),
            ),
            DocumentBlock(
                id="figure-captioned",
                kind=BlockKind.FIGURE,
                markdown="",
                source_region=_region(),
                derivation=_step(caption_text="Architecture."),
            ),
            DocumentBlock(
                id="figure-uncaptioned",
                kind=BlockKind.FIGURE,
                markdown="",
                source_region=_region(),
                derivation=_step(caption_text=""),
            ),
            DocumentBlock(
                id="table-block",
                kind=BlockKind.TABLE,
                markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
                source_region=_region(),
                derivation=_step(),
                asset_ids=("table-0001",),
            ),
            DocumentBlock(
                id="formula-block",
                kind=BlockKind.EQUATION,
                markdown="raw formula placeholder",
                source_region=_region(page=2),
                derivation=_step(),
                asset_ids=("equation-0001",),
            ),
        )
        tables = (
            TableAsset(
                id="table-0001",
                source_region=_region(),
                json_path="tables/table-0001.json",
                markdown_path="tables/table-0001.md",
                derivation=_step(),
            ),
        )
        equations = (
            EquationAsset(
                id="equation-0001",
                source_region=_region(page=2),
                latex="raw formula placeholder",
                derivation=_step(),
            ),
        )

    manifest = PaperPackageManifest(
        package_id=paper_package_id(
            source_pdf_sha256=SHA,
            compiler_identity=identity,
            compiler_version=version,
            compiler_config_sha256=config_sha,
        ),
        source_artifact_id=f"source-artifact:sha256:{SHA}",
        source_pdf_sha256=SHA,
        source_pdf_byte_size=100,
        source_page_count=2,
        compiler_identity=identity,
        compiler_version=version,
        compiler_config=config,
        compiler_config_sha256=config_sha,
        blocks=len(blocks),
        tables=len(tables),
        figures=0,
        equations=len(equations),
    )
    return PaperPackage(
        manifest=manifest,
        blocks=blocks,
        tables=tables,
        equations=equations,
    )


def _write_package(root: Path, *, standard: bool):
    case_root = root / "case" / "paper-package-v1"
    case_root.mkdir(parents=True)
    package = _package(standard=standard)
    (case_root / "package.json").write_text(
        package.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if standard:
        table_dir = case_root / "tables"
        table_dir.mkdir()
        (table_dir / "table-0001.json").write_text(
            json.dumps(
                {
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
                    }
                }
            ),
            encoding="utf-8",
        )


def _manifest(path: Path):
    data = {
        "cases": [
            {
                "case_id": "case",
                "source_pdf_sha256": SHA,
                "source_pdf_byte_size": 100,
                "page_count": 2,
                "difficulty_tags": ["fixture"],
                "source_location": "papers/case/case.pdf",
                "mathpix_oracle_location": "papers/case/case.mmd",
                "mathpix_oracle_sha256": "b" * 64,
                "mathpix_image_mode": "NO_IMAGES",
                "expected_features": {},
            }
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_hybrid_router_keeps_native_substrate_and_selective_structure(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    _manifest(manifest)
    native = tmp_path / "native"
    standard = tmp_path / "standard"
    _write_package(native, standard=False)
    _write_package(standard, standard=True)

    report = build_hybrid_routing_report(
        case_manifest_path=manifest,
        native_package_root=native,
        standard_package_root=standard,
    )
    row = report["results"][0]

    assert report["benchmark_oracle_used_for_routing"] is False
    assert row["lexical_substrate"]["route"] == "DOCLING_NATIVE"
    assert len(row["headings"]) == 1
    assert len(row["semantic_figures"]) == 1
    assert len(row["unpromoted_pictures"]) == 1
    assert row["tables"][0]["status"] == "ACCEPT_STRUCTURE_WITH_PDF_VERIFY"
    assert row["tables"][0]["authoritative_pdf_verification_required"] is True
    assert row["formula_target_pages"] == [2]
    assert row["page_routes"][1]["formula_escalation"] == (
        "docling_codeformulav2_targeted"
    )


def test_table_residual_audit_exposes_candidate_and_reference_cells(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    _manifest(manifest)
    standard = tmp_path / "standard"
    _write_package(standard, standard=True)

    bundle = tmp_path / "bundle"
    oracle = bundle / "papers" / "case" / "case.mmd"
    oracle.parent.mkdir(parents=True)
    oracle.write_text(
        "\\begin{table}\\begin{tabular}{cc}"
        "X & Y \\\\ 9 & 8"
        "\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )

    report = audit_table_residuals(
        case_manifest_path=manifest,
        bundle_root=bundle,
        standard_package_root=standard,
    )
    assert report["summary"]["residuals"] == 1
    residual = report["results"][0]["residuals"][0]
    assert residual["kind"] == "LOW_FIDELITY_MATCH"
    assert residual["candidate_grid"] == [["A", "B"], ["1", "2"]]
    assert residual["reference_grid"] == [["X", "Y"], ["9", "8"]]


def _raw_doc(formula: str, page: int = 2):
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "pages": {
            "2": {
                "page_no": page,
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
                "label": "formula",
                "text": formula,
                "prov": [
                    {
                        "page_no": page,
                        "bbox": {
                            "l": 10,
                            "t": 10,
                            "r": 100,
                            "b": 30,
                            "coord_origin": "TOPLEFT",
                        },
                    }
                ],
            }
        ],
        "tables": [],
        "pictures": [],
    }


def test_formula_plan_and_evaluation_are_page_bounded(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    _manifest(manifest)

    raw = tmp_path / "raw"
    standard_case = raw / "case" / "docling-standard_no_ocr"
    standard_case.mkdir(parents=True)
    (standard_case / "case.json").write_text(
        json.dumps(_raw_doc("placeholder")),
        encoding="utf-8",
    )

    plan = build_formula_target_plan(
        case_manifest_path=manifest,
        raw_standard_root=raw,
    )
    assert plan["summary"]["target_pages"] == 1
    assert plan["cases"][0]["target_pages"] == [2]

    bundle = tmp_path / "bundle"
    oracle = bundle / "papers" / "case" / "case.mmd"
    oracle.parent.mkdir(parents=True)
    oracle.write_text("\\[E = mc^2\\]\n", encoding="utf-8")

    formula_root = tmp_path / "formula"
    page_dir = formula_root / "case" / "page-0002"
    page_dir.mkdir(parents=True)
    (page_dir / "case.json").write_text(
        json.dumps(_raw_doc("E = mc^2")),
        encoding="utf-8",
    )

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "case",
                        "content_fidelity": {
                            "equations": {
                                "content_f1_on_matched": 0.0,
                                "reference_coverage_at_f1_0_5": 0.0,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_formula_enrichment(
        case_manifest_path=manifest,
        bundle_root=bundle,
        formula_root=formula_root,
        baseline_summary_path=baseline,
    )
    assert result["summary"]["weighted_content_f1_on_matched"] == 1.0
    assert result["summary"]["weighted_reference_coverage_at_f1_0_5"] == 1.0
    assert result["summary"]["decision"] == "PROMOTION_CANDIDATE"
    assert result["results"][0]["candidate_minus_baseline"][
        "content_f1_on_matched"
    ] == 1.0


def test_formula_runtime_is_explicitly_formula_only_and_page_bounded():
    root = Path(__file__).parents[1]
    text = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_formula_enrichment.py"
    ).read_text(encoding="utf-8")

    assert 'MODEL_REPO = "docling-project/CodeFormulaV2"' in text
    assert "do_ocr=False" in text
    assert "do_table_structure=False" in text
    assert "do_code_enrichment=False" in text
    assert "do_formula_enrichment=True" in text
    assert "enable_remote_services=False" in text
    assert "page_range=(args.page, args.page)" in text
    assert 'CodeFormulaVlmOptions.from_preset(' in text
    assert '"codeformulav2"' in text
    assert "TransformersVlmEngineOptions" in text


def test_combined_followup_runner_is_stopped_daemon_and_narrowly_scoped():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-document-hybrid-routing-followup.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "document_routing" in text
    assert "document_structure_audit" in text
    assert "document_formula_eval plan" in text
    assert "docling_formula_enrichment.py" in text
    assert "document_formula_eval evaluate" in text
    assert "FORMULA_TARGET_PAGES" in text
    assert "document-benchmark-external" not in text
    assert "marker" not in text.lower()
    assert "paddle" not in text.lower()
    assert "mineru" not in text.lower()
    assert "do_ocr" not in text
