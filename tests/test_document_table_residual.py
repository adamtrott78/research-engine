import json
from pathlib import Path

from research_engine.document_package import (
    DerivationStep,
    PaperPackage,
    PaperPackageManifest,
    PdfRegion,
    ProcessingMode,
    TableAsset,
    compiler_config_sha256,
    paper_package_id,
)
from research_engine.document_table_residual import (
    build_tableformer_accurate_target_plan,
    evaluate_tableformer_accurate,
)


SHA = "a" * 64


def _manifest(path: Path):
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "source_pdf_sha256": SHA,
                        "source_pdf_byte_size": 100,
                        "page_count": 3,
                        "difficulty_tags": ["fixture"],
                        "source_location": "papers/case/case.pdf",
                        "mathpix_oracle_location": "papers/case/case.mmd",
                        "mathpix_oracle_sha256": "b" * 64,
                        "mathpix_image_mode": "NO_IMAGES",
                        "expected_features": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _step():
    return (
        DerivationStep(
            processor="fixture",
            processor_version="1",
            mode=ProcessingMode.TABLE_MODEL,
        ),
    )


def _empty_standard_package(root: Path):
    case_root = root / "case" / "paper-package-v1"
    tables = case_root / "tables"
    tables.mkdir(parents=True)

    config = {"mode": "standard_no_ocr"}
    config_sha = compiler_config_sha256(config)
    manifest = PaperPackageManifest(
        package_id=paper_package_id(
            source_pdf_sha256=SHA,
            compiler_identity="fixture-standard",
            compiler_version="1",
            compiler_config_sha256=config_sha,
        ),
        source_artifact_id=f"source-artifact:sha256:{SHA}",
        source_pdf_sha256=SHA,
        source_pdf_byte_size=100,
        source_page_count=3,
        compiler_identity="fixture-standard",
        compiler_version="1",
        compiler_config=config,
        compiler_config_sha256=config_sha,
        blocks=0,
        tables=1,
        figures=0,
        equations=0,
    )
    asset = TableAsset(
        id="table-0001",
        source_region=PdfRegion(
            page_number=2,
            pdf_bbox=(1, 1, 10, 10),
        ),
        json_path="tables/table-0001.json",
        markdown_path="tables/table-0001.md",
        derivation=_step(),
    )
    package = PaperPackage(manifest=manifest, blocks=(), tables=(asset,))
    (case_root / "package.json").write_text(
        package.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (tables / "table-0001.json").write_text(
        json.dumps({"data": {"table_cells": []}}),
        encoding="utf-8",
    )


def _audit(path: Path):
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "case_id": "case",
                        "residuals": [
                            {
                                "kind": "LOW_FIDELITY_MATCH",
                                "candidate_index": 0,
                                "reference_index": 0,
                                "candidate_shape": [0, 0],
                                "reference_shape": [2, 2],
                                "content_f1": 0.0,
                                "aligned_cell_f1": 0.0,
                            },
                            {
                                "kind": "LOW_FIDELITY_MATCH",
                                "candidate_index": 0,
                                "reference_index": 0,
                                "candidate_shape": [2, 2],
                                "reference_shape": [2, 2],
                                "content_f1": 0.4,
                                "aligned_cell_f1": 0.4,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _accurate_doc(path: Path):
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "tables": [
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
                ]
            }
        ),
        encoding="utf-8",
    )


def test_plan_selects_only_true_empty_grid_residual(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    standard = tmp_path / "standard"
    audit = tmp_path / "audit.json"
    _manifest(manifest)
    _empty_standard_package(standard)
    _audit(audit)

    plan = build_tableformer_accurate_target_plan(
        case_manifest_path=manifest,
        standard_package_root=standard,
        table_audit_path=audit,
    )

    assert plan["summary"]["target_tables"] == 1
    assert plan["summary"]["target_pages"] == 1
    assert plan["targets"][0]["page_number"] == 2
    assert plan["targets"][0]["table_asset_id"] == "table-0001"


def test_accurate_evaluation_promotes_recovered_table(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    standard = tmp_path / "standard"
    audit = tmp_path / "audit.json"
    plan_path = tmp_path / "plan.json"
    _manifest(manifest)
    _empty_standard_package(standard)
    _audit(audit)

    build_tableformer_accurate_target_plan(
        case_manifest_path=manifest,
        standard_package_root=standard,
        table_audit_path=audit,
        output_path=plan_path,
    )

    bundle = tmp_path / "bundle"
    oracle = bundle / "papers" / "case" / "case.mmd"
    oracle.parent.mkdir(parents=True)
    oracle.write_text(
        "\\begin{table}\\begin{tabular}{cc}"
        "A & B \\\\ 1 & 2"
        "\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )

    accurate = tmp_path / "accurate"
    _accurate_doc(accurate / "case" / "page-0002" / "case.json")

    result = evaluate_tableformer_accurate(
        case_manifest_path=manifest,
        bundle_root=bundle,
        target_plan_path=plan_path,
        accurate_root=accurate,
    )
    assert result["summary"]["target_tables"] == 1
    assert result["summary"]["all_targets_materialized"] is True
    assert result["summary"]["weighted_content_f1_on_matched"] == 1.0
    assert result["summary"]["weighted_aligned_cell_f1_on_matched"] == 1.0
    assert result["summary"]["decision"] == "PROMOTION_CANDIDATE"


def test_accurate_runtime_is_page_bounded_and_no_ocr():
    root = Path(__file__).parents[1]
    text = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_tableformer_accurate.py"
    ).read_text(encoding="utf-8")

    assert "TableFormerMode.ACCURATE" in text
    assert "page_range=(args.page, args.page)" in text
    assert "do_ocr=False" in text
    assert "do_formula_enrichment=False" in text
    assert "enable_remote_services=False" in text


def test_runner_writes_new_outputs_to_visible_workspace_tmp():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-tableformer-accurate-residual-gate.sh"
    ).read_text(encoding="utf-8")

    assert 'visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"' in text
    assert 'output_root="$visible_tmp_root/' in text
    assert 'legacy_tmp_root="$adam/.workspace-control/tmp"' in text
    assert 'standard_root="$legacy_standard_raw/' in text
    assert "TABLEFORMER_ACCURATE_RESIDUAL_GATE=PASS" in text
