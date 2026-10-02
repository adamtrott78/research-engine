from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_benchmark import (
    _resolve,
    detect_mathpix_image_mode,
    load_benchmark_cases,
    oracle_comparison,
    sha256_file,
)
from .document_docling import normalize_docling_document
from .document_package import PaperPackage
from .document_structure_eval import evaluate_paper_package_structure


def _count_agreement(candidate: int, oracle: int) -> float:
    if candidate == 0 and oracle == 0:
        return 1.0
    return min(candidate, oracle) / max(candidate, oracle)


def _find_docling_json(case_root: Path, case_id: str) -> Path:
    expected = case_root / f"{case_id}.json"
    if expected.is_file():
        return expected
    candidates = [
        path
        for path in sorted(case_root.glob("*.json"))
        if path.name not in {"benchmark-result.json", "manifest.json", "package.json"}
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{case_id}: expected exactly one Docling structured JSON in "
            f"{case_root}, found {[p.name for p in candidates]}"
        )
    return candidates[0]


def _load_package(package_root: Path) -> PaperPackage:
    path = package_root / "package.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return PaperPackage.model_validate_json(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], *path: str) -> float:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(part)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _mean(rows: list[dict[str, Any]], *path: str) -> float:
    if not rows:
        return 1.0
    return sum(_metric(row, *path) for row in rows) / len(rows)


def _candidate_minus_native(
    candidate_comparison: dict[str, Any],
    candidate_fidelity: dict[str, Any],
    native_comparison: dict[str, Any],
    native_fidelity: dict[str, Any],
) -> dict[str, float]:
    return {
        "text_fidelity": _metric(candidate_comparison, "text_fidelity")
        - _metric(native_comparison, "text_fidelity"),
        "reading_order": _metric(candidate_comparison, "reading_order")
        - _metric(native_comparison, "reading_order"),
        "table_content_f1_on_matched": _metric(
            candidate_fidelity, "tables", "content_f1_on_matched"
        )
        - _metric(native_fidelity, "tables", "content_f1_on_matched"),
        "table_aligned_cell_f1_on_matched": _metric(
            candidate_fidelity, "tables", "aligned_cell_f1_on_matched"
        )
        - _metric(native_fidelity, "tables", "aligned_cell_f1_on_matched"),
        "equation_content_f1_on_matched": _metric(
            candidate_fidelity, "equations", "content_f1_on_matched"
        )
        - _metric(native_fidelity, "equations", "content_f1_on_matched"),
        "figure_caption_content_f1_on_matched": _metric(
            candidate_fidelity, "figure_captions", "content_f1_on_matched"
        )
        - _metric(native_fidelity, "figure_captions", "content_f1_on_matched"),
        "heading_content_f1_on_matched": _metric(
            candidate_fidelity, "headings", "content_f1_on_matched"
        )
        - _metric(native_fidelity, "headings", "content_f1_on_matched"),
    }


def run_docling_normalization_benchmark(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    raw_docling_root: str | Path,
    output_root: str | Path,
    docling_version: str = "2.131.0",
    docling_mode: str = "native",
    native_package_root: str | Path | None = None,
) -> dict[str, Any]:
    bundle = Path(bundle_root)
    raw_root = Path(raw_docling_root)
    output = Path(output_root)
    native_root = Path(native_package_root) if native_package_root is not None else None
    output.mkdir(parents=True, exist_ok=True)

    cases = load_benchmark_cases(case_manifest_path)
    rows: list[dict[str, Any]] = []
    mode_directory = f"docling-{docling_mode}"

    for case in cases:
        source = _resolve(bundle, case.source_location)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != case.source_pdf_byte_size:
            raise ValueError(f"{case.case_id}: source PDF byte size mismatch")
        if sha256_file(source) != case.source_pdf_sha256:
            raise ValueError(f"{case.case_id}: source PDF SHA-256 mismatch")

        raw_case = raw_root / case.case_id / mode_directory
        if not raw_case.is_dir():
            raise FileNotFoundError(raw_case)
        structured_json = _find_docling_json(raw_case, case.case_id)

        package_output = output / case.case_id / "paper-package-v1"
        package = normalize_docling_document(
            docling_json_path=structured_json,
            source_pdf_path=source,
            source_artifact_id=f"source-artifact:sha256:{case.source_pdf_sha256}",
            source_pdf_sha256=case.source_pdf_sha256,
            output_directory=package_output,
            docling_version=docling_version,
            docling_mode=docling_mode,
        )

        diagnostics = json.loads(
            (package_output / package.manifest.diagnostics_path).read_text(
                encoding="utf-8"
            )
        )
        structure = diagnostics["structure_inventory"]
        normalized_markdown = (
            package_output / package.manifest.document_markdown_path
        ).read_text(encoding="utf-8")

        comparison: dict[str, Any] = {}
        structural_agreement: dict[str, Any] = {}
        content_fidelity: dict[str, Any] = {}
        native_baseline: dict[str, Any] | None = None
        candidate_minus_native: dict[str, float] | None = None

        if case.mathpix_oracle_location:
            oracle = _resolve(bundle, case.mathpix_oracle_location)
            if not oracle.is_file():
                raise FileNotFoundError(oracle)
            if sha256_file(oracle) != case.mathpix_oracle_sha256:
                raise ValueError(f"{case.case_id}: Mathpix oracle SHA-256 mismatch")
            oracle_text = oracle.read_text(encoding="utf-8", errors="replace")
            detected = detect_mathpix_image_mode(oracle_text)
            if (
                case.mathpix_image_mode is not None
                and detected != case.mathpix_image_mode
            ):
                raise ValueError(
                    f"{case.case_id}: Mathpix image mode mismatch: "
                    f"expected {case.mathpix_image_mode.value}, "
                    f"observed {detected.value}"
                )
            comparison = oracle_comparison(
                native_markdown=normalized_markdown,
                oracle_markdown=oracle_text,
            )
            oracle_inventory = comparison["oracle_structural_inventory"]
            structural_agreement = {
                "table_count_agreement": _count_agreement(
                    int(structure["tables"]),
                    int(oracle_inventory["tables"]),
                ),
                "tabular_block_count_agreement": _count_agreement(
                    int(structure["tables"]),
                    int(oracle_inventory["tabular_blocks"]),
                ),
                "formula_count_agreement": _count_agreement(
                    int(structure["formulas"]),
                    int(oracle_inventory["display_equations"]),
                ),
                "picture_count_agreement": _count_agreement(
                    int(structure["pictures"]),
                    int(oracle_inventory["figures"]),
                ),
                "candidate_counts": {
                    "tables": int(structure["tables"]),
                    "formulas": int(structure["formulas"]),
                    "pictures": int(structure["pictures"]),
                },
                "oracle_counts": {
                    "table_environments": int(oracle_inventory["tables"]),
                    "tabular_blocks": int(oracle_inventory["tabular_blocks"]),
                    "display_equations": int(
                        oracle_inventory["display_equations"]
                    ),
                    "figures": int(oracle_inventory["figures"]),
                },
            }
            content_fidelity = evaluate_paper_package_structure(
                package=package,
                package_root=package_output,
                mathpix_markdown=oracle_text,
            )

            if native_root is not None:
                native_package_output = native_root / case.case_id / "paper-package-v1"
                native_package = _load_package(native_package_output)
                if native_package.manifest.source_pdf_sha256 != case.source_pdf_sha256:
                    raise ValueError(
                        f"{case.case_id}: native package source SHA-256 mismatch"
                    )
                if native_package.manifest.source_page_count != case.page_count:
                    raise ValueError(
                        f"{case.case_id}: native package source page count mismatch"
                    )
                native_markdown = (
                    native_package_output
                    / native_package.manifest.document_markdown_path
                ).read_text(encoding="utf-8")
                native_comparison = oracle_comparison(
                    native_markdown=native_markdown,
                    oracle_markdown=oracle_text,
                )
                native_fidelity = evaluate_paper_package_structure(
                    package=native_package,
                    package_root=native_package_output,
                    mathpix_markdown=oracle_text,
                )
                native_baseline = {
                    "package_id": native_package.manifest.package_id,
                    "compiler_identity": native_package.manifest.compiler_identity,
                    "oracle_comparison": native_comparison,
                    "content_fidelity": native_fidelity,
                }
                candidate_minus_native = _candidate_minus_native(
                    comparison,
                    content_fidelity,
                    native_comparison,
                    native_fidelity,
                )

        pictures = int(structure["pictures"])
        row = {
            "case_id": case.case_id,
            "package_id": package.manifest.package_id,
            "status": "PASS_NORMALIZED",
            "docling_mode": docling_mode,
            "source_pdf_sha256": case.source_pdf_sha256,
            "docling_structured_json": str(structured_json),
            "package_output": str(package_output),
            "blocks": package.manifest.blocks,
            "tables_materialized": package.manifest.tables,
            "equations_materialized": package.manifest.equations,
            "figures_materialized": package.manifest.figures,
            "structure_inventory": structure,
            "provenance_accuracy_proxy": {
                "body_item_provenance_coverage": structure["provenance_coverage"],
                "body_item_bbox_coverage": structure["bbox_coverage"],
                "multi_provenance_items": structure["multi_provenance_items"],
            },
            "picture_caption_coverage": (
                int(structure["picture_captioned"]) / pictures
                if pictures
                else 1.0
            ),
            "oracle_comparison": comparison,
            "structure_count_agreement": structural_agreement,
            "content_fidelity": content_fidelity,
            "native_package_baseline": native_baseline,
            "candidate_minus_native": candidate_minus_native,
            "warnings": list(package.manifest.warnings),
        }
        (package_output / "normalization-benchmark-result.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(row)

    aggregate: dict[str, Any] = {
        "mean_provenance_coverage": (
            sum(
                row["provenance_accuracy_proxy"][
                    "body_item_provenance_coverage"
                ]
                for row in rows
            )
            / len(rows)
            if rows
            else 1.0
        ),
        "mean_bbox_coverage": (
            sum(
                row["provenance_accuracy_proxy"]["body_item_bbox_coverage"]
                for row in rows
            )
            / len(rows)
            if rows
            else 1.0
        ),
        "mean_picture_caption_coverage": (
            sum(row["picture_caption_coverage"] for row in rows) / len(rows)
            if rows
            else 1.0
        ),
        "total_tables_materialized": sum(
            row["tables_materialized"] for row in rows
        ),
        "total_equations_materialized": sum(
            row["equations_materialized"] for row in rows
        ),
        "total_figures_materialized": sum(
            row["figures_materialized"] for row in rows
        ),
        "content_fidelity": {
            "mean_table_content_f1_on_matched": _mean(
                rows, "content_fidelity", "tables", "content_f1_on_matched"
            ),
            "mean_table_shape_similarity_on_matched": _mean(
                rows, "content_fidelity", "tables", "shape_similarity_on_matched"
            ),
            "mean_table_aligned_cell_f1_on_matched": _mean(
                rows, "content_fidelity", "tables", "aligned_cell_f1_on_matched"
            ),
            "mean_equation_content_f1_on_matched": _mean(
                rows, "content_fidelity", "equations", "content_f1_on_matched"
            ),
            "mean_figure_caption_content_f1_on_matched": _mean(
                rows,
                "content_fidelity",
                "figure_captions",
                "content_f1_on_matched",
            ),
            "mean_figure_caption_association_rate": _mean(
                rows,
                "content_fidelity",
                "figure_captions",
                "candidate_caption_association_rate",
            ),
            "mean_heading_content_f1_on_matched": _mean(
                rows, "content_fidelity", "headings", "content_f1_on_matched"
            ),
            "mean_heading_reference_coverage_at_f1_0_5": _mean(
                rows,
                "content_fidelity",
                "headings",
                "reference_coverage_at_f1_0_5",
            ),
        },
    }
    if native_root is not None and rows:
        aggregate["candidate_minus_native"] = {
            key: sum(
                float(row["candidate_minus_native"][key])
                for row in rows
                if row["candidate_minus_native"] is not None
            )
            / len(rows)
            for key in (
                "text_fidelity",
                "reading_order",
                "table_content_f1_on_matched",
                "table_aligned_cell_f1_on_matched",
                "equation_content_f1_on_matched",
                "figure_caption_content_f1_on_matched",
                "heading_content_f1_on_matched",
            )
        }

    summary = {
        "schema_version": 2,
        "adapter_id": "docling",
        "mode": docling_mode,
        "normalization_status": "PAPER_PACKAGE_V1",
        "authority": "ORIGINAL_PDF",
        "reference": "MATHPIX_MANUAL_REFERENCE",
        "count_agreement_is_separate_from_content_fidelity": True,
        "native_package_root": str(native_root) if native_root is not None else None,
        "cases": len(rows),
        "total_pages": sum(case.page_count for case in cases),
        "results": rows,
        "aggregate": aggregate,
    }
    (output / "docling-normalization-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
