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


def run_docling_normalization_benchmark(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    raw_docling_root: str | Path,
    output_root: str | Path,
    compiler_version: str = "2.131.0",
) -> dict[str, Any]:
    bundle = Path(bundle_root)
    raw_root = Path(raw_docling_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    cases = load_benchmark_cases(case_manifest_path)
    rows: list[dict[str, Any]] = []

    for case in cases:
        source = _resolve(bundle, case.source_location)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != case.source_pdf_byte_size:
            raise ValueError(f"{case.case_id}: source PDF byte size mismatch")
        if sha256_file(source) != case.source_pdf_sha256:
            raise ValueError(f"{case.case_id}: source PDF SHA-256 mismatch")

        raw_case = raw_root / case.case_id / "docling-native"
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
            compiler_version=compiler_version,
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

        pictures = int(structure["pictures"])
        row = {
            "case_id": case.case_id,
            "package_id": package.manifest.package_id,
            "status": "PASS_NORMALIZED",
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
            "warnings": list(package.manifest.warnings),
        }
        (package_output / "normalization-benchmark-result.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(row)

    summary = {
        "schema_version": 1,
        "adapter_id": "docling",
        "mode": "native",
        "normalization_status": "PAPER_PACKAGE_V1",
        "cases": len(rows),
        "total_pages": sum(case.page_count for case in cases),
        "results": rows,
        "aggregate": {
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
        },
    }
    (output / "docling-normalization-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
