from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_benchmark import load_benchmark_cases
from .document_docling import _table_grid
from .document_package import BlockKind, PaperPackage


def _load_package(root: Path, case_id: str) -> tuple[PaperPackage, Path]:
    package_root = root / case_id / "paper-package-v1"
    package_path = package_root / "package.json"
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    return (
        PaperPackage.model_validate_json(package_path.read_text(encoding="utf-8")),
        package_root,
    )


def _caption_text(block: Any) -> str:
    for step in block.derivation:
        value = step.parameters.get("caption_text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _table_signal(package_root: Path, table: Any) -> dict[str, Any]:
    if not table.json_path:
        return {
            "status": "RESIDUAL_REVIEW_REQUIRED",
            "confidence": 0.0,
            "rows": 0,
            "columns": 0,
            "nonempty_ratio": 0.0,
            "reason": "table has no structured JSON payload",
            "authoritative_pdf_verification_required": True,
        }

    item = json.loads((package_root / table.json_path).read_text(encoding="utf-8"))
    grid = _table_grid(item)
    rows = len(grid)
    columns = max((len(row) for row in grid), default=0)
    cells = rows * columns
    nonempty = sum(bool(cell.strip()) for row in grid for cell in row)
    nonempty_ratio = nonempty / cells if cells else 0.0

    structurally_usable = rows >= 1 and columns >= 1 and nonempty_ratio >= 0.35
    # This is deliberately an intrinsic structure proxy, not a claim that
    # extracted cell text agrees with the authoritative PDF.
    confidence = (
        min(0.95, 0.55 + 0.4 * nonempty_ratio)
        if structurally_usable
        else min(0.49, nonempty_ratio)
    )
    return {
        "status": (
            "ACCEPT_STRUCTURE_WITH_PDF_VERIFY"
            if structurally_usable
            else "RESIDUAL_REVIEW_REQUIRED"
        ),
        "confidence": confidence,
        "rows": rows,
        "columns": columns,
        "nonempty_ratio": nonempty_ratio,
        "reason": (
            "grid is structurally populated; semantic cell content remains derived"
            if structurally_usable
            else "grid is empty or sparsely populated"
        ),
        "authoritative_pdf_verification_required": True,
    }


def build_hybrid_routing_report(
    *,
    case_manifest_path: str | Path,
    native_package_root: str | Path,
    standard_package_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic native + structure-overlay routing contract.

    Mathpix is intentionally absent from this function. Production routing may
    use parser-intrinsic signals and provenance, but not a benchmark oracle.
    """

    native_root = Path(native_package_root)
    standard_root = Path(standard_package_root)
    cases = load_benchmark_cases(case_manifest_path)
    rows: list[dict[str, Any]] = []

    for case in cases:
        native, _ = _load_package(native_root, case.case_id)
        standard, standard_case_root = _load_package(standard_root, case.case_id)

        if native.manifest.source_pdf_sha256 != standard.manifest.source_pdf_sha256:
            raise ValueError(f"{case.case_id}: native/standard source SHA mismatch")
        if native.manifest.source_page_count != standard.manifest.source_page_count:
            raise ValueError(f"{case.case_id}: native/standard page-count mismatch")
        if native.manifest.source_pdf_sha256 != case.source_pdf_sha256:
            raise ValueError(f"{case.case_id}: package source SHA differs from manifest")

        headings: list[dict[str, Any]] = []
        semantic_figures: list[dict[str, Any]] = []
        unpromoted_pictures: list[dict[str, Any]] = []
        equation_pages: set[int] = set()

        for block in standard.blocks:
            if block.kind in {BlockKind.TITLE, BlockKind.HEADING}:
                headings.append(
                    {
                        "block_id": block.id,
                        "page_number": block.source_region.page_number,
                        "status": "ACCEPT_STANDARD_STRUCTURE_OVERLAY",
                        "reason": "validated heading/provenance tier",
                    }
                )
            elif block.kind == BlockKind.FIGURE:
                caption = _caption_text(block)
                row = {
                    "block_id": block.id,
                    "page_number": block.source_region.page_number,
                    "caption_text": caption,
                }
                if caption:
                    row.update(
                        {
                            "status": "ACCEPT_SEMANTIC_FIGURE",
                            "reason": "picture is explicitly caption-linked",
                        }
                    )
                    semantic_figures.append(row)
                else:
                    row.update(
                        {
                            "status": "DO_NOT_PROMOTE",
                            "reason": "picture lacks validated caption association",
                        }
                    )
                    unpromoted_pictures.append(row)
            elif block.kind == BlockKind.EQUATION:
                equation_pages.add(block.source_region.page_number)

        tables = []
        for table in standard.tables:
            signal = _table_signal(standard_case_root, table)
            tables.append(
                {
                    "asset_id": table.id,
                    "page_number": table.source_region.page_number,
                    **signal,
                }
            )

        page_routes: dict[int, dict[str, Any]] = {}
        for page_number in range(1, case.page_count + 1):
            reasons: list[str] = ["native lexical/order/provenance substrate"]
            overlay = False
            if any(row["page_number"] == page_number for row in headings):
                overlay = True
                reasons.append("validated heading structure")
            if any(row["page_number"] == page_number for row in semantic_figures):
                overlay = True
                reasons.append("caption-linked semantic figure")
            if any(row["page_number"] == page_number for row in tables):
                overlay = True
                reasons.append("table structure candidate; PDF verification required")
            if page_number in equation_pages:
                overlay = True
                reasons.append("formula region requires targeted enrichment")

            page_routes[page_number] = {
                "page_number": page_number,
                "lexical_source": "docling_native",
                "structure_overlay": "docling_standard_no_ocr" if overlay else None,
                "formula_escalation": (
                    "docling_codeformulav2_targeted"
                    if page_number in equation_pages
                    else None
                ),
                "reasons": reasons,
            }

        rows.append(
            {
                "case_id": case.case_id,
                "source_pdf_sha256": case.source_pdf_sha256,
                "source_page_count": case.page_count,
                "lexical_substrate": {
                    "package_id": native.manifest.package_id,
                    "route": "DOCLING_NATIVE",
                    "status": "ACCEPT",
                },
                "structure_overlay": {
                    "package_id": standard.manifest.package_id,
                    "route": "DOCLING_STANDARD_NO_OCR",
                    "status": "SELECTIVE_OVERLAY",
                },
                "headings": headings,
                "semantic_figures": semantic_figures,
                "unpromoted_pictures": unpromoted_pictures,
                "tables": tables,
                "formula_target_pages": sorted(equation_pages),
                "page_routes": list(page_routes.values()),
            }
        )

    report = {
        "schema_version": 1,
        "policy_id": "native-plus-docling-structure-overlay-v1",
        "authority": "ORIGINAL_PDF",
        "benchmark_oracle_used_for_routing": False,
        "default_lexical_substrate": "docling_native",
        "structure_overlay": "docling_standard_no_ocr",
        "formula_escalation": "docling_codeformulav2_targeted",
        "cases": len(rows),
        "summary": {
            "pages": sum(row["source_page_count"] for row in rows),
            "headings_promoted": sum(len(row["headings"]) for row in rows),
            "semantic_figures_promoted": sum(
                len(row["semantic_figures"]) for row in rows
            ),
            "pictures_unpromoted": sum(
                len(row["unpromoted_pictures"]) for row in rows
            ),
            "tables_structure_candidate": sum(
                table["status"] == "ACCEPT_STRUCTURE_WITH_PDF_VERIFY"
                for row in rows
                for table in row["tables"]
            ),
            "tables_residual_review": sum(
                table["status"] == "RESIDUAL_REVIEW_REQUIRED"
                for row in rows
                for table in row["tables"]
            ),
            "formula_target_pages": sum(
                len(row["formula_target_pages"]) for row in rows
            ),
        },
        "results": rows,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case_manifest")
    parser.add_argument("--native-package-root", required=True)
    parser.add_argument("--standard-package-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_hybrid_routing_report(
        case_manifest_path=args.case_manifest,
        native_package_root=args.native_package_root,
        standard_package_root=args.standard_package_root,
        output_path=args.output,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
