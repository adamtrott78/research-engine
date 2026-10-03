from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_benchmark import load_benchmark_cases
from .document_docling import _table_grid
from .document_package import PaperPackage
from .document_structure_eval import _table_fidelity, extract_mathpix_tables


def _load_package(root: Path, case_id: str) -> tuple[PaperPackage, Path]:
    package_root = root / case_id / "paper-package-v1"
    package_path = package_root / "package.json"
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    return (
        PaperPackage.model_validate_json(package_path.read_text(encoding="utf-8")),
        package_root,
    )


def _asset_grid(package_root: Path, asset: Any) -> list[list[str]]:
    if not asset.json_path:
        return []
    payload = json.loads((package_root / asset.json_path).read_text(encoding="utf-8"))
    return _table_grid(payload)


def build_tableformer_accurate_target_plan(
    *,
    case_manifest_path: str | Path,
    standard_package_root: str | Path,
    table_audit_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select only genuine empty-grid TableFormer residuals.

    The prior diagnostic audit intentionally over-reports representational
    differences. This plan opens a heavier TableFormer mode only when the
    standard FAST package contains a table asset whose structured grid is
    actually empty.
    """

    cases = {case.case_id: case for case in load_benchmark_cases(case_manifest_path)}
    standard_root = Path(standard_package_root)
    audit = json.loads(Path(table_audit_path).read_text(encoding="utf-8"))
    targets: list[dict[str, Any]] = []

    for row in audit.get("results", []):
        case_id = str(row.get("case_id") or "")
        if case_id not in cases:
            raise ValueError(f"audit case not present in manifest: {case_id}")
        package, package_root = _load_package(standard_root, case_id)

        for residual in row.get("residuals", []):
            if residual.get("kind") != "LOW_FIDELITY_MATCH":
                continue
            if residual.get("candidate_shape") != [0, 0]:
                continue

            candidate_index = int(residual["candidate_index"])
            reference_index = int(residual["reference_index"])
            if candidate_index < 0 or candidate_index >= len(package.tables):
                raise IndexError(
                    f"{case_id}: candidate table index {candidate_index} out of range"
                )
            asset = package.tables[candidate_index]
            if _asset_grid(package_root, asset):
                raise ValueError(
                    f"{case_id}: residual claims empty candidate but package grid is non-empty"
                )

            targets.append(
                {
                    "case_id": case_id,
                    "source_location": cases[case_id].source_location,
                    "source_pdf_sha256": cases[case_id].source_pdf_sha256,
                    "page_number": asset.source_region.page_number,
                    "table_asset_id": asset.id,
                    "baseline_candidate_index": candidate_index,
                    "reference_index": reference_index,
                    "baseline_candidate_shape": residual.get("candidate_shape"),
                    "reference_shape": residual.get("reference_shape"),
                    "baseline_content_f1": residual.get("content_f1"),
                    "baseline_aligned_cell_f1": residual.get("aligned_cell_f1"),
                }
            )

    page_keys = sorted({(x["case_id"], x["page_number"]) for x in targets})
    plan = {
        "schema_version": 1,
        "selector": "standard_fast_empty_table_grid_residual_only",
        "broad_document_rerun": False,
        "tableformer_mode": "accurate",
        "targets": targets,
        "summary": {
            "target_tables": len(targets),
            "target_pages": len(page_keys),
            "target_cases": sorted({x["case_id"] for x in targets}),
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return plan


def _accurate_page_tables(
    *,
    accurate_root: Path,
    case_id: str,
    page_number: int,
    source_stem: str,
) -> list[list[list[str]]]:
    raw = (
        accurate_root
        / case_id
        / f"page-{page_number:04d}"
        / f"{source_stem}.json"
    )
    if not raw.is_file():
        raise FileNotFoundError(raw)
    document = json.loads(raw.read_text(encoding="utf-8"))
    grids: list[list[list[str]]] = []
    for item in document.get("tables") or []:
        if isinstance(item, dict):
            grid = _table_grid(item)
            if grid:
                grids.append(grid)
    return grids


def evaluate_tableformer_accurate(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    target_plan_path: str | Path,
    accurate_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    cases = {case.case_id: case for case in load_benchmark_cases(case_manifest_path)}
    bundle = Path(bundle_root)
    target_plan = json.loads(Path(target_plan_path).read_text(encoding="utf-8"))
    accurate = Path(accurate_root)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for target in target_plan.get("targets", []):
        key = (str(target["case_id"]), int(target["page_number"]))
        grouped.setdefault(key, []).append(target)

    rows: list[dict[str, Any]] = []
    for (case_id, page_number), targets in sorted(grouped.items()):
        case = cases[case_id]
        oracle = (bundle / str(case.mathpix_oracle_location)).read_text(
            encoding="utf-8",
            errors="replace",
        )
        all_reference = extract_mathpix_tables(oracle)
        reference_indices = [int(x["reference_index"]) for x in targets]
        reference_grids = [all_reference[i] for i in reference_indices]
        candidate_grids = _accurate_page_tables(
            accurate_root=accurate,
            case_id=case_id,
            page_number=page_number,
            source_stem=Path(case.source_location).stem,
        )
        fidelity = _table_fidelity(candidate_grids, reference_grids)
        rows.append(
            {
                "case_id": case_id,
                "page_number": page_number,
                "target_table_count": len(targets),
                "reference_indices": reference_indices,
                "candidate_table_count": len(candidate_grids),
                "candidate_grids": candidate_grids,
                "reference_grids": reference_grids,
                "fidelity": fidelity,
            }
        )

    target_tables = sum(int(x["target_table_count"]) for x in rows)
    matched = sum(int(x["fidelity"]["matched_units"]) for x in rows)
    content_weighted = (
        sum(
            float(x["fidelity"]["content_f1_on_matched"])
            * int(x["fidelity"]["matched_units"])
            for x in rows
        )
        / matched
        if matched
        else 0.0
    )
    aligned_weighted = (
        sum(
            float(x["fidelity"]["aligned_cell_f1_on_matched"])
            * int(x["fidelity"]["matched_units"])
            for x in rows
        )
        / matched
        if matched
        else 0.0
    )
    references = sum(int(x["fidelity"]["reference_units"]) for x in rows)
    covered = sum(
        float(x["fidelity"]["reference_coverage_at_content_f1_0_5"])
        * int(x["fidelity"]["reference_units"])
        for x in rows
    )
    coverage = covered / references if references else 1.0
    all_targets_materialized = all(
        int(x["candidate_table_count"]) >= int(x["target_table_count"])
        for x in rows
    )
    promotion_candidate = (
        all_targets_materialized
        and content_weighted >= 0.75
        and aligned_weighted >= 0.50
        and coverage >= 1.0
    )

    report = {
        "schema_version": 1,
        "authority": "ORIGINAL_PDF",
        "reference": "MATHPIX_MANUAL_REFERENCE",
        "mode": "docling_tableformer_accurate_targeted",
        "broad_document_rerun": False,
        "promotion_thresholds": {
            "weighted_content_f1_on_matched_min": 0.75,
            "weighted_aligned_cell_f1_on_matched_min": 0.50,
            "reference_coverage_at_content_f1_0_5_min": 1.0,
            "all_targets_materialized": True,
        },
        "summary": {
            "target_tables": target_tables,
            "target_pages": len(rows),
            "matched_tables": matched,
            "weighted_content_f1_on_matched": content_weighted,
            "weighted_aligned_cell_f1_on_matched": aligned_weighted,
            "reference_coverage_at_content_f1_0_5": coverage,
            "all_targets_materialized": all_targets_materialized,
            "decision": (
                "PROMOTION_CANDIDATE"
                if promotion_candidate
                else "MEASURED_RESIDUAL"
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
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("case_manifest")
    plan.add_argument("--standard-package-root", required=True)
    plan.add_argument("--table-audit", required=True)
    plan.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("case_manifest")
    evaluate.add_argument("--bundle-root", required=True)
    evaluate.add_argument("--target-plan", required=True)
    evaluate.add_argument("--accurate-root", required=True)
    evaluate.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        result = build_tableformer_accurate_target_plan(
            case_manifest_path=args.case_manifest,
            standard_package_root=args.standard_package_root,
            table_audit_path=args.table_audit,
            output_path=args.output,
        )
    else:
        result = evaluate_tableformer_accurate(
            case_manifest_path=args.case_manifest,
            bundle_root=args.bundle_root,
            target_plan_path=args.target_plan,
            accurate_root=args.accurate_root,
            output_path=args.output,
        )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
