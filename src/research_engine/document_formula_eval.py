from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_benchmark import load_benchmark_cases
from .document_docling import _iter_body_items
from .document_structure_eval import (
    _math_tokens,
    _matched_summary,
    extract_mathpix_equations,
)


def _formula_items(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _iter_body_items(document)
        if str(item.get("label") or "") == "formula"
    ]


def build_formula_target_plan(
    *,
    case_manifest_path: str | Path,
    raw_standard_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive a bounded target set from preserved standard-no-OCR provenance."""

    raw_root = Path(raw_standard_root)
    cases = load_benchmark_cases(case_manifest_path)
    rows: list[dict[str, Any]] = []

    for case in cases:
        raw_json = (
            raw_root
            / case.case_id
            / "docling-standard_no_ocr"
            / (Path(case.source_location).stem + ".json")
        )
        if not raw_json.is_file():
            raise FileNotFoundError(raw_json)
        document = json.loads(raw_json.read_text(encoding="utf-8"))

        regions: list[dict[str, Any]] = []
        pages: set[int] = set()
        for item in _formula_items(document):
            for prov in item.get("prov") or []:
                if not isinstance(prov, dict):
                    continue
                page_number = prov.get("page_no")
                if not isinstance(page_number, int) or page_number < 1:
                    continue
                pages.add(page_number)
                regions.append(
                    {
                        "page_number": page_number,
                        "bbox": prov.get("bbox"),
                        "docling_self_ref": item.get("self_ref"),
                    }
                )

        rows.append(
            {
                "case_id": case.case_id,
                "source_location": case.source_location,
                "source_pdf_sha256": case.source_pdf_sha256,
                "formula_region_count": len(regions),
                "target_pages": sorted(pages),
                "regions": regions,
            }
        )

    plan = {
        "schema_version": 1,
        "selector": "preserved_standard_no_ocr_formula_provenance",
        "broad_document_rerun": False,
        "page_execution": "one_page_per_conversion",
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "formula_cases": sum(bool(row["target_pages"]) for row in rows),
            "formula_regions": sum(row["formula_region_count"] for row in rows),
            "target_pages": sum(len(row["target_pages"]) for row in rows),
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


def _read_enriched_formulas(
    *,
    formula_root: Path,
    case_id: str,
    source_stem: str,
) -> tuple[list[str], list[int]]:
    formulas: list[str] = []
    pages: list[int] = []
    case_root = formula_root / case_id
    if not case_root.exists():
        return formulas, pages
    for page_dir in sorted(case_root.glob("page-*")):
        if not page_dir.is_dir():
            continue
        raw_json = page_dir / f"{source_stem}.json"
        if not raw_json.is_file():
            continue
        document = json.loads(raw_json.read_text(encoding="utf-8"))
        for item in _formula_items(document):
            value = str(item.get("text") or "").strip()
            if value:
                formulas.append(value)
        try:
            pages.append(int(page_dir.name.removeprefix("page-")))
        except ValueError:
            pass
    return formulas, pages


def evaluate_formula_enrichment(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    formula_root: str | Path,
    baseline_summary_path: str | Path | None = None,
    output_path: str | Path | None = None,
    promotion_content_f1: float = 0.8,
    promotion_reference_coverage: float = 0.8,
) -> dict[str, Any]:
    bundle = Path(bundle_root)
    enriched_root = Path(formula_root)
    cases = load_benchmark_cases(case_manifest_path)

    baseline_rows: dict[str, dict[str, Any]] = {}
    if baseline_summary_path is not None:
        baseline = json.loads(
            Path(baseline_summary_path).read_text(encoding="utf-8")
        )
        baseline_rows = {
            str(row["case_id"]): row
            for row in baseline.get("results", [])
            if isinstance(row, dict) and row.get("case_id")
        }

    rows: list[dict[str, Any]] = []
    for case in cases:
        oracle_path = bundle / str(case.mathpix_oracle_location)
        oracle = oracle_path.read_text(encoding="utf-8", errors="replace")
        reference = extract_mathpix_equations(oracle)
        candidate, pages = _read_enriched_formulas(
            formula_root=enriched_root,
            case_id=case.case_id,
            source_stem=Path(case.source_location).stem,
        )
        metrics = _matched_summary(
            candidate,
            reference,
            tokenizer=_math_tokens,
        )

        base_metrics = (
            baseline_rows.get(case.case_id, {})
            .get("content_fidelity", {})
            .get("equations", {})
        )
        rows.append(
            {
                "case_id": case.case_id,
                "target_pages_completed": sorted(set(pages)),
                "candidate_equations": candidate,
                "metrics": metrics,
                "baseline": base_metrics or None,
                "candidate_minus_baseline": {
                    "content_f1_on_matched": (
                        float(metrics["content_f1_on_matched"])
                        - float(base_metrics.get("content_f1_on_matched", 0.0))
                    ),
                    "reference_coverage_at_f1_0_5": (
                        float(metrics["reference_coverage_at_f1_0_5"])
                        - float(
                            base_metrics.get(
                                "reference_coverage_at_f1_0_5",
                                0.0,
                            )
                        )
                    ),
                }
                if base_metrics
                else None,
            }
        )

    total_matches = sum(int(row["metrics"]["matched_units"]) for row in rows)
    total_reference = sum(int(row["metrics"]["reference_units"]) for row in rows)
    weighted_content = (
        sum(
            float(row["metrics"]["content_f1_on_matched"])
            * int(row["metrics"]["matched_units"])
            for row in rows
        )
        / total_matches
        if total_matches
        else 0.0
    )
    weighted_coverage = (
        sum(
            float(row["metrics"]["reference_coverage_at_f1_0_5"])
            * int(row["metrics"]["reference_units"])
            for row in rows
        )
        / total_reference
        if total_reference
        else 1.0
    )
    equation_rows = [
        row for row in rows if int(row["metrics"]["reference_units"]) > 0
    ]
    all_equation_cases_nonzero = all(
        float(row["metrics"]["content_f1_on_matched"]) > 0
        and float(row["metrics"]["reference_coverage_at_f1_0_5"]) > 0
        for row in equation_rows
    )

    promotion_candidate = (
        weighted_content >= promotion_content_f1
        and weighted_coverage >= promotion_reference_coverage
        and all_equation_cases_nonzero
    )
    report = {
        "schema_version": 1,
        "authority": "ORIGINAL_PDF",
        "reference": "MATHPIX_MANUAL_REFERENCE",
        "mode": "docling_codeformulav2_targeted",
        "general_document_vlm_pipeline": False,
        "ocr": False,
        "table_structure": False,
        "promotion_thresholds": {
            "weighted_content_f1_on_matched": promotion_content_f1,
            "weighted_reference_coverage_at_f1_0_5": (
                promotion_reference_coverage
            ),
            "all_equation_cases_nonzero": True,
        },
        "summary": {
            "cases": len(rows),
            "equation_cases": len(equation_rows),
            "candidate_equations": sum(
                int(row["metrics"]["candidate_units"]) for row in rows
            ),
            "reference_equations": total_reference,
            "matched_equations": total_matches,
            "weighted_content_f1_on_matched": weighted_content,
            "weighted_reference_coverage_at_f1_0_5": weighted_coverage,
            "all_equation_cases_nonzero": all_equation_cases_nonzero,
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
    plan.add_argument("--raw-standard-root", required=True)
    plan.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("case_manifest")
    evaluate.add_argument("--bundle-root", required=True)
    evaluate.add_argument("--formula-root", required=True)
    evaluate.add_argument("--baseline-summary")
    evaluate.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        result = build_formula_target_plan(
            case_manifest_path=args.case_manifest,
            raw_standard_root=args.raw_standard_root,
            output_path=args.output,
        )
    else:
        result = evaluate_formula_enrichment(
            case_manifest_path=args.case_manifest,
            bundle_root=args.bundle_root,
            formula_root=args.formula_root,
            baseline_summary_path=args.baseline_summary,
            output_path=args.output,
        )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
