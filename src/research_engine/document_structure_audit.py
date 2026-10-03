from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .document_benchmark import _text_tokens, load_benchmark_cases
from .document_package import PaperPackage
from .document_structure_eval import (
    _candidate_table_grids,
    _table_fidelity,
    extract_mathpix_tables,
)


def _grid_tokens(grid: list[list[str]]) -> list[str]:
    return _text_tokens(" ".join(cell for row in grid for cell in row))


def audit_table_residuals(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    standard_package_root: str | Path,
    output_path: str | Path | None = None,
    content_threshold: float = 0.5,
    aligned_cell_threshold: float = 0.5,
    shape_threshold: float = 0.5,
) -> dict[str, Any]:
    """Expose low-scoring table pairs without rerunning document inference."""

    bundle = Path(bundle_root)
    package_root = Path(standard_package_root)
    cases = load_benchmark_cases(case_manifest_path)
    results: list[dict[str, Any]] = []

    for case in cases:
        case_root = package_root / case.case_id / "paper-package-v1"
        package = PaperPackage.model_validate_json(
            (case_root / "package.json").read_text(encoding="utf-8")
        )
        oracle_path = bundle / str(case.mathpix_oracle_location)
        oracle = oracle_path.read_text(encoding="utf-8", errors="replace")

        candidate = _candidate_table_grids(package, case_root)
        reference = extract_mathpix_tables(oracle)
        fidelity = _table_fidelity(candidate, reference)

        residuals: list[dict[str, Any]] = []
        matched_candidate: set[int] = set()
        matched_reference: set[int] = set()
        for match in fidelity["matches"]:
            ci = int(match["candidate_index"])
            ri = int(match["reference_index"])
            matched_candidate.add(ci)
            matched_reference.add(ri)
            failed = (
                float(match["content_f1"]) < content_threshold
                or float(match["aligned_cell_f1"]) < aligned_cell_threshold
                or float(match["shape_similarity"]) < shape_threshold
            )
            if not failed:
                continue
            residuals.append(
                {
                    "kind": "LOW_FIDELITY_MATCH",
                    **match,
                    "candidate_grid": candidate[ci],
                    "reference_grid": reference[ri],
                    "candidate_tokens": _grid_tokens(candidate[ci]),
                    "reference_tokens": _grid_tokens(reference[ri]),
                    "flags": {
                        "content_below_threshold": (
                            float(match["content_f1"]) < content_threshold
                        ),
                        "aligned_cells_below_threshold": (
                            float(match["aligned_cell_f1"])
                            < aligned_cell_threshold
                        ),
                        "shape_below_threshold": (
                            float(match["shape_similarity"]) < shape_threshold
                        ),
                    },
                }
            )

        for ci in sorted(set(range(len(candidate))) - matched_candidate):
            residuals.append(
                {
                    "kind": "UNMATCHED_CANDIDATE",
                    "candidate_index": ci,
                    "candidate_grid": candidate[ci],
                    "candidate_tokens": _grid_tokens(candidate[ci]),
                }
            )
        for ri in sorted(set(range(len(reference))) - matched_reference):
            residuals.append(
                {
                    "kind": "UNMATCHED_REFERENCE",
                    "reference_index": ri,
                    "reference_grid": reference[ri],
                    "reference_tokens": _grid_tokens(reference[ri]),
                }
            )

        results.append(
            {
                "case_id": case.case_id,
                "candidate_tables": len(candidate),
                "reference_tables": len(reference),
                "fidelity": {
                    key: value
                    for key, value in fidelity.items()
                    if key != "matches"
                },
                "residual_count": len(residuals),
                "residuals": residuals,
            }
        )

    report = {
        "schema_version": 1,
        "authority": "ORIGINAL_PDF",
        "reference": "MATHPIX_MANUAL_REFERENCE",
        "purpose": "diagnostic_only_not_production_routing",
        "thresholds": {
            "content_f1": content_threshold,
            "aligned_cell_f1": aligned_cell_threshold,
            "shape_similarity": shape_threshold,
        },
        "summary": {
            "cases": len(results),
            "candidate_tables": sum(row["candidate_tables"] for row in results),
            "reference_tables": sum(row["reference_tables"] for row in results),
            "residuals": sum(row["residual_count"] for row in results),
            "cases_with_residuals": [
                row["case_id"] for row in results if row["residual_count"]
            ],
        },
        "results": results,
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
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--standard-package-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = audit_table_residuals(
        case_manifest_path=args.case_manifest,
        bundle_root=args.bundle_root,
        standard_package_root=args.standard_package_root,
        output_path=args.output,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
