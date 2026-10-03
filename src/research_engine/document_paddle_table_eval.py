from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .document_benchmark import load_benchmark_cases
from .document_structure_eval import _table_fidelity, extract_mathpix_tables


class _FirstTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.done = False
        self.rows: list[list[tuple[str, int, int]]] = []
        self.current_row: list[tuple[str, int, int]] | None = None
        self.current_cell: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if self.done:
            return
        if tag == "table":
            self.table_depth += 1
            return
        if self.table_depth != 1:
            return
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            values = {k.lower(): v for k, v in attrs}
            try:
                rowspan = max(1, int(values.get("rowspan") or 1))
            except ValueError:
                rowspan = 1
            try:
                colspan = max(1, int(values.get("colspan") or 1))
            except ValueError:
                colspan = 1
            self.current_cell = {
                "parts": [],
                "rowspan": rowspan,
                "colspan": colspan,
            }

    def handle_data(self, data: str):
        if self.current_cell is not None and self.table_depth == 1:
            self.current_cell["parts"].append(data)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self.done:
            return
        if tag in {"td", "th"} and self.current_cell is not None:
            text = " ".join(
                " ".join(self.current_cell["parts"]).split()
            ).strip()
            assert self.current_row is not None
            self.current_row.append(
                (
                    text,
                    int(self.current_cell["rowspan"]),
                    int(self.current_cell["colspan"]),
                )
            )
            self.current_cell = None
            return
        if tag == "tr" and self.table_depth == 1:
            if self.current_row is not None:
                self.rows.append(self.current_row)
            self.current_row = None
            return
        if tag == "table":
            if self.table_depth == 1:
                self.table_depth = 0
                self.done = True
            elif self.table_depth > 1:
                self.table_depth -= 1


def html_table_to_grid(html: str) -> list[list[str]]:
    parser = _FirstTableParser()
    parser.feed(html)
    if not parser.rows:
        return []

    grid: list[list[str]] = []
    active_spans: dict[int, int] = {}

    for raw_row in parser.rows:
        row: list[str] = []
        col = 0

        def consume_spans_until_free() -> None:
            nonlocal col
            while active_spans.get(col, 0) > 0:
                while len(row) <= col:
                    row.append("")
                active_spans[col] -= 1
                if active_spans[col] <= 0:
                    del active_spans[col]
                col += 1

        for text, rowspan, colspan in raw_row:
            consume_spans_until_free()
            while len(row) < col + colspan:
                row.append("")
            row[col] = text
            for offset in range(colspan):
                if rowspan > 1:
                    active_spans[col + offset] = max(
                        active_spans.get(col + offset, 0),
                        rowspan - 1,
                    )
            col += colspan

        max_span_col = max(active_spans.keys(), default=-1)
        while col <= max_span_col:
            consume_spans_until_free()
            if col <= max_span_col:
                while len(row) <= col:
                    row.append("")
                col += 1

        grid.append(row)

    width = max((len(row) for row in grid), default=0)
    if width == 0:
        return []
    return [row + [""] * (width - len(row)) for row in grid]


def _find_pred_htmls(value: Any) -> list[str]:
    rows: list[str] = []
    if isinstance(value, dict):
        pred = value.get("pred_html")
        if isinstance(pred, str) and pred.strip():
            rows.append(pred)
        for child in value.values():
            rows.extend(_find_pred_htmls(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_find_pred_htmls(child))
    return rows


def evaluate_paddle_table_fallback(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    target_plan_path: str | Path,
    paddle_output_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    cases = {
        case.case_id: case
        for case in load_benchmark_cases(case_manifest_path)
    }
    bundle = Path(bundle_root)
    plan = json.loads(Path(target_plan_path).read_text(encoding="utf-8"))
    output_root = Path(paddle_output_root)
    runtime_summary = json.loads(
        (output_root / "paddle-table-output-summary.json").read_text(
            encoding="utf-8"
        )
    )
    output_by_key = {
        (row["case_id"], row["table_asset_id"]): row
        for row in runtime_summary.get("outputs", [])
    }

    rows: list[dict[str, Any]] = []
    for target in plan.get("targets", []):
        case_id = str(target["case_id"])
        asset_id = str(target["table_asset_id"])
        key = (case_id, asset_id)
        if key not in output_by_key:
            raise ValueError(f"missing Paddle output for {case_id}/{asset_id}")

        runtime_row = output_by_key[key]
        payload = json.loads(
            (output_root / runtime_row["result_json"]).read_text(
                encoding="utf-8"
            )
        )
        htmls = _find_pred_htmls(payload)
        candidate_grids = [
            grid
            for grid in (html_table_to_grid(html) for html in htmls)
            if grid
        ]

        case = cases[case_id]
        oracle = (
            bundle / str(case.mathpix_oracle_location)
        ).read_text(encoding="utf-8", errors="replace")
        references = extract_mathpix_tables(oracle)
        reference_index = int(target["reference_index"])
        reference_grid = references[reference_index]

        fidelity = _table_fidelity(candidate_grids, [reference_grid])
        rows.append(
            {
                "case_id": case_id,
                "table_asset_id": asset_id,
                "page_number": int(target["page_number"]),
                "reference_index": reference_index,
                "candidate_html_count": len(htmls),
                "candidate_table_count": len(candidate_grids),
                "candidate_grids": candidate_grids,
                "reference_grid": reference_grid,
                "fidelity": fidelity,
            }
        )

    target_tables = len(rows)
    matched = sum(int(row["fidelity"]["matched_units"]) for row in rows)
    content_weighted = (
        sum(
            float(row["fidelity"]["content_f1_on_matched"])
            * int(row["fidelity"]["matched_units"])
            for row in rows
        )
        / matched
        if matched
        else 0.0
    )
    aligned_weighted = (
        sum(
            float(row["fidelity"]["aligned_cell_f1_on_matched"])
            * int(row["fidelity"]["matched_units"])
            for row in rows
        )
        / matched
        if matched
        else 0.0
    )
    coverage = (
        sum(
            float(
                row["fidelity"]["reference_coverage_at_content_f1_0_5"]
            )
            for row in rows
        )
        / target_tables
        if target_tables
        else 1.0
    )
    all_targets_materialized = all(
        row["candidate_table_count"] >= 1 for row in rows
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
        "mode": "paddle_table_recognition_v2_crop",
        "broad_document_rerun": False,
        "promotion_thresholds": {
            "weighted_content_f1_on_matched_min": 0.75,
            "weighted_aligned_cell_f1_on_matched_min": 0.50,
            "reference_coverage_at_content_f1_0_5_min": 1.0,
            "all_targets_materialized": True,
        },
        "summary": {
            "target_tables": target_tables,
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
    parser.add_argument("case_manifest")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--target-plan", required=True)
    parser.add_argument("--paddle-output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_paddle_table_fallback(
        case_manifest_path=args.case_manifest,
        bundle_root=args.bundle_root,
        target_plan_path=args.target_plan,
        paddle_output_root=args.paddle_output_root,
        output_path=args.output,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
