from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .document_benchmark import _multiset_f1, _text_tokens
from .document_docling import _table_grid
from .document_package import BlockKind, PaperPackage


def _score(left: Iterable[Any], right: Iterable[Any]) -> float:
    value = _multiset_f1(left, right)
    return float(value if value is not None else 0.0)


def _plainish_latex(value: str) -> str:
    text = value
    for _ in range(3):
        updated = re.sub(
            r"\\(?:textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
        if updated == text:
            break
        text = updated
    text = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", " ", text)
    text = re.sub(r"\\(?:cline|cmidrule)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:label)\s*\{[^{}]*\}", " ", text)
    text = text.replace("\\&", "&")
    text = re.sub(r"[{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _braced_command_values(markdown: str, command: str) -> list[str]:
    pattern = re.compile(rf"\\{re.escape(command)}\*?\s*\{{")
    values: list[str] = []
    for match in pattern.finditer(markdown):
        start = match.end() - 1
        depth = 0
        end: int | None = None
        for index in range(start, len(markdown)):
            char = markdown[index]
            if char == "{" and (index == 0 or markdown[index - 1] != "\\"):
                depth += 1
            elif char == "}" and (index == 0 or markdown[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is not None:
            value = _plainish_latex(markdown[start + 1 : end])
            if value:
                values.append(value)
    return values


def extract_mathpix_headings(markdown: str) -> list[str]:
    rows: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            value = _plainish_latex(match.group(1))
            if value:
                rows.append(value)
    for command in ("section", "subsection", "subsubsection", "paragraph"):
        rows.extend(_braced_command_values(markdown, command))
    return rows


def extract_mathpix_equations(markdown: str) -> list[str]:
    rows = [
        value.strip()
        for value in re.findall(r"\\\[(.*?)\\\]", markdown, flags=re.DOTALL)
        if value.strip()
    ]
    for environment in ("equation", "equation*", "align", "align*"):
        escaped = re.escape(environment)
        rows.extend(
            value.strip()
            for value in re.findall(
                rf"\\begin\{{{escaped}\}}(.*?)\\end\{{{escaped}\}}",
                markdown,
                flags=re.DOTALL,
            )
            if value.strip()
        )
    return rows


def _tabular_bodies(markdown: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(
            r"\\begin\{tabular\*?\}(?:\{[^}]*\})?(.*?)\\end\{tabular\*?\}",
            markdown,
            flags=re.DOTALL,
        )
    ]


def _tabular_grid(body: str) -> list[list[str]]:
    body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", " ", body)
    body = re.sub(r"\\(?:cline|cmidrule)\s*\{[^{}]*\}", " ", body)
    rows: list[list[str]] = []
    for raw_row in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        cells = [_plainish_latex(cell) for cell in raw_row.split("&")]
        if any(cells):
            rows.append(cells)
    return rows


def extract_mathpix_tables(markdown: str) -> list[list[list[str]]]:
    outer = [
        match.group(1)
        for match in re.finditer(
            r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}",
            markdown,
            flags=re.DOTALL,
        )
    ]
    units: list[list[list[str]]] = []
    if outer:
        for body in outer:
            combined: list[list[str]] = []
            for tabular in _tabular_bodies(body):
                combined.extend(_tabular_grid(tabular))
            if combined:
                units.append(combined)
        return units
    return [
        grid
        for grid in (_tabular_grid(body) for body in _tabular_bodies(markdown))
        if grid
    ]


def extract_mathpix_figure_captions(markdown: str) -> list[str]:
    captions: list[str] = []
    figures = [
        match.group(1)
        for match in re.finditer(
            r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",
            markdown,
            flags=re.DOTALL,
        )
    ]
    for figure in figures:
        values = _braced_command_values(figure, "caption")
        if values:
            captions.append(" ".join(values))
            continue
        match = re.search(
            r"(?im)^\s*(?:Figure|Fig\.)\s*\d+[.:]?\s*(.+?)\s*$",
            figure,
        )
        if match:
            captions.append(_plainish_latex(match.group(0)))
    return captions


def _math_tokens(value: str) -> list[str]:
    return re.findall(
        r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?|[=+\-*/^_<>≤≥≈∑∫√]+",
        value.lower(),
    )


def _grid_tokens(grid: list[list[str]]) -> list[str]:
    return _text_tokens(" ".join(cell for row in grid for cell in row))


def _grid_shape(grid: list[list[str]]) -> tuple[int, int]:
    return len(grid), max((len(row) for row in grid), default=0)


def _shape_similarity(left: list[list[str]], right: list[list[str]]) -> float:
    l_rows, l_cols = _grid_shape(left)
    r_rows, r_cols = _grid_shape(right)
    if (l_rows, l_cols) == (0, 0) and (r_rows, r_cols) == (0, 0):
        return 1.0
    if not l_rows or not l_cols or not r_rows or not r_cols:
        return 0.0
    return (min(l_rows, r_rows) / max(l_rows, r_rows)) * (
        min(l_cols, r_cols) / max(l_cols, r_cols)
    )


def _aligned_cell_f1(left: list[list[str]], right: list[list[str]]) -> float:
    l_rows, l_cols = _grid_shape(left)
    r_rows, r_cols = _grid_shape(right)
    rows = max(l_rows, r_rows)
    cols = max(l_cols, r_cols)
    if rows == 0 or cols == 0:
        return 1.0 if rows == 0 and cols == 0 else 0.0
    scores: list[float] = []
    for row in range(rows):
        for col in range(cols):
            l_value = left[row][col] if row < l_rows and col < len(left[row]) else ""
            r_value = right[row][col] if row < r_rows and col < len(right[row]) else ""
            scores.append(_score(_text_tokens(l_value), _text_tokens(r_value)))
    return sum(scores) / len(scores)


def _greedy_match(
    left: list[Any],
    right: list[Any],
    scorer: Callable[[Any, Any], float],
) -> list[tuple[int, int, float]]:
    ranked = sorted(
        (
            (scorer(l_value, r_value), l_index, r_index)
            for l_index, l_value in enumerate(left)
            for r_index, r_value in enumerate(right)
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, l_index, r_index in ranked:
        if l_index in used_left or r_index in used_right:
            continue
        used_left.add(l_index)
        used_right.add(r_index)
        matches.append((l_index, r_index, float(score)))
        if len(used_left) == len(left) or len(used_right) == len(right):
            break
    return matches


def _matched_summary(
    left: list[str],
    right: list[str],
    *,
    tokenizer: Callable[[str], list[str]] = _text_tokens,
) -> dict[str, Any]:
    matches = _greedy_match(
        left,
        right,
        lambda a, b: _score(tokenizer(a), tokenizer(b)),
    )
    return {
        "candidate_units": len(left),
        "reference_units": len(right),
        "matched_units": len(matches),
        "content_f1_on_matched": (
            sum(score for _, _, score in matches) / len(matches)
            if matches
            else (1.0 if not left and not right else 0.0)
        ),
        "reference_coverage_at_f1_0_5": (
            sum(score >= 0.5 for _, _, score in matches) / len(right)
            if right
            else 1.0
        ),
        "matches": [
            {
                "candidate_index": left_index,
                "reference_index": right_index,
                "content_f1": score,
            }
            for left_index, right_index, score in matches
        ],
    }


def _candidate_figure_captions(package: PaperPackage) -> tuple[int, list[str]]:
    figure_blocks = [block for block in package.blocks if block.kind == BlockKind.FIGURE]
    captions: list[str] = []
    for block in figure_blocks:
        for step in block.derivation:
            value = step.parameters.get("caption_text")
            if isinstance(value, str) and value.strip():
                captions.append(value.strip())
                break
    return len(figure_blocks), captions


def _candidate_table_grids(
    package: PaperPackage,
    package_root: Path,
) -> list[list[list[str]]]:
    grids: list[list[list[str]]] = []
    for table in package.tables:
        if not table.json_path:
            grids.append([])
            continue
        item = json.loads((package_root / table.json_path).read_text(encoding="utf-8"))
        grids.append(_table_grid(item))
    return grids


def _table_fidelity(
    candidate: list[list[list[str]]],
    reference: list[list[list[str]]],
) -> dict[str, Any]:
    matches = _greedy_match(
        candidate,
        reference,
        lambda a, b: _score(_grid_tokens(a), _grid_tokens(b)),
    )
    detailed = []
    for left_index, right_index, content_f1 in matches:
        left = candidate[left_index]
        right = reference[right_index]
        detailed.append(
            {
                "candidate_index": left_index,
                "reference_index": right_index,
                "content_f1": content_f1,
                "shape_similarity": _shape_similarity(left, right),
                "aligned_cell_f1": _aligned_cell_f1(left, right),
                "candidate_shape": list(_grid_shape(left)),
                "reference_shape": list(_grid_shape(right)),
            }
        )
    return {
        "candidate_units": len(candidate),
        "reference_units": len(reference),
        "matched_units": len(matches),
        "content_f1_on_matched": (
            sum(row["content_f1"] for row in detailed) / len(detailed)
            if detailed
            else (1.0 if not candidate and not reference else 0.0)
        ),
        "shape_similarity_on_matched": (
            sum(row["shape_similarity"] for row in detailed) / len(detailed)
            if detailed
            else (1.0 if not candidate and not reference else 0.0)
        ),
        "aligned_cell_f1_on_matched": (
            sum(row["aligned_cell_f1"] for row in detailed) / len(detailed)
            if detailed
            else (1.0 if not candidate and not reference else 0.0)
        ),
        "reference_coverage_at_content_f1_0_5": (
            sum(row["content_f1"] >= 0.5 for row in detailed) / len(reference)
            if reference
            else 1.0
        ),
        "matches": detailed,
    }


def _provenance_integrity(package: PaperPackage) -> dict[str, Any]:
    blocks = list(package.blocks)
    assets = [*package.tables, *package.figures, *package.equations]
    max_page = package.manifest.source_page_count

    def valid_page(region: Any) -> bool:
        return 1 <= region.page_number <= max_page

    heading_blocks = [block for block in blocks if block.kind == BlockKind.HEADING]
    heading_path_ok = [
        bool(block.source_region.section_path)
        and block.source_region.section_path[-1].strip() == block.markdown.strip()
        for block in heading_blocks
    ]
    return {
        "block_page_validity": (
            sum(valid_page(block.source_region) for block in blocks) / len(blocks)
            if blocks
            else 1.0
        ),
        "block_bbox_coverage": (
            sum(block.source_region.pdf_bbox is not None for block in blocks) / len(blocks)
            if blocks
            else 1.0
        ),
        "asset_page_validity": (
            sum(valid_page(asset.source_region) for asset in assets) / len(assets)
            if assets
            else 1.0
        ),
        "asset_bbox_coverage": (
            sum(asset.source_region.pdf_bbox is not None for asset in assets) / len(assets)
            if assets
            else 1.0
        ),
        "heading_section_path_self_consistency": (
            sum(heading_path_ok) / len(heading_path_ok) if heading_path_ok else 1.0
        ),
    }


def evaluate_paper_package_structure(
    *,
    package: PaperPackage,
    package_root: str | Path,
    mathpix_markdown: str,
) -> dict[str, Any]:
    root = Path(package_root)
    candidate_tables = _candidate_table_grids(package, root)
    reference_tables = extract_mathpix_tables(mathpix_markdown)

    candidate_equations = [equation.latex for equation in package.equations]
    reference_equations = extract_mathpix_equations(mathpix_markdown)

    figure_count, candidate_captions = _candidate_figure_captions(package)
    reference_captions = extract_mathpix_figure_captions(mathpix_markdown)

    candidate_headings = [
        block.markdown.strip()
        for block in package.blocks
        if block.kind in {BlockKind.TITLE, BlockKind.HEADING}
        and block.markdown.strip()
    ]
    reference_headings = extract_mathpix_headings(mathpix_markdown)

    figure_caption = _matched_summary(candidate_captions, reference_captions)
    figure_caption.update(
        {
            "candidate_figure_blocks": figure_count,
            "candidate_captioned_figures": len(candidate_captions),
            "candidate_caption_association_rate": (
                len(candidate_captions) / figure_count if figure_count else 1.0
            ),
        }
    )

    return {
        "reference": "MATHPIX_MANUAL_REFERENCE",
        "authority": "ORIGINAL_PDF",
        "count_agreement_excluded": True,
        "tables": _table_fidelity(candidate_tables, reference_tables),
        "equations": _matched_summary(
            candidate_equations,
            reference_equations,
            tokenizer=_math_tokens,
        ),
        "figure_captions": figure_caption,
        "headings": _matched_summary(candidate_headings, reference_headings),
        "provenance_integrity": _provenance_integrity(package),
    }
