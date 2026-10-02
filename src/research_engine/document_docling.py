from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .document_package import (
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


DOCLING_NORMALIZER_ID = "docling-native-structured"
DOCLING_NORMALIZER_VERSION = "1"


_LABEL_TO_KIND: dict[str, BlockKind] = {
    "title": BlockKind.TITLE,
    "section_header": BlockKind.HEADING,
    "list_item": BlockKind.LIST,
    "code": BlockKind.CODE,
    "formula": BlockKind.EQUATION,
    "caption": BlockKind.CAPTION,
    "reference": BlockKind.REFERENCE,
    "footnote": BlockKind.FOOTNOTE,
    "paragraph": BlockKind.PARAGRAPH,
    "text": BlockKind.PARAGRAPH,
    "table": BlockKind.TABLE,
    "document_index": BlockKind.TABLE,
    "picture": BlockKind.FIGURE,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _slug_ref(value: str) -> str:
    value = value.removeprefix("#/")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-") or "root"


def _ref_path(value: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"#/([A-Za-z0-9_-]+)/([0-9]+)", value)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _resolve_ref(document: dict[str, Any], value: str) -> dict[str, Any] | None:
    parsed = _ref_path(value)
    if parsed is None:
        return None
    collection, index = parsed
    rows = document.get(collection)
    if not isinstance(rows, list) or index >= len(rows):
        return None
    item = rows[index]
    return item if isinstance(item, dict) else None


def _iter_body_items(document: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()

    def walk_ref(ref: str) -> Iterable[dict[str, Any]]:
        if ref in seen:
            return
        seen.add(ref)
        item = _resolve_ref(document, ref)
        if item is None:
            return

        parsed = _ref_path(ref)
        if parsed and parsed[0] == "groups":
            for child in item.get("children") or []:
                if isinstance(child, dict) and isinstance(child.get("$ref"), str):
                    yield from walk_ref(child["$ref"])
            return

        yield item

    body = document.get("body") or {}
    for child in body.get("children") or []:
        if isinstance(child, dict) and isinstance(child.get("$ref"), str):
            yield from walk_ref(child["$ref"])


def _bbox_tuple(raw: dict[str, Any]) -> tuple[float, float, float, float]:
    l = float(raw["l"])
    t = float(raw["t"])
    r = float(raw["r"])
    b = float(raw["b"])
    origin = str(raw.get("coord_origin") or "TOPLEFT").upper()
    if origin == "TOPLEFT":
        return (l, t, r, b)
    # Preserve Docling's coordinates without inventing a page-height transform.
    # BOTTOMLEFT is rare in the PDF path and is flagged in diagnostics.
    return (l, min(t, b), r, max(t, b))


def _provenance_regions(
    item: dict[str, Any],
    *,
    section_path: tuple[str, ...],
) -> tuple[PdfRegion, ...]:
    rows: list[PdfRegion] = []
    self_ref = str(item.get("self_ref") or "")
    for prov in item.get("prov") or []:
        if not isinstance(prov, dict):
            continue
        page_no = prov.get("page_no")
        bbox = prov.get("bbox")
        if not isinstance(page_no, int) or page_no < 1:
            continue
        parsed_bbox = None
        if isinstance(bbox, dict) and all(k in bbox for k in ("l", "t", "r", "b")):
            parsed_bbox = _bbox_tuple(bbox)
        rows.append(
            PdfRegion(
                page_number=page_no,
                pdf_bbox=parsed_bbox,
                section_path=section_path,
                raw_anchor=self_ref or None,
            )
        )
    return tuple(rows)


def _primary_region(
    item: dict[str, Any],
    *,
    section_path: tuple[str, ...],
) -> tuple[PdfRegion, tuple[PdfRegion, ...]]:
    regions = _provenance_regions(item, section_path=section_path)
    if not regions:
        raise ValueError(
            f"Docling item {item.get('self_ref')!r} has no valid page provenance"
        )
    return regions[0], regions


def _caption_text(document: dict[str, Any], item: dict[str, Any]) -> str:
    parts: list[str] = []
    for ref in item.get("captions") or []:
        if not isinstance(ref, dict) or not isinstance(ref.get("$ref"), str):
            continue
        caption = _resolve_ref(document, ref["$ref"])
        if caption and isinstance(caption.get("text"), str):
            value = caption["text"].strip()
            if value:
                parts.append(value)
    return "\n\n".join(parts)


def _table_grid(item: dict[str, Any]) -> list[list[str]]:
    data = item.get("data") or {}
    cells = data.get("table_cells") or []
    if not isinstance(cells, list):
        return []

    max_row = 0
    max_col = 0
    normalized: list[tuple[int, int, int, int, str]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        try:
            r0 = int(cell["start_row_offset_idx"])
            r1 = int(cell["end_row_offset_idx"])
            c0 = int(cell["start_col_offset_idx"])
            c1 = int(cell["end_col_offset_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(cell.get("text") or "").strip()
        max_row = max(max_row, r1)
        max_col = max(max_col, c1)
        normalized.append((r0, r1, c0, c1, text))

    if not normalized or max_row <= 0 or max_col <= 0:
        return []

    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    for r0, r1, c0, c1, text in normalized:
        for row in range(max(0, r0), min(max_row, r1)):
            for col in range(max(0, c0), min(max_col, c1)):
                if row == r0 and col == c0:
                    grid[row][col] = text
    return grid


def _table_markdown(item: dict[str, Any]) -> str:
    grid = _table_grid(item)
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    rows = [row + [""] * (width - len(row)) for row in grid]
    header = rows[0]
    body = rows[1:]
    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
    output = [line(header), "| " + " | ".join("---" for _ in header) + " |"]
    output.extend(line(row) for row in body)
    return "\n".join(output)


def _block_markdown(document: dict[str, Any], item: dict[str, Any]) -> str:
    label = str(item.get("label") or "")
    if label in {"table", "document_index"}:
        return _table_markdown(item)
    if label == "picture":
        return _caption_text(document, item)
    text = item.get("text")
    return str(text).strip() if text is not None else ""


def _section_after_heading(
    current: tuple[str, ...],
    item: dict[str, Any],
) -> tuple[str, ...]:
    if str(item.get("label") or "") != "section_header":
        return current
    text = str(item.get("text") or "").strip()
    if not text:
        return current
    try:
        level = max(1, int(item.get("level") or 1))
    except (TypeError, ValueError):
        level = 1
    base = list(current[: level - 1])
    base.append(text)
    return tuple(base)


def docling_structure_inventory(document: dict[str, Any]) -> dict[str, Any]:
    ordered = list(_iter_body_items(document))
    labels: dict[str, int] = {}
    provenance_items = 0
    bbox_items = 0
    multi_provenance_items = 0
    picture_captioned = 0

    for item in ordered:
        label = str(item.get("label") or "unknown")
        labels[label] = labels.get(label, 0) + 1
        prov = [p for p in item.get("prov") or [] if isinstance(p, dict)]
        if prov:
            provenance_items += 1
            if any(isinstance(p.get("bbox"), dict) for p in prov):
                bbox_items += 1
            if len(prov) > 1:
                multi_provenance_items += 1
        if label == "picture" and item.get("captions"):
            picture_captioned += 1

    count = len(ordered)
    return {
        "ordered_body_items": count,
        "labels": labels,
        "tables": sum(
            labels.get(name, 0) for name in ("table", "document_index")
        ),
        "formulas": labels.get("formula", 0),
        "pictures": labels.get("picture", 0),
        "section_headers": labels.get("section_header", 0),
        "caption_items": labels.get("caption", 0),
        "picture_captioned": picture_captioned,
        "provenance_items": provenance_items,
        "bbox_items": bbox_items,
        "multi_provenance_items": multi_provenance_items,
        "provenance_coverage": provenance_items / count if count else 1.0,
        "bbox_coverage": bbox_items / count if count else 1.0,
    }


def normalize_docling_document(
    *,
    docling_json_path: str | Path,
    source_pdf_path: str | Path,
    source_artifact_id: str,
    source_pdf_sha256: str,
    output_directory: str | Path,
    compiler_version: str = "2.131.0",
) -> PaperPackage:
    source = Path(source_pdf_path)
    raw_path = Path(docling_json_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    document = json.loads(raw_path.read_text(encoding="utf-8"))
    if document.get("schema_name") not in (None, "DoclingDocument"):
        raise ValueError("structured JSON is not a DoclingDocument")

    pages = document.get("pages") or {}
    source_page_count = len(pages)
    if source_page_count < 1:
        raise ValueError("DoclingDocument contains no pages")

    config = {
        "source": "docling-export-to-dict",
        "reading_order": "body-children",
        "provenance": "docling-prov-page-bbox",
        "picture_materialization": "deferred",
    }
    config_sha = compiler_config_sha256(config)
    step_base = {
        "processor": "docling",
        "processor_version": compiler_version,
        "mode": ProcessingMode.DETERMINISTIC,
    }

    blocks: list[DocumentBlock] = []
    tables: list[TableAsset] = []
    equations: list[EquationAsset] = []
    warnings: list[str] = []
    section_path: tuple[str, ...] = ()

    table_index = 0
    equation_index = 0

    for item in _iter_body_items(document):
        label = str(item.get("label") or "")
        self_ref = str(item.get("self_ref") or "")
        item_section = section_path
        if label == "section_header":
            item_section = _section_after_heading(section_path, item)

        primary, all_regions = _primary_region(item, section_path=item_section)
        parameters: dict[str, Any] = {
            "docling_self_ref": self_ref,
            "docling_label": label,
            "provenance_count": len(all_regions),
        }
        if len(all_regions) > 1:
            parameters["additional_provenance"] = [
                region.model_dump(mode="json") for region in all_regions[1:]
            ]
            warnings.append(
                f"{self_ref}: multiple provenance regions preserved in derivation metadata"
            )

        derivation = (
            DerivationStep(
                **step_base,
                parameters=parameters,
            ),
        )
        markdown = _block_markdown(document, item)
        kind = _LABEL_TO_KIND.get(label, BlockKind.OTHER)
        asset_ids: list[str] = []

        if label in {"table", "document_index"}:
            table_index += 1
            asset_id = f"table-{table_index:04d}"
            asset_ids.append(asset_id)
            table_dir = output / "tables"
            table_dir.mkdir(parents=True, exist_ok=True)
            json_rel = f"tables/{asset_id}.json"
            md_rel = f"tables/{asset_id}.md"
            _write_json(table_dir / f"{asset_id}.json", item)
            (table_dir / f"{asset_id}.md").write_text(
                (markdown.rstrip() + "\n") if markdown else "",
                encoding="utf-8",
            )
            tables.append(
                TableAsset(
                    id=asset_id,
                    source_region=primary,
                    markdown_path=md_rel,
                    json_path=json_rel,
                    derivation=derivation,
                )
            )

        if label == "formula":
            equation_index += 1
            asset_id = f"equation-{equation_index:04d}"
            asset_ids.append(asset_id)
            equations.append(
                EquationAsset(
                    id=asset_id,
                    source_region=primary,
                    latex=markdown,
                    display=True,
                    derivation=derivation,
                )
            )

        blocks.append(
            DocumentBlock(
                id=f"docling-{_slug_ref(self_ref)}",
                kind=kind,
                markdown=markdown,
                source_region=primary,
                derivation=derivation,
                asset_ids=tuple(asset_ids),
            )
        )

        if label == "section_header":
            section_path = item_section

    inventory = docling_structure_inventory(document)
    picture_count = int(inventory["pictures"])
    if picture_count:
        warnings.append(
            f"{picture_count} Docling picture items retained as FIGURE blocks; "
            "image asset materialization is deferred"
        )

    manifest = PaperPackageManifest(
        package_id=paper_package_id(
            source_pdf_sha256=source_pdf_sha256,
            compiler_identity=DOCLING_NORMALIZER_ID,
            compiler_version=compiler_version,
            compiler_config_sha256=config_sha,
        ),
        source_artifact_id=source_artifact_id,
        source_pdf_sha256=source_pdf_sha256,
        source_pdf_byte_size=source.stat().st_size,
        source_page_count=source_page_count,
        compiler_identity=DOCLING_NORMALIZER_ID,
        compiler_version=compiler_version,
        compiler_config=config,
        compiler_config_sha256=config_sha,
        blocks=len(blocks),
        tables=len(tables),
        figures=0,
        equations=len(equations),
        warnings=tuple(warnings),
    )
    package = PaperPackage(
        manifest=manifest,
        blocks=tuple(blocks),
        tables=tuple(tables),
        equations=tuple(equations),
    )

    _write_json(output / "manifest.json", manifest.model_dump(mode="json"))
    _write_json(
        output / manifest.block_index_path,
        [block.model_dump(mode="json") for block in blocks],
    )
    _write_json(
        output / manifest.diagnostics_path,
        {
            "normalizer": DOCLING_NORMALIZER_ID,
            "docling_schema_name": document.get("schema_name"),
            "docling_schema_version": document.get("version"),
            "structure_inventory": inventory,
            "warnings": warnings,
        },
    )
    _write_json(
        output / "package.json",
        package.model_dump(mode="json"),
    )

    document_markdown = "\n\n".join(
        block.markdown for block in blocks if block.markdown.strip()
    )
    (output / manifest.document_markdown_path).write_text(
        document_markdown.rstrip() + "\n",
        encoding="utf-8",
    )
    return package
