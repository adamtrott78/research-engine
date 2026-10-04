from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .document_benchmark import _multiset_f1, _text_tokens, sha256_file
from .document_docling import normalize_docling_document
from .document_package import BlockKind, PaperPackage


REQUIRED_CASES = (
    "scan-mixed-quality-v1",
    "semantic-visuals-v1",
    "malformed-layout-v1",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_residual_case_manifests(case_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(case_dir)
    result: dict[str, dict[str, Any]] = {}
    for case_id in REQUIRED_CASES:
        path = root / f"{case_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_json(path)
        if payload.get("case_id") != case_id:
            raise ValueError(f"case_id mismatch in {path}")
        result[case_id] = payload
    return result


def fetch_registered_sources(
    *,
    case_dir: str | Path,
    source_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifests = load_residual_case_manifests(case_dir)
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for case_id, manifest in manifests.items():
        source = manifest["source"]
        url = str(source["pdf_url"])
        destination = root / f"{case_id}.pdf"
        partial = destination.with_suffix(".pdf.partial")
        if partial.exists():
            partial.unlink()

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "research-engine-residual-source-audit/1.0 "
                    "(bounded scientific benchmark acquisition)"
                )
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            with partial.open("rb") as handle:
                magic = handle.read(5)
            if magic != b"%PDF-":
                raise RuntimeError(f"download is not a PDF for {case_id}: {magic!r}")
            partial.replace(destination)

            digest = sha256_file(destination)
            reader = PdfReader(str(destination))
            page_count = len(reader.pages)
            expected = int(source["expected_page_count_from_public_source"])
            if page_count != expected:
                raise RuntimeError(
                    f"page count mismatch for {case_id}: expected {expected}, got {page_count}"
                )
            rows.append(
                {
                    "case_id": case_id,
                    "source_url": url,
                    "local_path": str(destination),
                    "sha256": digest,
                    "byte_size": destination.stat().st_size,
                    "page_count": page_count,
                    "expected_page_count": expected,
                    "page_count_match": True,
                }
            )
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    report = {
        "schema_version": 1,
        "decision": "PASS",
        "purpose": "bounded_source_prefetch_only",
        "candidate_model_downloads": 0,
        "sources": rows,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def normalize_case_outputs(
    *,
    case_id: str,
    source_pdf: str | Path,
    native_json: str | Path,
    standard_json: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_pdf).resolve()
    root = Path(output_root).resolve()
    source_sha256 = sha256_file(source)
    source_artifact_id = f"benchmark-source:{case_id}"

    native_root = root / case_id / "native" / "paper-package-v1"
    standard_root = root / case_id / "standard" / "paper-package-v1"

    native = normalize_docling_document(
        docling_json_path=native_json,
        source_pdf_path=source,
        source_artifact_id=source_artifact_id,
        source_pdf_sha256=source_sha256,
        output_directory=native_root,
        docling_version="2.131.0",
        docling_mode="native",
    )
    standard = normalize_docling_document(
        docling_json_path=standard_json,
        source_pdf_path=source,
        source_artifact_id=source_artifact_id,
        source_pdf_sha256=source_sha256,
        output_directory=standard_root,
        docling_version="2.131.0",
        docling_mode="standard_no_ocr",
    )
    return {
        "case_id": case_id,
        "source_sha256": source_sha256,
        "native_package": str(native_root / "package.json"),
        "standard_package": str(standard_root / "package.json"),
        "native_blocks": len(native.blocks),
        "standard_blocks": len(standard.blocks),
    }


def _page_blocks(package: PaperPackage, page_number: int):
    return [
        block
        for block in package.blocks
        if int(block.source_region.page_number) == int(page_number)
    ]


def _page_text(package: PaperPackage, page_number: int) -> str:
    return "\n".join(
        block.markdown.strip()
        for block in _page_blocks(package, page_number)
        if block.markdown.strip()
    )


def _nonspace_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _replacement_control_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for char in text:
        code = ord(char)
        if char == "\ufffd":
            bad += 1
        elif code < 32 and char not in "\t\r\n":
            bad += 1
    return bad / len(text)


def _alphabetic_token_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    alphabetic = sum(1 for token in tokens if any(char.isalpha() for char in token))
    return alphabetic / len(tokens)


def _duplicate_block_ratio(package: PaperPackage, page_number: int) -> float:
    normalized = [
        re.sub(r"\s+", " ", block.markdown.strip().lower())
        for block in _page_blocks(package, page_number)
        if block.markdown.strip()
    ]
    if not normalized:
        return 0.0
    duplicates = len(normalized) - len(set(normalized))
    return duplicates / len(normalized)


def _clip_rect(
    rect: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = rect
    lo_x, hi_x = sorted((float(x1), float(x2)))
    lo_y, hi_y = sorted((float(y1), float(y2)))
    lo_x = max(0.0, min(width, lo_x))
    hi_x = max(0.0, min(width, hi_x))
    lo_y = max(0.0, min(height, lo_y))
    hi_y = max(0.0, min(height, hi_y))
    if hi_x <= lo_x or hi_y <= lo_y:
        return None
    return (lo_x, lo_y, hi_x, hi_y)


def _rect_union_area(rects: list[tuple[float, float, float, float]]) -> float:
    if not rects:
        return 0.0
    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in rects:
            if x1 < right and x2 > left:
                intervals.append((y1, y2))
        if not intervals:
            continue
        intervals.sort()
        covered = 0.0
        start, end = intervals[0]
        for y1, y2 in intervals[1:]:
            if y1 <= end:
                end = max(end, y2)
            else:
                covered += max(0.0, end - start)
                start, end = y1, y2
        covered += max(0.0, end - start)
        area += (right - left) * covered
    return area


def _figure_blocks(package: PaperPackage, page_number: int) -> list[Any]:
    return [
        block
        for block in _page_blocks(package, page_number)
        if block.kind == BlockKind.FIGURE
    ]


def _derivation_parameters(block: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for step in getattr(block, "derivation", ()) or ():
        parameters = getattr(step, "parameters", None)
        if isinstance(parameters, dict):
            merged.update(parameters)
    return merged


def _figure_coverage(
    *,
    native: PaperPackage,
    standard: PaperPackage,
    page_number: int,
    width: float,
    height: float,
) -> float:
    rects: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for package in (native, standard):
        # The current Docling normalizer retains pictures as FIGURE blocks.
        # FigureAsset materialization is deferred, so block provenance is the
        # production representation used by this audit.
        for block in _figure_blocks(package, page_number):
            bbox = block.source_region.pdf_bbox
            if bbox is None:
                continue
            clipped = _clip_rect(tuple(float(v) for v in bbox), width, height)
            if clipped is not None and clipped not in seen:
                seen.add(clipped)
                rects.append(clipped)

        # Preserve compatibility with future package writers that materialize
        # FigureAsset records in addition to FIGURE blocks.
        for figure in package.figures:
            if int(figure.source_region.page_number) != int(page_number):
                continue
            bbox = figure.source_region.pdf_bbox
            if bbox is None:
                continue
            clipped = _clip_rect(tuple(float(v) for v in bbox), width, height)
            if clipped is not None and clipped not in seen:
                seen.add(clipped)
                rects.append(clipped)
    if width <= 0 or height <= 0:
        return 0.0
    return min(1.0, _rect_union_area(rects) / (width * height))


def _max_text_block_overlap(
    package: PaperPackage,
    page_number: int,
) -> float:
    blocks = [
        block
        for block in _page_blocks(package, page_number)
        if block.markdown.strip() and block.source_region.pdf_bbox is not None
    ]
    maximum = 0.0
    for index, left in enumerate(blocks):
        lx1, ly1, lx2, ly2 = (float(v) for v in left.source_region.pdf_bbox)
        left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        if left_area <= 0:
            continue
        for right in blocks[index + 1 :]:
            rx1, ry1, rx2, ry2 = (float(v) for v in right.source_region.pdf_bbox)
            right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
            if right_area <= 0:
                continue
            ix = max(0.0, min(lx2, rx2) - max(lx1, rx1))
            iy = max(0.0, min(ly2, ry2) - max(ly1, ry1))
            overlap = ix * iy
            if overlap <= 0:
                continue
            maximum = max(maximum, overlap / min(left_area, right_area))
    return maximum


def _column_order_diagnostics(
    package: PaperPackage,
    page_number: int,
    page_width: float,
) -> dict[str, Any]:
    labels: list[str] = []
    for block in _page_blocks(package, page_number):
        bbox = block.source_region.pdf_bbox
        if bbox is None or not block.markdown.strip():
            continue
        x1, _, x2, _ = (float(v) for v in bbox)
        width = max(0.0, x2 - x1)
        if page_width <= 0 or width > page_width * 0.55:
            continue
        center = (x1 + x2) / 2.0
        if center < page_width * 0.45:
            labels.append("L")
        elif center > page_width * 0.55:
            labels.append("R")

    compressed: list[str] = []
    for label in labels:
        if not compressed or compressed[-1] != label:
            compressed.append(label)
    switches = max(0, len(compressed) - 1)
    return {
        "narrow_block_labels": labels,
        "compressed_column_sequence": compressed,
        "column_switches": switches,
        "interleaved_column_order": switches >= 3,
    }


def _page_metrics(
    *,
    native: PaperPackage,
    standard: PaperPackage,
    reader: PdfReader,
    page_number: int,
) -> dict[str, Any]:
    page = reader.pages[page_number - 1]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    native_text = _page_text(native, page_number)
    standard_text = _page_text(standard, page_number)
    native_tokens = _text_tokens(native_text)
    standard_tokens = _text_tokens(standard_text)
    native_bigrams = list(zip(native_tokens, native_tokens[1:]))
    standard_bigrams = list(zip(standard_tokens, standard_tokens[1:]))

    try:
        pdf_image_count = len(list(page.images))
    except Exception:
        pdf_image_count = 0

    figure_coverage = _figure_coverage(
        native=native,
        standard=standard,
        page_number=page_number,
        width=width,
        height=height,
    )
    nontext_assets = sum(
        1
        for package in (native, standard)
        for collection in (package.figures, package.tables, package.equations)
        for asset in collection
        if int(asset.source_region.page_number) == int(page_number)
    ) + sum(
        len(_figure_blocks(package, page_number))
        for package in (native, standard)
    )
    columns = _column_order_diagnostics(native, page_number, width)
    token_f1 = _multiset_f1(native_tokens, standard_tokens)
    order_f1 = _multiset_f1(native_bigrams, standard_bigrams)

    metrics = {
        "page_number": page_number,
        "page_index_zero_based": page_number - 1,
        "page_width": width,
        "page_height": height,
        "native_nonspace_characters": _nonspace_chars(native_text),
        "native_token_count": len(native_tokens),
        "standard_token_count": len(standard_tokens),
        "native_standard_token_f1": token_f1,
        "native_standard_ordered_bigram_f1": order_f1,
        "replacement_control_character_ratio": _replacement_control_ratio(native_text),
        "alphabetic_token_ratio": _alphabetic_token_ratio(native_text),
        "duplicate_block_ratio": _duplicate_block_ratio(native, page_number),
        "max_text_block_overlap_fraction": _max_text_block_overlap(native, page_number),
        "figure_bbox_page_coverage": figure_coverage,
        "pdf_image_count": pdf_image_count,
        "nontext_asset_count": nontext_assets,
        **columns,
    }

    chars = int(metrics["native_nonspace_characters"])
    visible_nontext = (
        figure_coverage > 0.0
        or pdf_image_count > 0
        or nontext_assets > 0
    )
    scan_open = (
        chars < 32 and figure_coverage >= 0.50
    ) or (
        chars == 0 and visible_nontext
    )
    low_confidence_open = (
        float(metrics["replacement_control_character_ratio"]) > 0.01
        or (
            chars >= 32
            and float(metrics["alphabetic_token_ratio"]) < 0.45
        )
        or (
            len(native_tokens) >= 10
            and len(standard_tokens) >= 10
            and token_f1 is not None
            and float(token_f1) < 0.80
        )
    )
    layout_open = (
        float(metrics["duplicate_block_ratio"]) > 0.20
        or bool(metrics["interleaved_column_order"])
        or float(metrics["max_text_block_overlap_fraction"]) >= 0.50
        or (
            len(native_tokens) >= 10
            and len(standard_tokens) >= 10
            and token_f1 is not None
            and order_f1 is not None
            and float(token_f1) >= 0.85
            and float(order_f1) < 0.70
        )
    )
    metrics["residual_openers"] = {
        "SCAN_IMAGE_ONLY_OR_TEXTLESS_PAGE": scan_open,
        "LOW_CONFIDENCE_TEXT_OR_OCR_RECOVERY": low_confidence_open,
        "LAYOUT_OR_READING_ORDER_ANOMALY": layout_open,
    }
    return metrics


def _caption_blocks(
    package: PaperPackage,
    page_number: int,
    label: str,
) -> list[Any]:
    needle = label.strip().lower()
    return [
        block
        for block in _page_blocks(package, page_number)
        if needle in block.markdown.lower()
    ]


def _semantic_region_locator(
    *,
    standard: PaperPackage,
    target: dict[str, Any],
) -> dict[str, Any]:
    page_number = int(target["page_index_zero_based"]) + 1
    label = str(target["figure_label"])
    needle = label.lower()
    block_by_id = {block.id: block for block in standard.blocks}
    linked: list[dict[str, Any]] = []

    # Current normalized Docling packages retain picture provenance and
    # caption linkage on FIGURE blocks through derivation parameters.
    for block in _figure_blocks(standard, page_number):
        parameters = _derivation_parameters(block)
        caption_text = str(parameters.get("caption_text") or "").strip()
        if needle not in caption_text.lower():
            continue
        caption_refs = [
            str(value)
            for value in (parameters.get("caption_refs") or [])
            if isinstance(value, str)
        ]
        linked.append(
            {
                "figure_block_id": block.id,
                "caption_refs": caption_refs,
                "caption_excerpt": caption_text[:240],
                "pdf_bbox": (
                    list(block.source_region.pdf_bbox)
                    if block.source_region.pdf_bbox is not None
                    else None
                ),
            }
        )

    # Future package writers may materialize FigureAsset records. Support them
    # without making them a requirement for the current source audit.
    for figure in standard.figures:
        if int(figure.source_region.page_number) != page_number:
            continue
        if not figure.caption_block_id:
            continue
        caption = block_by_id.get(figure.caption_block_id)
        if caption is None or needle not in caption.markdown.lower():
            continue
        linked.append(
            {
                "figure_asset_id": figure.id,
                "caption_block_id": caption.id,
                "caption_excerpt": caption.markdown[:240],
                "pdf_bbox": (
                    list(figure.source_region.pdf_bbox)
                    if figure.source_region.pdf_bbox is not None
                    else None
                ),
            }
        )

    captions = _caption_blocks(standard, page_number, label)
    status = (
        "LINKED_FIGURE"
        if len(linked) == 1 and linked[0]["pdf_bbox"]
        else ("AMBIGUOUS_LINKED_FIGURES" if linked else "CAPTION_ONLY_OR_UNLOCATED")
    )
    return {
        "page_number": page_number,
        "page_index_zero_based": page_number - 1,
        "figure_label": label,
        "status": status,
        "linked_figures": linked,
        "caption_blocks_found": [
            {
                "id": block.id,
                "markdown_excerpt": block.markdown[:240],
                "pdf_bbox": (
                    list(block.source_region.pdf_bbox)
                    if block.source_region.pdf_bbox is not None
                    else None
                ),
            }
            for block in captions
        ],
    }


def _load_package(path: Path) -> PaperPackage:
    return PaperPackage.model_validate_json(path.read_text(encoding="utf-8"))


def audit_registered_sources(
    *,
    case_dir: str | Path,
    source_root: str | Path,
    package_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifests = load_residual_case_manifests(case_dir)
    sources = Path(source_root)
    packages = Path(package_root)
    cases: dict[str, Any] = {}
    overall_pass = True
    model_prefetch_ready = True

    for case_id, manifest in manifests.items():
        source = sources / f"{case_id}.pdf"
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        reader = PdfReader(str(source))
        page_count = len(reader.pages)

        native_path = packages / case_id / "native" / "paper-package-v1" / "package.json"
        standard_path = (
            packages / case_id / "standard" / "paper-package-v1" / "package.json"
        )
        native = _load_package(native_path)
        standard = _load_package(standard_path)
        if native.manifest.source_pdf_sha256 != digest:
            raise RuntimeError(f"native package hash mismatch for {case_id}")
        if standard.manifest.source_pdf_sha256 != digest:
            raise RuntimeError(f"standard package hash mismatch for {case_id}")

        per_page = [
            _page_metrics(
                native=native,
                standard=standard,
                reader=reader,
                page_number=page_number,
            )
            for page_number in range(1, page_count + 1)
        ]
        by_index = {row["page_index_zero_based"]: row for row in per_page}

        row: dict[str, Any] = {
            "case_id": case_id,
            "sha256": digest,
            "byte_size": source.stat().st_size,
            "page_count": page_count,
            "native_blocks": len(native.blocks),
            "standard_blocks": len(standard.blocks),
            "per_page": per_page,
        }

        if case_id == "scan-mixed-quality-v1":
            scan_pages = [
                item["page_index_zero_based"]
                for item in per_page
                if item["residual_openers"]["SCAN_IMAGE_ONLY_OR_TEXTLESS_PAGE"]
            ]
            low_conf_pages = [
                item["page_index_zero_based"]
                for item in per_page
                if item["residual_openers"]["LOW_CONFIDENCE_TEXT_OR_OCR_RECOVERY"]
            ]
            target_results = []
            for target in manifest["target_pages"]:
                page_index = int(target["page_index_zero_based"])
                target_results.append(
                    {
                        "page_index_zero_based": page_index,
                        "intended_residual": target["intended_residual"],
                        "metrics": by_index[page_index],
                    }
                )
            accepted = bool(scan_pages) and bool(low_conf_pages)
            row.update(
                {
                    "accepted": accepted,
                    "scan_opener_pages_zero_based": scan_pages,
                    "low_confidence_opener_pages_zero_based": low_conf_pages,
                    "target_results": target_results,
                }
            )
        elif case_id == "semantic-visuals-v1":
            locators = [
                _semantic_region_locator(standard=standard, target=target)
                for target in manifest["target_regions"]
            ]
            valid_pages = all(
                0 <= int(target["page_index_zero_based"]) < page_count
                for target in manifest["target_regions"]
            )
            captions_present = all(
                bool(locator["caption_blocks_found"]) for locator in locators
            )
            accepted = valid_pages and captions_present
            ready = accepted and all(
                locator["status"] == "LINKED_FIGURE" for locator in locators
            )
            row.update(
                {
                    "accepted": accepted,
                    "semantic_region_locators": locators,
                    "region_locators_ready_for_model_prefetch": ready,
                }
            )
            if not ready:
                model_prefetch_ready = False
        elif case_id == "malformed-layout-v1":
            targets = []
            for target in manifest["target_pages"]:
                page_index = int(target["page_index_zero_based"])
                metrics = by_index[page_index]
                targets.append(
                    {
                        "page_index_zero_based": page_index,
                        "layout_opener_fired": metrics["residual_openers"][
                            "LAYOUT_OR_READING_ORDER_ANOMALY"
                        ],
                        "metrics": metrics,
                    }
                )
            accepted = any(item["layout_opener_fired"] for item in targets)
            row.update(
                {
                    "accepted": accepted,
                    "target_results": targets,
                    "replacement_required": not accepted,
                }
            )
            if not accepted:
                model_prefetch_ready = False
        else:
            raise AssertionError(case_id)

        if not row["accepted"]:
            overall_pass = False
        cases[case_id] = row

    report = {
        "schema_version": 1,
        "decision": "PASS" if overall_pass else "REJECT",
        "authority": "DOWNLOADED_ORIGINAL_PDF",
        "purpose": "residual_source_prefetch_and_deterministic_opener_audit",
        "candidate_model_downloads": 0,
        "candidate_model_inference_calls": 0,
        "scientific_state_materialized": False,
        "model_prefetch_ready": bool(overall_pass and model_prefetch_ready),
        "opener_thresholds": {
            "scan_native_nonspace_characters_lt": 32,
            "scan_figure_bbox_page_coverage_gte": 0.50,
            "low_confidence_replacement_control_ratio_gt": 0.01,
            "low_confidence_alphabetic_token_ratio_lt": 0.45,
            "low_confidence_native_standard_token_f1_lt": 0.80,
            "layout_duplicate_block_ratio_gt": 0.20,
            "layout_max_text_block_overlap_fraction_gte": 0.50,
            "layout_native_standard_token_f1_gte": 0.85,
            "layout_native_standard_ordered_bigram_f1_lt": 0.70,
            "layout_interleaved_column_switches_gte": 3,
        },
        "cases": cases,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--case-dir", required=True)
    fetch.add_argument("--source-root", required=True)
    fetch.add_argument("--output", required=True)

    normalize = sub.add_parser("normalize")
    normalize.add_argument("--case-id", required=True, choices=REQUIRED_CASES)
    normalize.add_argument("--source-pdf", required=True)
    normalize.add_argument("--native-json", required=True)
    normalize.add_argument("--standard-json", required=True)
    normalize.add_argument("--output-root", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--case-dir", required=True)
    audit.add_argument("--source-root", required=True)
    audit.add_argument("--package-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--fail-on-reject", action="store_true")

    args = parser.parse_args()
    if args.command == "fetch":
        report = fetch_registered_sources(
            case_dir=args.case_dir,
            source_root=args.source_root,
            output_path=args.output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "normalize":
        report = normalize_case_outputs(
            case_id=args.case_id,
            source_pdf=args.source_pdf,
            native_json=args.native_json,
            standard_json=args.standard_json,
            output_root=args.output_root,
        )
        print(json.dumps(report, sort_keys=True))
        return 0

    report = audit_registered_sources(
        case_dir=args.case_dir,
        source_root=args.source_root,
        package_root=args.package_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "model_prefetch_ready": report["model_prefetch_ready"],
                "cases": {
                    key: value["accepted"]
                    for key, value in report["cases"].items()
                },
            },
            sort_keys=True,
        )
    )
    if args.fail_on_reject and report["decision"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
