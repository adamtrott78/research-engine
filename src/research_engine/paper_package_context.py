from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .document_package import BlockKind, PaperPackage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CatalogEntry:
    package_path: Path
    package_root: Path
    package: PaperPackage


class PaperPackageCatalog:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._by_sha: dict[str, CatalogEntry] = {}
        paths = sorted(self.root.rglob("package.json"))
        if not paths:
            raise ValueError(f"no paper-package package.json files under {self.root}")
        for path in paths:
            package = PaperPackage.model_validate_json(path.read_text(encoding="utf-8"))
            sha = package.manifest.source_pdf_sha256
            entry = CatalogEntry(path, path.parent, package)
            existing = self._by_sha.get(sha)
            if existing is not None:
                if existing.package.manifest.package_id != package.manifest.package_id:
                    raise ValueError(
                        f"ambiguous paper-package catalog for source SHA {sha}: "
                        f"{existing.package_path} vs {path}"
                    )
                continue
            self._by_sha[sha] = entry

    def source_shas(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_sha))

    def resolve_sha(self, sha256: str) -> CatalogEntry:
        try:
            return self._by_sha[sha256]
        except KeyError as exc:
            raise FileNotFoundError(
                f"paper-package unavailable for source SHA {sha256} under {self.root}"
            ) from exc

    def resolve_pdf(self, source_pdf: str | Path) -> CatalogEntry:
        source = Path(source_pdf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        entry = self.resolve_sha(_sha256_file(source))
        if entry.package.manifest.source_pdf_byte_size != source.stat().st_size:
            raise ValueError(f"paper-package byte-size mismatch for {source}")
        return entry


def _caption_text(block) -> str:
    for step in block.derivation:
        value = step.parameters.get("caption_text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _render_block(block) -> str:
    text = str(block.markdown or "").strip()
    if block.kind == BlockKind.FIGURE and not text:
        caption = _caption_text(block)
        return f"Figure: {caption}" if caption else ""
    if not text:
        return ""
    if block.kind == BlockKind.TITLE:
        return text if text.startswith("#") else f"# {text}"
    if block.kind == BlockKind.HEADING:
        if text.startswith("#"):
            return text
        depth = max(1, min(6, len(block.source_region.section_path) or 1))
        return f"{'#' * depth} {text}"
    return text


def _normalized_heading(text: str) -> str:
    value = re.sub(r"^[#\s\d.IVXivx-]+", "", text or "")
    return re.sub(r"\s+", " ", value).strip().lower()


def reference_start_page(package: PaperPackage) -> int | None:
    pages: list[int] = []
    for block in package.blocks:
        if block.kind == BlockKind.REFERENCE:
            pages.append(block.source_region.page_number)
        elif block.kind == BlockKind.HEADING:
            heading = _normalized_heading(block.markdown)
            if heading in {"references", "bibliography", "reference"}:
                pages.append(block.source_region.page_number)
    return min(pages) if pages else None


class HybridPaperPackagePageTextExtractor:
    """Render native lexical blocks with standard/no-OCR structural gating."""

    def __init__(
        self,
        *,
        native_package_root: str | Path,
        standard_package_root: str | Path | None = None,
        max_chars: int = 120_000,
        exclude_trailing_references: bool = True,
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.native = PaperPackageCatalog(native_package_root)
        self.standard = (
            PaperPackageCatalog(standard_package_root)
            if standard_package_root is not None
            else None
        )
        self.max_chars = int(max_chars)
        self.exclude_trailing_references = bool(exclude_trailing_references)

    def coverage(self, source_shas: Iterable[str]) -> dict[str, object]:
        requested = sorted(set(str(x) for x in source_shas))
        native = set(self.native.source_shas())
        standard = set(self.standard.source_shas()) if self.standard else set()
        return {
            "requested": len(requested),
            "native_covered": sum(sha in native for sha in requested),
            "standard_covered": (
                sum(sha in standard for sha in requested)
                if self.standard is not None
                else None
            ),
            "missing_native": [sha for sha in requested if sha not in native],
            "missing_standard": (
                [sha for sha in requested if sha not in standard]
                if self.standard is not None
                else []
            ),
        }

    def _paired_entries(
        self, source_pdf: str | Path
    ) -> tuple[CatalogEntry, CatalogEntry | None]:
        native = self.native.resolve_pdf(source_pdf)
        standard = None
        if self.standard is not None:
            standard = self.standard.resolve_sha(
                native.package.manifest.source_pdf_sha256
            )
            if standard.package.manifest.source_page_count != native.package.manifest.source_page_count:
                raise ValueError("native/standard paper-package page-count mismatch")
        return native, standard

    def describe(self, source_pdf: str | Path) -> dict[str, object]:
        native, standard = self._paired_entries(source_pdf)
        return {
            "source_pdf_sha256": native.package.manifest.source_pdf_sha256,
            "native_package_id": native.package.manifest.package_id,
            "native_compiler_identity": native.package.manifest.compiler_identity,
            "standard_package_id": standard.package.manifest.package_id if standard else None,
            "standard_compiler_identity": (
                standard.package.manifest.compiler_identity if standard else None
            ),
            "reference_start_page": (
                reference_start_page(standard.package) if standard else None
            ),
            "exclude_trailing_references": self.exclude_trailing_references,
        }

    def __call__(self, source_pdf: Path) -> list[dict[str, object]]:
        native, standard = self._paired_entries(source_pdf)
        package = native.package
        page_count = package.manifest.source_page_count
        ref_page = reference_start_page(standard.package) if standard else None
        by_page: dict[int, list[str]] = {
            page: [] for page in range(1, page_count + 1)
        }
        for block in package.blocks:
            page = int(block.source_region.page_number)
            if page < 1 or page > page_count:
                raise ValueError(f"paper-package block {block.id} page is out of range")
            if block.kind in {BlockKind.REFERENCE, BlockKind.FOOTNOTE}:
                continue
            rendered = _render_block(block)
            if rendered:
                by_page[page].append(rendered)

        pages: list[dict[str, object]] = []
        remaining = self.max_chars
        for page in range(1, page_count + 1):
            if remaining <= 0:
                break
            if self.exclude_trailing_references and ref_page is not None and page > ref_page:
                continue
            text = "\n\n".join(by_page[page]).strip()
            if len(text) > remaining:
                text = text[:remaining]
            pages.append({"page_number": page, "text": text})
            remaining -= len(text)
        if not pages or not any(str(page["text"]).strip() for page in pages):
            raise ValueError("paper-package yielded no readable context text")
        return pages
