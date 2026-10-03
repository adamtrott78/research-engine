import hashlib
import json
from pathlib import Path

import pytest

from research_engine.document_package import (
    BlockKind,
    DerivationStep,
    DocumentBlock,
    PaperPackage,
    PaperPackageManifest,
    PdfRegion,
    ProcessingMode,
    compiler_config_sha256,
    paper_package_id,
)
from research_engine.paper_package_context import (
    HybridPaperPackagePageTextExtractor,
    PaperPackageCatalog,
    reference_start_page,
)


def _step(*, caption_text: str | None = None):
    parameters = {}
    if caption_text is not None:
        parameters["caption_text"] = caption_text
    return DerivationStep(
        processor="fixture",
        processor_version="1",
        mode=ProcessingMode.NATIVE_TEXT,
        parameters=parameters,
    )


def _package(
    *,
    source: Path,
    compiler: str,
    blocks: list[DocumentBlock],
    page_count: int = 4,
) -> PaperPackage:
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {"compiler": compiler}
    config_sha = compiler_config_sha256(config)
    manifest = PaperPackageManifest(
        package_id=paper_package_id(
            source_pdf_sha256=source_sha,
            compiler_identity=compiler,
            compiler_version="1",
            compiler_config_sha256=config_sha,
        ),
        source_artifact_id=f"source-artifact:sha256:{source_sha}",
        source_pdf_sha256=source_sha,
        source_pdf_byte_size=source.stat().st_size,
        source_page_count=page_count,
        compiler_identity=compiler,
        compiler_version="1",
        compiler_config=config,
        compiler_config_sha256=config_sha,
        blocks=len(blocks),
    )
    return PaperPackage(manifest=manifest, blocks=tuple(blocks))


def _write_package(root: Path, case: str, package: PaperPackage):
    path = root / case / "paper-package-v1"
    path.mkdir(parents=True)
    (path / "package.json").write_text(
        json.dumps(package.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def test_catalog_resolves_by_authoritative_source_sha(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    block = DocumentBlock(
        id="b1",
        kind=BlockKind.PARAGRAPH,
        markdown="Body text",
        source_region=PdfRegion(page_number=1),
        derivation=(_step(),),
    )
    package = _package(source=source, compiler="native", blocks=[block])
    root = tmp_path / "packages"
    _write_package(root, "case", package)

    catalog = PaperPackageCatalog(root)
    resolved = catalog.resolve_pdf(source)
    assert resolved.package.manifest.package_id == package.manifest.package_id


def test_catalog_rejects_two_compilers_for_same_sha_in_one_root(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    block = DocumentBlock(
        id="b1",
        kind=BlockKind.PARAGRAPH,
        markdown="Body",
        source_region=PdfRegion(page_number=1),
        derivation=(_step(),),
    )
    root = tmp_path / "packages"
    _write_package(root, "native", _package(source=source, compiler="native", blocks=[block]))
    _write_package(root, "standard", _package(source=source, compiler="standard", blocks=[block]))

    with pytest.raises(ValueError, match="ambiguous"):
        PaperPackageCatalog(root)


def test_reference_start_uses_standard_structure(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    blocks = [
        DocumentBlock(
            id="h1",
            kind=BlockKind.HEADING,
            markdown="REFERENCES",
            source_region=PdfRegion(page_number=3, section_path=("REFERENCES",)),
            derivation=(_step(),),
        ),
        DocumentBlock(
            id="r1",
            kind=BlockKind.REFERENCE,
            markdown="[1] prior work",
            source_region=PdfRegion(page_number=3),
            derivation=(_step(),),
        ),
    ]
    package = _package(source=source, compiler="standard", blocks=blocks)
    assert reference_start_page(package) == 3


def test_hybrid_context_keeps_native_lexical_text_and_drops_trailing_reference_pages(
    tmp_path: Path,
):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    native_blocks = [
        DocumentBlock(
            id="p1",
            kind=BlockKind.PARAGRAPH,
            markdown="Native body page one",
            source_region=PdfRegion(page_number=1),
            derivation=(_step(),),
        ),
        DocumentBlock(
            id="p2",
            kind=BlockKind.PARAGRAPH,
            markdown="Native result page two",
            source_region=PdfRegion(page_number=2),
            derivation=(_step(),),
        ),
        DocumentBlock(
            id="p3",
            kind=BlockKind.PARAGRAPH,
            markdown="Native mixed page three",
            source_region=PdfRegion(page_number=3),
            derivation=(_step(),),
        ),
        DocumentBlock(
            id="p4",
            kind=BlockKind.PARAGRAPH,
            markdown="Native references page four",
            source_region=PdfRegion(page_number=4),
            derivation=(_step(),),
        ),
    ]
    standard_blocks = [
        DocumentBlock(
            id="h1",
            kind=BlockKind.HEADING,
            markdown="References",
            source_region=PdfRegion(page_number=3, section_path=("References",)),
            derivation=(_step(),),
        ),
        DocumentBlock(
            id="r1",
            kind=BlockKind.REFERENCE,
            markdown="[1] prior work",
            source_region=PdfRegion(page_number=3),
            derivation=(_step(),),
        ),
    ]

    native_root = tmp_path / "native"
    standard_root = tmp_path / "standard"
    _write_package(
        native_root,
        "case",
        _package(source=source, compiler="native", blocks=native_blocks),
    )
    _write_package(
        standard_root,
        "case",
        _package(source=source, compiler="standard", blocks=standard_blocks),
    )

    extractor = HybridPaperPackagePageTextExtractor(
        native_package_root=native_root,
        standard_package_root=standard_root,
    )
    pages = extractor(source)

    assert [row["page_number"] for row in pages] == [1, 2, 3]
    assert pages[1]["text"] == "Native result page two"
    assert "page four" not in "\n".join(str(row["text"]) for row in pages)
    description = extractor.describe(source)
    assert description["reference_start_page"] == 3
    assert description["native_compiler_identity"] == "native"
    assert description["standard_compiler_identity"] == "standard"


def test_hybrid_context_uses_caption_for_blank_figure(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    figure = DocumentBlock(
        id="fig1",
        kind=BlockKind.FIGURE,
        markdown="",
        source_region=PdfRegion(page_number=1),
        derivation=(_step(caption_text="Figure 1. Detection rate."),),
    )
    package = _package(source=source, compiler="native", blocks=[figure])
    root = tmp_path / "native"
    _write_package(root, "case", package)

    extractor = HybridPaperPackagePageTextExtractor(native_package_root=root)
    pages = extractor(source)
    assert pages[0]["text"] == "Figure: Figure 1. Detection rate."



def test_verify_policy_keeps_pages_after_reference_start(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-fixture")
    native_blocks = [
        DocumentBlock(
            id=f"p{page}",
            kind=BlockKind.PARAGRAPH,
            markdown=f"Native page {page}",
            source_region=PdfRegion(page_number=page),
            derivation=(_step(),),
        )
        for page in range(1, 5)
    ]
    standard_blocks = [
        DocumentBlock(
            id="h1",
            kind=BlockKind.HEADING,
            markdown="References",
            source_region=PdfRegion(page_number=3, section_path=("References",)),
            derivation=(_step(),),
        )
    ]
    native_root = tmp_path / "native"
    standard_root = tmp_path / "standard"
    _write_package(
        native_root,
        "case",
        _package(source=source, compiler="native", blocks=native_blocks),
    )
    _write_package(
        standard_root,
        "case",
        _package(source=source, compiler="standard", blocks=standard_blocks),
    )

    extractor = HybridPaperPackagePageTextExtractor(
        native_package_root=native_root,
        standard_package_root=standard_root,
        exclude_trailing_references=False,
    )
    pages = extractor(source)
    assert [row["page_number"] for row in pages] == [1, 2, 3, 4]
