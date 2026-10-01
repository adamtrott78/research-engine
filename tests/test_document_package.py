import pytest
from pydantic import ValidationError

from research_engine.document_package import (
    BlockKind,
    DerivationStep,
    DocumentBlock,
    FigureAsset,
    PaperPackage,
    PaperPackageManifest,
    PdfRegion,
    ProcessingMode,
    compiler_config_sha256,
    paper_package_id,
)


SOURCE_SHA = "a" * 64


def _manifest(*, config=None, blocks=1, figures=0):
    config = config or {"mode": "native-first", "ocr": "auto"}
    config_sha = compiler_config_sha256(config)
    return PaperPackageManifest(
        package_id=paper_package_id(
            source_pdf_sha256=SOURCE_SHA,
            compiler_identity="test-adapter",
            compiler_version="1.0.0",
            compiler_config_sha256=config_sha,
        ),
        source_artifact_id=f"source-artifact:sha256:{SOURCE_SHA}",
        source_pdf_sha256=SOURCE_SHA,
        source_pdf_byte_size=12345,
        source_page_count=3,
        compiler_identity="test-adapter",
        compiler_version="1.0.0",
        compiler_config=config,
        compiler_config_sha256=config_sha,
        blocks=blocks,
        figures=figures,
    )


def _step(mode=ProcessingMode.NATIVE_TEXT):
    return DerivationStep(
        processor="test-adapter",
        processor_version="1.0.0",
        mode=mode,
        confidence=0.99,
    )


def test_paper_package_identity_is_stable_and_config_sensitive():
    c1 = compiler_config_sha256({"ocr": "auto", "layout": True})
    c2 = compiler_config_sha256({"layout": True, "ocr": "auto"})
    assert c1 == c2

    p1 = paper_package_id(
        source_pdf_sha256=SOURCE_SHA,
        compiler_identity="adapter",
        compiler_version="1",
        compiler_config_sha256=c1,
    )
    p2 = paper_package_id(
        source_pdf_sha256=SOURCE_SHA,
        compiler_identity="adapter",
        compiler_version="1",
        compiler_config_sha256=c2,
    )
    assert p1 == p2

    c3 = compiler_config_sha256({"ocr": "force", "layout": True})
    p3 = paper_package_id(
        source_pdf_sha256=SOURCE_SHA,
        compiler_identity="adapter",
        compiler_version="1",
        compiler_config_sha256=c3,
    )
    assert p3 != p1


def test_manifest_rejects_mismatched_config_or_package_identity():
    config = {"ocr": "auto"}
    config_sha = compiler_config_sha256(config)

    with pytest.raises(ValidationError, match="compiler_config_sha256"):
        PaperPackageManifest(
            package_id="paper-package:v1:sha256:" + "b" * 64,
            source_artifact_id=f"source-artifact:sha256:{SOURCE_SHA}",
            source_pdf_sha256=SOURCE_SHA,
            source_pdf_byte_size=1,
            source_page_count=1,
            compiler_identity="adapter",
            compiler_version="1",
            compiler_config=config,
            compiler_config_sha256="c" * 64,
        )

    with pytest.raises(ValidationError, match="package_id"):
        PaperPackageManifest(
            package_id="paper-package:v1:sha256:" + "b" * 64,
            source_artifact_id=f"source-artifact:sha256:{SOURCE_SHA}",
            source_pdf_sha256=SOURCE_SHA,
            source_pdf_byte_size=1,
            source_page_count=1,
            compiler_identity="adapter",
            compiler_version="1",
            compiler_config=config,
            compiler_config_sha256=config_sha,
        )


def test_every_block_requires_source_provenance_and_derivation():
    with pytest.raises(ValidationError):
        DocumentBlock(
            id="block-1",
            kind=BlockKind.PARAGRAPH,
            markdown="hello",
            source_region=PdfRegion(page_number=1),
            derivation=(),
        )

    block = DocumentBlock(
        id="block-1",
        kind=BlockKind.PARAGRAPH,
        markdown="hello",
        source_region=PdfRegion(
            page_number=2,
            pdf_bbox=(10, 20, 100, 80),
            section_path=("Methods",),
        ),
        derivation=(_step(),),
    )
    assert block.source_region.page_number == 2
    assert block.derivation[0].mode == ProcessingMode.NATIVE_TEXT


def test_region_rejects_invalid_bbox():
    with pytest.raises(ValidationError, match="pdf_bbox"):
        PdfRegion(page_number=1, pdf_bbox=(100, 20, 10, 80))


def test_package_validates_asset_links_and_manifest_counts():
    caption = DocumentBlock(
        id="caption-1",
        kind=BlockKind.CAPTION,
        markdown="Figure 1.",
        source_region=PdfRegion(page_number=1),
        derivation=(_step(),),
        asset_ids=("figure-1",),
    )
    figure = FigureAsset(
        id="figure-1",
        source_region=PdfRegion(page_number=1, pdf_bbox=(0, 0, 100, 100)),
        image_path="figures/figure-1.png",
        caption_block_id="caption-1",
        derivation=(_step(ProcessingMode.DETERMINISTIC),),
    )

    package = PaperPackage(
        manifest=_manifest(figures=1),
        blocks=(caption,),
        figures=(figure,),
    )
    assert package.manifest.authority == "DERIVED_FROM_AUTHORITATIVE_PDF"

    with pytest.raises(ValidationError, match="block count"):
        PaperPackage(
            manifest=_manifest(blocks=2, figures=1),
            blocks=(caption,),
            figures=(figure,),
        )


def test_package_paths_are_relative_and_semantic_figure_requires_description():
    with pytest.raises(ValidationError):
        FigureAsset(
            id="figure-1",
            source_region=PdfRegion(page_number=1),
            image_path="/tmp/figure.png",
            derivation=(_step(),),
        )

    with pytest.raises(ValidationError, match="semantic_description_path"):
        FigureAsset(
            id="figure-1",
            source_region=PdfRegion(page_number=1),
            image_path="figures/figure.png",
            semantic_analyzed=True,
            derivation=(_step(ProcessingMode.DOCUMENT_VLM),),
        )
