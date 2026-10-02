from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PAPER_PACKAGE_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1


class BlockKind(StrEnum):
    TITLE = "TITLE"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST = "LIST"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    EQUATION = "EQUATION"
    CODE = "CODE"
    CAPTION = "CAPTION"
    REFERENCE = "REFERENCE"
    FOOTNOTE = "FOOTNOTE"
    OTHER = "OTHER"


class ProcessingMode(StrEnum):
    NATIVE_TEXT = "NATIVE_TEXT"
    DETERMINISTIC = "DETERMINISTIC"
    OCR = "OCR"
    LAYOUT_MODEL = "LAYOUT_MODEL"
    TABLE_MODEL = "TABLE_MODEL"
    FORMULA_MODEL = "FORMULA_MODEL"
    DOCUMENT_VLM = "DOCUMENT_VLM"
    GENERAL_VLM_REPAIR = "GENERAL_VLM_REPAIR"
    MANUAL_ORACLE = "MANUAL_ORACLE"


class CandidateClass(StrEnum):
    DETERMINISTIC_LOCAL = "DETERMINISTIC_LOCAL"
    SPECIALIZED_LOCAL_MODEL = "SPECIALIZED_LOCAL_MODEL"
    GENERAL_LOCAL_MODEL = "GENERAL_LOCAL_MODEL"
    EXTERNAL_WEB = "EXTERNAL_WEB"
    REMOTE_REASONING = "REMOTE_REASONING"
    HUMAN = "HUMAN"


class OracleImageMode(StrEnum):
    REMOTE_IMAGES = "REMOTE_IMAGES"
    LOCAL_IMAGES = "LOCAL_IMAGES"
    MIXED = "MIXED"
    NO_IMAGES = "NO_IMAGES"


def _validate_sha256(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return value


def _validate_relative_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("package paths must be relative and may not traverse upward")
    return value


class PdfRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    pdf_bbox: tuple[float, float, float, float] | None = None
    section_path: tuple[str, ...] = ()
    raw_anchor: str | None = None

    @field_validator("pdf_bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        x1, y1, x2, y2 = value
        if x2 < x1 or y2 < y1:
            raise ValueError("pdf_bbox must satisfy x2>=x1 and y2>=y1")
        return value


class DerivationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processor: str
    processor_version: str | None = None
    mode: ProcessingMode
    model_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)


class DocumentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: BlockKind
    markdown: str
    source_region: PdfRegion
    derivation: tuple[DerivationStep, ...]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    asset_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_derivation(self) -> "DocumentBlock":
        if not self.derivation:
            raise ValueError("every derived block must record at least one derivation step")
        return self


class TableAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_region: PdfRegion
    markdown_path: str | None = None
    csv_path: str | None = None
    json_path: str | None = None
    caption_block_id: str | None = None
    derivation: tuple[DerivationStep, ...]

    _markdown_path = field_validator("markdown_path")(_validate_relative_path)
    _csv_path = field_validator("csv_path")(_validate_relative_path)
    _json_path = field_validator("json_path")(_validate_relative_path)


class FigureAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_region: PdfRegion
    image_path: str
    caption_block_id: str | None = None
    semantic_description_path: str | None = None
    semantic_analyzed: bool = False
    derivation: tuple[DerivationStep, ...]

    _image_path = field_validator("image_path")(_validate_relative_path)
    _semantic_path = field_validator("semantic_description_path")(_validate_relative_path)

    @model_validator(mode="after")
    def semantic_path_matches_flag(self) -> "FigureAsset":
        if self.semantic_analyzed and not self.semantic_description_path:
            raise ValueError(
                "semantic_analyzed figures must provide semantic_description_path"
            )
        return self


class EquationAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_region: PdfRegion
    latex: str
    display: bool = True
    derivation: tuple[DerivationStep, ...]


class PaperPackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=PAPER_PACKAGE_SCHEMA_VERSION, frozen=True)
    package_id: str
    source_artifact_id: str
    source_pdf_sha256: str
    source_pdf_byte_size: int = Field(ge=1)
    source_page_count: int = Field(ge=1)
    compiler_identity: str
    compiler_version: str
    compiler_config: dict[str, Any] = Field(default_factory=dict)
    compiler_config_sha256: str
    document_markdown_path: str = "document.md"
    block_index_path: str = "blocks.json"
    diagnostics_path: str = "diagnostics.json"
    authority: str = Field(default="DERIVED_FROM_AUTHORITATIVE_PDF", frozen=True)
    blocks: int = Field(default=0, ge=0)
    tables: int = Field(default=0, ge=0)
    figures: int = Field(default=0, ge=0)
    equations: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = ()

    _source_sha = field_validator("source_pdf_sha256")(_validate_sha256)
    _config_sha = field_validator("compiler_config_sha256")(_validate_sha256)
    _document_path = field_validator("document_markdown_path")(_validate_relative_path)
    _block_path = field_validator("block_index_path")(_validate_relative_path)
    _diagnostics_path = field_validator("diagnostics_path")(_validate_relative_path)

    @model_validator(mode="after")
    def validate_identity(self) -> "PaperPackageManifest":
        expected_config = compiler_config_sha256(self.compiler_config)
        if self.compiler_config_sha256 != expected_config:
            raise ValueError("compiler_config_sha256 does not match compiler_config")
        expected_id = paper_package_id(
            source_pdf_sha256=self.source_pdf_sha256,
            compiler_identity=self.compiler_identity,
            compiler_version=self.compiler_version,
            compiler_config_sha256=self.compiler_config_sha256,
        )
        if self.package_id != expected_id:
            raise ValueError("package_id does not match source/compiler identity")
        return self


class PaperPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: PaperPackageManifest
    blocks: tuple[DocumentBlock, ...]
    tables: tuple[TableAsset, ...] = ()
    figures: tuple[FigureAsset, ...] = ()
    equations: tuple[EquationAsset, ...] = ()

    @model_validator(mode="after")
    def validate_counts_and_links(self) -> "PaperPackage":
        if self.manifest.blocks != len(self.blocks):
            raise ValueError("manifest block count does not match package")
        if self.manifest.tables != len(self.tables):
            raise ValueError("manifest table count does not match package")
        if self.manifest.figures != len(self.figures):
            raise ValueError("manifest figure count does not match package")
        if self.manifest.equations != len(self.equations):
            raise ValueError("manifest equation count does not match package")

        block_ids = {block.id for block in self.blocks}
        asset_ids = {
            *(asset.id for asset in self.tables),
            *(asset.id for asset in self.figures),
            *(asset.id for asset in self.equations),
        }
        if len(block_ids) != len(self.blocks):
            raise ValueError("document block ids must be unique")
        if len(asset_ids) != len(self.tables) + len(self.figures) + len(self.equations):
            raise ValueError("asset ids must be unique across asset types")
        for block in self.blocks:
            missing = set(block.asset_ids) - asset_ids
            if missing:
                raise ValueError(
                    f"block {block.id} references missing assets: {sorted(missing)}"
                )
        for asset in (*self.tables, *self.figures):
            if asset.caption_block_id and asset.caption_block_id not in block_ids:
                raise ValueError(
                    f"asset {asset.id} references missing caption block "
                    f"{asset.caption_block_id}"
                )
        return self


class DocumentBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=BENCHMARK_SCHEMA_VERSION, frozen=True)
    case_id: str
    source_pdf_sha256: str
    source_pdf_byte_size: int = Field(ge=1)
    page_count: int = Field(ge=1)
    difficulty_tags: tuple[str, ...]
    source_location: str
    mathpix_oracle_location: str | None = None
    mathpix_oracle_sha256: str | None = None
    mathpix_image_mode: OracleImageMode | None = None
    expected_features: dict[str, int | bool | str | list[str]] = Field(
        default_factory=dict
    )
    notes: str | None = None

    _source_sha = field_validator("source_pdf_sha256")(_validate_sha256)

    @field_validator("mathpix_oracle_sha256")
    @classmethod
    def validate_optional_oracle_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value)

    @model_validator(mode="after")
    def validate_oracle_metadata(self) -> "DocumentBenchmarkCase":
        if self.mathpix_oracle_location is None:
            if self.mathpix_oracle_sha256 is not None or self.mathpix_image_mode is not None:
                raise ValueError(
                    "Mathpix oracle metadata requires mathpix_oracle_location"
                )
        elif self.mathpix_oracle_sha256 is None:
            raise ValueError(
                "mathpix_oracle_location requires mathpix_oracle_sha256"
            )
        return self


class ResourceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_time_seconds: float = Field(ge=0)
    pages_per_second: float = Field(ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    mean_cpu_percent: float | None = Field(default=None, ge=0)
    mean_gpu_percent: float | None = Field(default=None, ge=0)
    model_load_seconds: float | None = Field(default=None, ge=0)


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_fidelity: float | None = Field(default=None, ge=0.0, le=1.0)
    reading_order: float | None = Field(default=None, ge=0.0, le=1.0)
    equation_fidelity: float | None = Field(default=None, ge=0.0, le=1.0)
    table_structure: float | None = Field(default=None, ge=0.0, le=1.0)
    figure_caption_association: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    provenance_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    downstream_evidence_agreement: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    downstream_verify_agreement: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class DocumentBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=BENCHMARK_SCHEMA_VERSION, frozen=True)
    case_id: str
    adapter_id: str
    adapter_version: str
    package_id: str | None = None
    normalization_status: str = "PAPER_PACKAGE_V1"
    raw_output_paths: tuple[str, ...] = ()
    status: str
    quality: QualityMetrics
    resources: ResourceMetrics
    output_markdown_chars: int = Field(ge=0)
    output_markdown_tokens_estimate: int | None = Field(default=None, ge=0)
    native_pages: int = Field(default=0, ge=0)
    ocr_pages: int = Field(default=0, ge=0)
    specialized_model_pages: int = Field(default=0, ge=0)
    general_vlm_pages: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = ()


def compiler_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def paper_package_id(
    *,
    source_pdf_sha256: str,
    compiler_identity: str,
    compiler_version: str,
    compiler_config_sha256: str,
) -> str:
    _validate_sha256(source_pdf_sha256)
    _validate_sha256(compiler_config_sha256)
    payload = "\n".join(
        (
            f"paper-package-v{PAPER_PACKAGE_SCHEMA_VERSION}",
            source_pdf_sha256,
            compiler_identity,
            compiler_version,
            compiler_config_sha256,
        )
    ).encode("utf-8")
    return f"paper-package:v{PAPER_PACKAGE_SCHEMA_VERSION}:sha256:{hashlib.sha256(payload).hexdigest()}"


@runtime_checkable
class DocumentCompilerAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def available(self) -> tuple[bool, str | None]:
        """Return whether the adapter can run in the current environment."""

    def compile(
        self,
        *,
        source_pdf_path: str,
        source_artifact_id: str,
        source_pdf_sha256: str,
        output_directory: str,
        config: dict[str, Any] | None = None,
    ) -> PaperPackage:
        """Compile one authoritative PDF into paper-package.v1 without mutating research state."""
