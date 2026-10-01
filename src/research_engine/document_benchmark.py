from __future__ import annotations

import hashlib
import json
import re
import resource
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader

from .document_package import (
    BlockKind,
    DerivationStep,
    DocumentBenchmarkCase,
    DocumentBenchmarkResult,
    DocumentBlock,
    OracleImageMode,
    PaperPackage,
    PaperPackageManifest,
    PdfRegion,
    ProcessingMode,
    QualityMetrics,
    ResourceMetrics,
    compiler_config_sha256,
    paper_package_id,
)


NATIVE_ADAPTER_ID = "native-pypdf"
NATIVE_ADAPTER_VERSION = "1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def _safe_extract_text(page: Any, *, mode: str) -> tuple[str, str, tuple[str, ...]]:
    warnings: list[str] = []
    try:
        text = page.extract_text(extraction_mode=mode) or ""
        return text, mode, tuple(warnings)
    except Exception as exc:
        warnings.append(f"{mode} extraction failed: {type(exc).__name__}: {exc}")
    try:
        text = page.extract_text() or ""
        warnings.append("fell back to plain pypdf extraction")
        return text, "plain", tuple(warnings)
    except Exception as exc:
        warnings.append(f"plain extraction failed: {type(exc).__name__}: {exc}")
        return "", "failed", tuple(warnings)


def _page_confidence(text: str) -> float:
    visible = len(text.strip())
    if visible >= 500:
        return 0.95
    if visible >= 100:
        return 0.8
    if visible > 0:
        return 0.4
    return 0.0


def _canonical_markdown_page(page_number: int, text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return f"<!-- source-page: {page_number} -->\n\n{cleaned}\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class NativePdfAdapter:
    adapter_id = NATIVE_ADAPTER_ID
    adapter_version = NATIVE_ADAPTER_VERSION

    def available(self) -> tuple[bool, str | None]:
        try:
            return True, version("pypdf")
        except Exception as exc:
            return False, f"pypdf unavailable: {exc}"

    def compile(
        self,
        *,
        source_pdf_path: str,
        source_artifact_id: str,
        source_pdf_sha256: str,
        output_directory: str,
        config: dict[str, Any] | None = None,
    ) -> PaperPackage:
        source = Path(source_pdf_path)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        if sha256_file(source) != source_pdf_sha256:
            raise ValueError("source PDF SHA-256 does not match benchmark manifest")

        cfg = {
            "extraction_mode": "layout",
            "fallback_plain": True,
            "block_granularity": "page",
            **(config or {}),
        }
        cfg_sha = compiler_config_sha256(cfg)
        compiler_version = f"adapter:{self.adapter_version}/pypdf:{version('pypdf')}"

        reader = PdfReader(str(source))
        blocks: list[DocumentBlock] = []
        document_pages: list[str] = []
        page_diagnostics: list[dict[str, Any]] = []
        package_warnings: list[str] = []

        for index, page in enumerate(reader.pages, start=1):
            text, used_mode, page_warnings = _safe_extract_text(
                page,
                mode=str(cfg["extraction_mode"]),
            )
            confidence = _page_confidence(text)
            if not text.strip():
                package_warnings.append(f"page {index} produced no native text")
            package_warnings.extend(
                f"page {index}: {warning}" for warning in page_warnings
            )
            step = DerivationStep(
                processor="pypdf",
                processor_version=version("pypdf"),
                mode=ProcessingMode.NATIVE_TEXT,
                confidence=confidence,
                warnings=page_warnings,
                parameters={
                    "requested_extraction_mode": cfg["extraction_mode"],
                    "used_extraction_mode": used_mode,
                },
            )
            block = DocumentBlock(
                id=f"page-{index:04d}",
                kind=BlockKind.PARAGRAPH,
                markdown=text.strip(),
                source_region=PdfRegion(
                    page_number=index,
                    raw_anchor=f"page:{index}",
                ),
                derivation=(step,),
                confidence=confidence,
            )
            blocks.append(block)
            document_pages.append(_canonical_markdown_page(index, text))
            page_diagnostics.append(
                {
                    "page_number": index,
                    "chars": len(text),
                    "lines": len(text.splitlines()),
                    "non_whitespace_chars": len(re.sub(r"\s+", "", text)),
                    "confidence": confidence,
                    "requested_extraction_mode": cfg["extraction_mode"],
                    "used_extraction_mode": used_mode,
                    "warnings": list(page_warnings),
                }
            )

        manifest = PaperPackageManifest(
            package_id=paper_package_id(
                source_pdf_sha256=source_pdf_sha256,
                compiler_identity=self.adapter_id,
                compiler_version=compiler_version,
                compiler_config_sha256=cfg_sha,
            ),
            source_artifact_id=source_artifact_id,
            source_pdf_sha256=source_pdf_sha256,
            source_pdf_byte_size=source.stat().st_size,
            source_page_count=len(reader.pages),
            compiler_identity=self.adapter_id,
            compiler_version=compiler_version,
            compiler_config=cfg,
            compiler_config_sha256=cfg_sha,
            blocks=len(blocks),
            warnings=tuple(package_warnings),
        )
        package = PaperPackage(
            manifest=manifest,
            blocks=tuple(blocks),
        )

        (output / manifest.document_markdown_path).write_text(
            "\n".join(document_pages).rstrip() + "\n",
            encoding="utf-8",
        )
        _write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        _write_json(
            output / manifest.block_index_path,
            [block.model_dump(mode="json") for block in blocks],
        )
        _write_json(
            output / manifest.diagnostics_path,
            {
                "adapter_id": self.adapter_id,
                "adapter_version": self.adapter_version,
                "page_diagnostics": page_diagnostics,
                "warnings": package_warnings,
            },
        )
        return package


def detect_mathpix_image_mode(markdown: str) -> OracleImageMode:
    latex_targets = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        markdown,
    )
    markdown_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    targets = latex_targets + markdown_targets
    if not targets:
        return OracleImageMode.NO_IMAGES
    remote = sum(
        1
        for target in targets
        if target.strip().lower().startswith(("http://", "https://"))
    )
    local = len(targets) - remote
    if remote and local:
        return OracleImageMode.MIXED
    if remote:
        return OracleImageMode.REMOTE_IMAGES
    return OracleImageMode.LOCAL_IMAGES


def _text_tokens(value: str) -> list[str]:
    value = re.sub(r"https?://\S+", " ", value)
    return re.findall(r"[a-z0-9]+", value.lower())


def _multiset_f1(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    a = Counter(left)
    b = Counter(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def oracle_comparison(
    *,
    native_markdown: str,
    oracle_markdown: str,
) -> dict[str, Any]:
    native_tokens = _text_tokens(native_markdown)
    oracle_tokens = _text_tokens(oracle_markdown)
    native_bigrams = list(zip(native_tokens, native_tokens[1:]))
    oracle_bigrams = list(zip(oracle_tokens, oracle_tokens[1:]))
    image_mode = detect_mathpix_image_mode(oracle_markdown)

    return {
        "text_fidelity": _multiset_f1(native_tokens, oracle_tokens),
        "reading_order": _multiset_f1(native_bigrams, oracle_bigrams),
        "mathpix_image_mode": image_mode.value,
        "native_token_count": len(native_tokens),
        "oracle_token_count": len(oracle_tokens),
        "oracle_structural_inventory": {
            "figures": len(re.findall(r"\\begin\{figure\}", oracle_markdown)),
            "tables": len(re.findall(r"\\begin\{table", oracle_markdown)),
            "tabular_blocks": len(re.findall(r"\\begin\{tabular", oracle_markdown)),
            "display_equations": len(re.findall(r"\\\[", oracle_markdown)),
            "algorithm_mentions": len(
                re.findall(r"\bAlgorithm\s+\d+", oracle_markdown)
            ),
            "remote_image_links": len(
                re.findall(r"https?://[^\s})\]]+", oracle_markdown)
            ),
        },
    }


def load_benchmark_cases(path: str | Path) -> list[DocumentBenchmarkCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("benchmark case file must contain a list or {'cases': [...]}")
    return [DocumentBenchmarkCase.model_validate(row) for row in rows]


def _resolve(root: Path, location: str) -> Path:
    path = Path(location)
    if path.is_absolute():
        return path
    return root / path


def run_native_benchmark(
    *,
    case_manifest_path: str | Path,
    bundle_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    bundle = Path(bundle_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    adapter = NativePdfAdapter()

    results: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for case in load_benchmark_cases(case_manifest_path):
        source = _resolve(bundle, case.source_location)
        oracle = (
            _resolve(bundle, case.mathpix_oracle_location)
            if case.mathpix_oracle_location
            else None
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != case.source_pdf_byte_size:
            raise ValueError(f"{case.case_id}: source PDF byte size mismatch")
        if sha256_file(source) != case.source_pdf_sha256:
            raise ValueError(f"{case.case_id}: source PDF SHA-256 mismatch")

        reader = PdfReader(str(source))
        if len(reader.pages) != case.page_count:
            raise ValueError(f"{case.case_id}: source PDF page count mismatch")

        oracle_text: str | None = None
        detected_image_mode: OracleImageMode | None = None
        if oracle is not None:
            if not oracle.is_file():
                raise FileNotFoundError(oracle)
            if sha256_file(oracle) != case.mathpix_oracle_sha256:
                raise ValueError(f"{case.case_id}: Mathpix oracle SHA-256 mismatch")
            oracle_text = oracle.read_text(encoding="utf-8", errors="replace")
            detected_image_mode = detect_mathpix_image_mode(oracle_text)
            if (
                case.mathpix_image_mode is not None
                and detected_image_mode != case.mathpix_image_mode
            ):
                raise ValueError(
                    f"{case.case_id}: Mathpix image mode mismatch: "
                    f"expected {case.mathpix_image_mode.value}, "
                    f"observed {detected_image_mode.value}"
                )

        case_output = output / case.case_id / adapter.adapter_id
        started = time.perf_counter()
        package = adapter.compile(
            source_pdf_path=str(source),
            source_artifact_id=f"source-artifact:sha256:{case.source_pdf_sha256}",
            source_pdf_sha256=case.source_pdf_sha256,
            output_directory=str(case_output),
        )
        elapsed = time.perf_counter() - started
        markdown_path = case_output / package.manifest.document_markdown_path
        native_markdown = markdown_path.read_text(encoding="utf-8")

        comparison: dict[str, Any] = {}
        if oracle_text is not None:
            comparison = oracle_comparison(
                native_markdown=native_markdown,
                oracle_markdown=oracle_text,
            )

        diagnostics_path = case_output / package.manifest.diagnostics_path
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["benchmark_case_id"] = case.case_id
        diagnostics["difficulty_tags"] = list(case.difficulty_tags)
        diagnostics["expected_features"] = case.expected_features
        diagnostics["oracle_comparison"] = comparison
        _write_json(diagnostics_path, diagnostics)

        result = DocumentBenchmarkResult(
            case_id=case.case_id,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            package_id=package.manifest.package_id,
            status="PASS",
            quality=QualityMetrics(
                text_fidelity=comparison.get("text_fidelity"),
                reading_order=comparison.get("reading_order"),
                provenance_accuracy=1.0,
            ),
            resources=ResourceMetrics(
                wall_time_seconds=elapsed,
                pages_per_second=case.page_count / elapsed if elapsed else 0.0,
                peak_rss_bytes=_peak_rss_bytes(),
                peak_vram_bytes=None,
                model_load_seconds=0.0,
            ),
            output_markdown_chars=len(native_markdown),
            output_markdown_tokens_estimate=max(
                1,
                (len(native_markdown) + 3) // 4,
            ),
            native_pages=case.page_count,
            warnings=(
                "native baseline preserves page provenance but does not "
                "reconstruct table/equation/figure structure"
            ,),
        )
        result_row = result.model_dump(mode="json")
        result_row["mathpix_image_mode"] = (
            detected_image_mode.value if detected_image_mode else None
        )
        result_row["oracle_comparison"] = comparison
        _write_json(case_output / "benchmark-result.json", result_row)
        results.append(result_row)

    elapsed_total = time.perf_counter() - total_started
    summary = {
        "schema_version": 1,
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.adapter_version,
        "cases": len(results),
        "total_pages": sum(
            case.page_count for case in load_benchmark_cases(case_manifest_path)
        ),
        "wall_time_seconds": elapsed_total,
        "results": results,
    }
    _write_json(output / "benchmark-summary.json", summary)
    return summary
