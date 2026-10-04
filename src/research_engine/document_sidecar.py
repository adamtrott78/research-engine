from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .artifact_store import ContentAddressedArtifactStore
from .document_docling import _table_grid, normalize_docling_document
from .document_package import PaperPackage


DOCUMENT_SIDECAR_COMPILER_VERSION = "1"


class DocumentSidecarCompiler:
    """Build a derived local document sidecar without changing scientific authority."""

    def __init__(
        self,
        *,
        store,
        artifact_root: str | Path,
        cache_root: str | Path,
        repo_root: str | Path,
        native_runtime: str | Path,
        standard_runtime: str | Path,
        standard_model_cache: str | Path,
        local_inference=None,
        specialized_enabled: bool = True,
        structure_gpu_index: int = 0,
        command_timeout_seconds: float = 900.0,
        command_runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ):
        self.store = store
        self.artifact_store = ContentAddressedArtifactStore(artifact_root)
        self.cache_root = Path(cache_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.native_runtime = Path(native_runtime).resolve()
        self.standard_runtime = Path(standard_runtime).resolve()
        self.standard_model_cache = Path(standard_model_cache).resolve()
        self.local_inference = local_inference
        self.specialized_enabled = bool(specialized_enabled)
        self.structure_gpu_index = int(structure_gpu_index)
        self.command_timeout_seconds = float(command_timeout_seconds)
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        self.command_runner = command_runner or subprocess.run

    def cache_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return (
            self.cache_root
            / "sha256"
            / sha256[:2]
            / sha256[2:4]
            / sha256
        )

    def _record_event(
        self,
        *,
        source_artifact_id: str,
        source_sha256: str,
        status: str,
        cache_path: Path | None,
        wall_time_ms: int,
        routing_summary: dict[str, Any] | None = None,
        local_inference_summary: dict[str, Any] | None = None,
        error_detail: str | None = None,
    ) -> None:
        from psycopg.types.json import Jsonb

        routing = dict(routing_summary or {})
        local = dict(local_inference_summary or {})
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_sidecar_events(
                      source_artifact_id,source_sha256,status,cache_path,
                      compiler_version,native_status,standard_status,
                      formula_pages,fast_empty_tables,accurate_pages,
                      paddle_fallbacks,wall_time_ms,routing_summary,
                      local_inference_summary,error_detail
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        source_artifact_id,
                        source_sha256,
                        status,
                        str(cache_path) if cache_path is not None else None,
                        DOCUMENT_SIDECAR_COMPILER_VERSION,
                        routing.get("native_status"),
                        routing.get("standard_status"),
                        int(routing.get("formula_pages") or 0),
                        int(routing.get("fast_empty_tables") or 0),
                        int(routing.get("accurate_pages") or 0),
                        int(routing.get("paddle_fallbacks") or 0),
                        max(0, int(wall_time_ms)),
                        Jsonb(routing),
                        Jsonb(local),
                        error_detail,
                    ),
                )
            conn.commit()

    def _run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        stdout_path: Path,
    ) -> None:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = self.command_runner(
            command,
            cwd=self.repo_root,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            (result.stdout or "")
            + (("\n=== STDERR ===\n" + result.stderr) if result.stderr else ""),
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"document sidecar command failed ({result.returncode}): "
                + " ".join(command)
            )

    @staticmethod
    def _package_from(path: Path) -> PaperPackage:
        return PaperPackage.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _single_runtime_json(output_dir: Path) -> Path:
        candidates = sorted(output_dir.glob("*.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one runtime JSON in {output_dir}, "
                f"found {len(candidates)}: {[path.name for path in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _empty_fast_tables(
        package: PaperPackage,
        package_root: Path,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for asset in package.tables:
            grid: list[list[str]] = []
            if asset.json_path:
                payload_path = package_root / asset.json_path
                if payload_path.is_file():
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                    grid = _table_grid(payload)
            if grid:
                continue
            rows.append(
                {
                    "table_asset_id": asset.id,
                    "page_number": asset.source_region.page_number,
                    "pdf_bbox": (
                        list(asset.source_region.pdf_bbox)
                        if asset.source_region.pdf_bbox is not None
                        else None
                    ),
                }
            )
        return rows

    @staticmethod
    def _accurate_has_nonempty_table(raw_json: Path) -> bool:
        if not raw_json.is_file():
            return False
        payload = json.loads(raw_json.read_text(encoding="utf-8"))
        for item in payload.get("tables") or []:
            if isinstance(item, dict) and _table_grid(item):
                return True
        return False

    def _build_sidecar(
        self,
        *,
        source_artifact_id: str,
        source_sha256: str,
        source_pdf: Path,
        work_root: Path,
    ) -> dict[str, Any]:
        native_python = self.native_runtime / "bin" / "python"
        standard_python = self.standard_runtime / "bin" / "python"
        native_script = (
            self.repo_root
            / "benchmarks"
            / "document_intelligence"
            / "runtimes"
            / "docling_native.py"
        )
        standard_script = (
            self.repo_root
            / "benchmarks"
            / "document_intelligence"
            / "runtimes"
            / "docling_standard_no_ocr.py"
        )
        accurate_script = (
            self.repo_root
            / "benchmarks"
            / "document_intelligence"
            / "runtimes"
            / "docling_tableformer_accurate.py"
        )
        for path in (
            native_python,
            standard_python,
            native_script,
            standard_script,
            accurate_script,
            self.standard_model_cache,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        source_link = work_root / "source.pdf"
        source_link.symlink_to(source_pdf.resolve())

        raw_native = work_root / "raw" / "native"
        raw_standard = work_root / "raw" / "standard"
        native_package_root = work_root / "packages" / "native" / "paper-package-v1"
        standard_package_root = (
            work_root / "packages" / "standard" / "paper-package-v1"
        )
        logs = work_root / "logs"

        self._run(
            [str(native_python), str(native_script), str(source_link), str(raw_native)],
            stdout_path=logs / "docling-native.log",
        )
        standard_env = {
            "CUDA_VISIBLE_DEVICES": str(self.structure_gpu_index),
            "RESEARCH_ENGINE_DOCLING_DEVICE": "cuda",
            "RESEARCH_ENGINE_DOCLING_ARTIFACTS": str(self.standard_model_cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        self._run(
            [
                str(standard_python),
                str(standard_script),
                str(source_link),
                str(raw_standard),
            ],
            env=standard_env,
            stdout_path=logs / "docling-standard-no-ocr.log",
        )

        native_package = normalize_docling_document(
            docling_json_path=self._single_runtime_json(raw_native),
            source_pdf_path=source_pdf,
            source_artifact_id=source_artifact_id,
            source_pdf_sha256=source_sha256,
            output_directory=native_package_root,
            docling_version="2.131.0",
            docling_mode="native",
        )
        standard_package = normalize_docling_document(
            docling_json_path=self._single_runtime_json(raw_standard),
            source_pdf_path=source_pdf,
            source_artifact_id=source_artifact_id,
            source_pdf_sha256=source_sha256,
            output_directory=standard_package_root,
            docling_version="2.131.0",
            docling_mode="standard_no_ocr",
        )

        formula_pages = sorted(
            {
                int(asset.source_region.page_number)
                for asset in standard_package.equations
            }
        )
        fast_empty_tables = self._empty_fast_tables(
            standard_package,
            standard_package_root,
        )

        formula_results: list[dict[str, Any]] = []
        accurate_results: list[dict[str, Any]] = []
        paddle_results: list[dict[str, Any]] = []
        broker_before = (
            self.local_inference.health()
            if self.local_inference is not None
            else {"enabled": False}
        )

        if self.specialized_enabled and self.local_inference is not None:
            for page_number in formula_pages:
                output_dir = (
                    work_root
                    / "specialized"
                    / "formula"
                    / f"page-{page_number:04d}"
                )
                formula_results.append(
                    self.local_inference.formula_page(
                        source_pdf=source_link,
                        page_number=page_number,
                        output_dir=output_dir,
                    )
                )

            by_page: dict[int, list[dict[str, Any]]] = {}
            for target in fast_empty_tables:
                by_page.setdefault(int(target["page_number"]), []).append(target)

            for page_number, targets in sorted(by_page.items()):
                output_dir = (
                    work_root
                    / "specialized"
                    / "table-accurate"
                    / f"page-{page_number:04d}"
                )
                self._run(
                    [
                        str(standard_python),
                        str(accurate_script),
                        str(source_link),
                        str(output_dir),
                        "--page",
                        str(page_number),
                    ],
                    env=standard_env,
                    stdout_path=(
                        logs / f"table-accurate-page-{page_number:04d}.log"
                    ),
                )
                accurate_nonempty = self._accurate_has_nonempty_table(
                    self._single_runtime_json(output_dir)
                )
                accurate_results.append(
                    {
                        "page_number": page_number,
                        "targets": len(targets),
                        "nonempty_table_recovered": accurate_nonempty,
                    }
                )
                if accurate_nonempty:
                    continue
                for target in targets:
                    bbox = target.get("pdf_bbox")
                    if bbox is None:
                        continue
                    paddle_results.append(
                        self.local_inference.table_crop(
                            source_pdf=source_pdf,
                            page_number=page_number,
                            pdf_bbox=bbox,
                            output_dir=(
                                work_root
                                / "specialized"
                                / "paddle-table"
                                / str(target["table_asset_id"])
                            ),
                        )
                    )

        broker_after = (
            self.local_inference.health()
            if self.local_inference is not None
            else {"enabled": False}
        )
        routing = {
            "authority": "AUTHORITATIVE_PDF",
            "semantic_baseline": "AUTHORITATIVE_PDF_HYDRATOR",
            "native_status": "COMPLETED",
            "standard_status": "COMPLETED",
            "native_blocks": len(native_package.blocks),
            "standard_blocks": len(standard_package.blocks),
            "formula_pages": len(formula_pages),
            "formula_page_numbers": formula_pages,
            "fast_empty_tables": len(fast_empty_tables),
            "fast_empty_table_targets": fast_empty_tables,
            "accurate_pages": len(accurate_results),
            "accurate_results": accurate_results,
            "paddle_fallbacks": len(paddle_results),
            "specialized_enabled": self.specialized_enabled,
        }
        local_summary = {
            "broker_before": broker_before,
            "broker_after": broker_after,
            "formula_requests": len(formula_results),
            "table_requests": len(paddle_results),
        }
        return {
            "routing_summary": routing,
            "local_inference_summary": local_summary,
            "native_package_path": str(native_package_root / "package.json"),
            "standard_package_path": str(standard_package_root / "package.json"),
        }

    def compile(
        self,
        *,
        source_artifact_id: str,
        source_sha256: str,
        source_pdf: str | Path,
    ) -> dict[str, Any]:
        source = Path(source_pdf).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = self.artifact_store.path_for_hash(source_sha256).resolve()
        if source != expected:
            raise ValueError(
                "document sidecar source must be the authoritative "
                "content-addressed artifact path"
            )

        started = time.monotonic()
        final = self.cache_path(source_sha256)
        receipt_path = final / "receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            wall_ms = max(0, int((time.monotonic() - started) * 1000))
            self._record_event(
                source_artifact_id=source_artifact_id,
                source_sha256=source_sha256,
                status="CACHE_HIT",
                cache_path=final,
                wall_time_ms=wall_ms,
                routing_summary=receipt.get("routing_summary") or {},
                local_inference_summary={
                    "cache_hit": True,
                    "formula_requests": 0,
                    "table_requests": 0,
                    "broker_current": (
                        self.local_inference.health()
                        if self.local_inference is not None
                        else {"enabled": False}
                    ),
                },
            )
            return {
                **receipt,
                "status": "CACHE_HIT",
                "cache_hit": True,
                "cache_path": str(final),
                "wall_time_ms": wall_ms,
            }

        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        work_root = Path(
            tempfile.mkdtemp(
                prefix=f".building-{source_sha256[:12]}-",
                dir=final.parent,
            )
        )
        try:
            built = self._build_sidecar(
                source_artifact_id=source_artifact_id,
                source_sha256=source_sha256,
                source_pdf=source,
                work_root=work_root,
            )
            receipt = {
                "schema_version": 1,
                "authority": "DERIVED_SIDECAR_ONLY",
                "semantic_baseline": "AUTHORITATIVE_PDF_HYDRATOR",
                "compiler_version": DOCUMENT_SIDECAR_COMPILER_VERSION,
                "source_artifact_id": source_artifact_id,
                "source_sha256": source_sha256,
                "routing_summary": built["routing_summary"],
                "local_inference_summary": built["local_inference_summary"],
                "native_package_relative_path": (
                    "packages/native/paper-package-v1/package.json"
                ),
                "standard_package_relative_path": (
                    "packages/standard/paper-package-v1/package.json"
                ),
            }
            (work_root / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(work_root, final)
            wall_ms = max(0, int((time.monotonic() - started) * 1000))
            self._record_event(
                source_artifact_id=source_artifact_id,
                source_sha256=source_sha256,
                status="COMPLETED",
                cache_path=final,
                wall_time_ms=wall_ms,
                routing_summary=receipt["routing_summary"],
                local_inference_summary=receipt["local_inference_summary"],
            )
            return {
                **receipt,
                "status": "COMPLETED",
                "cache_hit": False,
                "cache_path": str(final),
                "wall_time_ms": wall_ms,
            }
        except Exception as exc:
            wall_ms = max(0, int((time.monotonic() - started) * 1000))
            try:
                self._record_event(
                    source_artifact_id=source_artifact_id,
                    source_sha256=source_sha256,
                    status="FAILED_DERIVED_SIDECAR",
                    cache_path=None,
                    wall_time_ms=wall_ms,
                    error_detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if work_root.exists():
                    shutil.rmtree(work_root, ignore_errors=True)
            raise

    def compile_artifact(self, source_artifact_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,sha256,media_type,format
                    FROM source_artifacts
                    WHERE id=%s
                    """,
                    (source_artifact_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"source artifact not found: {source_artifact_id}")
        if str(row[2] or "").lower() != "application/pdf" and str(
            row[3] or ""
        ).lower() != "pdf":
            raise ValueError("document sidecar currently supports PDF artifacts only")
        sha256 = str(row[1])
        return self.compile(
            source_artifact_id=str(row[0]),
            source_sha256=sha256,
            source_pdf=self.artifact_store.path_for_hash(sha256),
        )


def document_sidecar_compiler_from_env(
    *,
    store,
    artifact_root: str | Path,
    local_inference=None,
    require_enabled: bool = False,
) -> DocumentSidecarCompiler | None:
    enabled = os.getenv("RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED", "0").strip()
    if enabled not in {"0", "1"}:
        raise ValueError("RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED must be 0 or 1")
    if enabled == "0":
        if require_enabled:
            raise RuntimeError("document sidecar compiler is disabled")
        return None

    specialized = os.getenv(
        "RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED",
        "1",
    ).strip()
    if specialized not in {"0", "1"}:
        raise ValueError(
            "RESEARCH_ENGINE_DOCUMENT_SIDECAR_SPECIALIZED_ENABLED must be 0 or 1"
        )

    repo_root = Path(__file__).resolve().parents[2]
    adam = Path.home() / "adamArchives" / "Adam"
    state = adam / ".research-engine"
    return DocumentSidecarCompiler(
        store=store,
        artifact_root=artifact_root,
        cache_root=os.getenv(
            "RESEARCH_ENGINE_DOCUMENT_SIDECAR_ROOT",
            str(state / "document-sidecars"),
        ),
        repo_root=repo_root,
        native_runtime=os.getenv(
            "RESEARCH_ENGINE_DOCLING_NATIVE_RUNTIME",
            str(adam / ".document-runtimes" / "docling-native"),
        ),
        standard_runtime=os.getenv(
            "RESEARCH_ENGINE_DOCLING_STANDARD_RUNTIME",
            str(state / "document-runtimes" / "docling-standard-no-ocr"),
        ),
        standard_model_cache=os.getenv(
            "RESEARCH_ENGINE_DOCLING_STANDARD_MODEL_CACHE",
            str(state / "document-models" / "docling-standard-no-ocr-2.131.0"),
        ),
        local_inference=local_inference,
        specialized_enabled=specialized == "1",
        structure_gpu_index=int(
            os.getenv("RESEARCH_ENGINE_DOCUMENT_STRUCTURE_GPU_INDEX", "1")
        ),
        command_timeout_seconds=float(
            os.getenv("RESEARCH_ENGINE_DOCUMENT_SIDECAR_TIMEOUT_SECONDS", "900")
        ),
    )


def select_smoke_artifact(
    store,
    compiler: DocumentSidecarCompiler | None = None,
) -> str:
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,sha256
                FROM source_artifacts
                WHERE lower(coalesce(media_type,''))='application/pdf'
                   OR lower(coalesce(format,''))='pdf'
                ORDER BY id
                """
            )
            rows = cur.fetchall()
    if not rows:
        raise ValueError("no PDF source artifact is available for sidecar smoke")
    if compiler is not None:
        for artifact_id, sha256 in rows:
            if not (compiler.cache_path(str(sha256)) / "receipt.json").is_file():
                return str(artifact_id)
    return str(rows[0][0])
