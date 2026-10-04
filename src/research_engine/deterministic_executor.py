from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .arxiv import ArxivProvider
from .artifact_store import ContentAddressedArtifactStore
from .orchestration import (
    ExecutorTier,
    JobExecutionError,
    JobType,
    ResearchJob,
)


def arxiv_id_from_uri(uri: str) -> str | None:
    parsed = urlparse(uri)
    if parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        return None
    path = parsed.path.strip("/")
    for prefix in ("abs/", "pdf/"):
        if path.startswith(prefix):
            value = path[len(prefix):]
            if value.endswith(".pdf"):
                value = value[:-4]
            return value or None
    return None


class DeterministicResearchExecutor:
    """Production deterministic executor for source acquisition and checkpoints."""

    def __init__(
        self,
        *,
        store,
        artifact_root: str | Path,
        arxiv_provider: Any | None = None,
        local_inference: Any | None = None,
        document_sidecar: Any | None = None,
    ):
        self.store = store
        self.artifact_store = ContentAddressedArtifactStore(artifact_root)
        self.arxiv_provider = arxiv_provider or ArxivProvider(self.artifact_store)
        self.local_inference = local_inference
        self.document_sidecar = document_sidecar

    def execute(self, job: ResearchJob) -> dict[str, Any]:
        if job.executor_tier != ExecutorTier.DETERMINISTIC:
            raise JobExecutionError(
                f"deterministic executor cannot serve tier {job.executor_tier}",
                code="TIER_MISMATCH",
                retryable=False,
            )
        if job.job_type == JobType.ACQUIRE:
            return self._acquire(job)
        if job.job_type == JobType.CHECKPOINT:
            return self._checkpoint(job)
        raise JobExecutionError(
            f"unsupported deterministic job type {job.job_type}",
            code="UNSUPPORTED_JOB_TYPE",
            retryable=False,
        )

    def _acquire(self, job: ResearchJob) -> dict[str, Any]:
        item = job.payload.get("item") or {}
        work_id = str(item.get("id") or "")
        title = str(item.get("title") or "")
        source_uri = str(item.get("source_uri") or "")
        if not work_id or not title or not source_uri:
            raise JobExecutionError(
                "acquisition item requires id, title, and source_uri",
                code="INVALID_ACQUISITION_ITEM",
                retryable=False,
            )

        arxiv_id = arxiv_id_from_uri(source_uri)
        if arxiv_id is None:
            raise JobExecutionError(
                f"no deterministic acquisition provider for {source_uri}",
                code="UNSUPPORTED_SOURCE_PROVIDER",
                retryable=False,
                detail={"source_uri": source_uri},
            )

        try:
            acquired = self.arxiv_provider.acquire_pdf(arxiv_id)
        except ValueError as exc:
            raise JobExecutionError(
                str(exc),
                code="INVALID_SOURCE_BYTES",
                retryable=False,
                detail={"source_uri": source_uri, "provider": "arxiv"},
            ) from exc
        except Exception as exc:
            raise JobExecutionError(
                f"arXiv acquisition failed: {exc}",
                code="ACQUISITION_FAILED",
                retryable=True,
                detail={"source_uri": source_uri, "provider": "arxiv"},
            ) from exc

        version_id = f"{work_id}:arxiv:{acquired.arxiv_id}"
        artifact_id = acquired.receipt.artifact_id
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title FROM works WHERE id=%s",
                    (work_id,),
                )
                existing_work = cur.fetchone()
                if existing_work is None:
                    cur.execute(
                        """
                        INSERT INTO works(id,title,attributes)
                        VALUES (%s,%s,%s)
                        """,
                        (
                            work_id,
                            title,
                            Jsonb(
                                {
                                    "source": "agentic-acquisition",
                                    "discovery_rationale": item.get("rationale", ""),
                                }
                            ),
                        ),
                    )
                elif existing_work[0] != title:
                    raise JobExecutionError(
                        f"work id {work_id} already exists with a different title",
                        code="WORK_IDENTITY_CONFLICT",
                        retryable=False,
                    )

                cur.execute(
                    "SELECT work_id, version_label FROM work_versions WHERE id=%s",
                    (version_id,),
                )
                existing_version = cur.fetchone()
                if existing_version is None:
                    cur.execute(
                        """
                        INSERT INTO work_versions(id,work_id,version_label,attributes)
                        VALUES (%s,%s,%s,%s)
                        """,
                        (
                            version_id,
                            work_id,
                            acquired.arxiv_id,
                            Jsonb({"provider": "arxiv"}),
                        ),
                    )
                elif tuple(existing_version) != (work_id, acquired.arxiv_id):
                    raise JobExecutionError(
                        f"work version id {version_id} conflicts with existing identity",
                        code="WORK_VERSION_IDENTITY_CONFLICT",
                        retryable=False,
                    )

                cur.execute(
                    """
                    SELECT work_version_id, sha256, byte_size, media_type, format,
                           access_class, artifact_uri
                    FROM source_artifacts WHERE id=%s
                    """,
                    (artifact_id,),
                )
                existing_artifact = cur.fetchone()
                expected_artifact = (
                    version_id,
                    acquired.receipt.sha256,
                    acquired.receipt.byte_size,
                    acquired.media_type,
                    "pdf",
                    "OPEN_ACCESS",
                    acquired.resolved_url,
                )
                if existing_artifact is None:
                    cur.execute(
                        """
                        INSERT INTO source_artifacts(
                          id,work_version_id,sha256,byte_size,media_type,format,
                          access_class,artifact_uri,metadata
                        )
                        VALUES (%s,%s,%s,%s,%s,'pdf','OPEN_ACCESS',%s,%s)
                        """,
                        (
                            artifact_id,
                            version_id,
                            acquired.receipt.sha256,
                            acquired.receipt.byte_size,
                            acquired.media_type,
                            acquired.resolved_url,
                            Jsonb(
                                {
                                    "provider": "arxiv",
                                    "versioned_pdf_url": acquired.versioned_pdf_url,
                                }
                            ),
                        ),
                    )
                elif tuple(existing_artifact) != expected_artifact:
                    raise JobExecutionError(
                        f"source artifact {artifact_id} conflicts with existing identity",
                        code="SOURCE_ARTIFACT_CONFLICT",
                        retryable=False,
                    )

                cur.execute(
                    """
                    INSERT INTO external_identifiers(work_id,work_version_id,scheme,value)
                    VALUES (%s,%s,'arxiv',%s)
                    ON CONFLICT(scheme,value) DO NOTHING
                    """,
                    (work_id, version_id, acquired.arxiv_id),
                )
            conn.commit()

        result = {
            "item_id": work_id,
            "provider": "arxiv",
            "arxiv_id": acquired.arxiv_id,
            "source_uri": source_uri,
            "resolved_url": acquired.resolved_url,
            "work_version_id": version_id,
            "artifact_id": artifact_id,
            "sha256": acquired.receipt.sha256,
            "byte_size": acquired.receipt.byte_size,
            "media_type": acquired.media_type,
            "artifact_path": str(acquired.receipt.path),
        }
        if self.document_sidecar is not None:
            try:
                sidecar = self.document_sidecar.compile(
                    source_artifact_id=artifact_id,
                    source_sha256=acquired.receipt.sha256,
                    source_pdf=acquired.receipt.path,
                )
                result["_document_sidecar"] = sidecar
            except Exception as exc:
                result["_document_sidecar"] = {
                    "status": "FAILED_DERIVED_SIDECAR",
                    "authority": "AUTHORITATIVE_PDF_UNCHANGED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return result

    def _checkpoint(self, job: ResearchJob) -> dict[str, Any]:
        verify_ids = list(job.payload.get("verified_job_ids") or [])
        if not verify_ids:
            raise JobExecutionError(
                "checkpoint requires verified_job_ids",
                code="EMPTY_CHECKPOINT_SCOPE",
                retryable=False,
            )

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id,status,payload
                    FROM research_jobs
                    WHERE id = ANY(%s) AND job_type='VERIFY'
                    ORDER BY id
                    """,
                    (verify_ids,),
                )
                rows = cur.fetchall()
        if len(rows) != len(set(verify_ids)):
            raise JobExecutionError(
                "checkpoint references missing verification jobs",
                code="MISSING_VERIFICATION_JOB",
                retryable=False,
            )
        if any(row[1] != "COMPLETED" for row in rows):
            raise JobExecutionError(
                "checkpoint references incomplete verification jobs",
                code="INCOMPLETE_VERIFICATION_SCOPE",
                retryable=True,
            )

        work_ids = sorted({str(row[2]["item"]["id"]) for row in rows})
        try:
            checkpoint = self.store.create_knowledge_checkpoint(work_ids)
        except ValueError as exc:
            raise JobExecutionError(
                str(exc),
                code="CHECKPOINT_BUILD_FAILED",
                retryable=False,
                detail={"work_ids": work_ids},
            ) from exc

        return {
            "knowledge_checkpoint_id": checkpoint.id,
            "fingerprint": checkpoint.fingerprint,
            "scope": checkpoint.scope,
            "record_counts": checkpoint.record_counts,
        }
