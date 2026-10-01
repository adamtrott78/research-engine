from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .artifact_store import ContentAddressedArtifactStore
from .orchestration import (
    JobExecutionError,
    JobType,
    ResearchJob,
    bounded_gap_candidates,
)


def _without_execution(result: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in result.items() if k != "_execution"}


class PdfPageTextExtractor:
    def __init__(self, max_chars: int = 120_000):
        self.max_chars = max_chars

    def __call__(self, path: Path) -> list[dict[str, Any]]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
        except Exception as exc:
            raise JobExecutionError(
                f"failed to parse source PDF: {exc}",
                code="SOURCE_PARSE_FAILED",
                retryable=False,
                detail={"path": str(path)},
            ) from exc

        pages: list[dict[str, Any]] = []
        remaining = self.max_chars
        for index, page in enumerate(reader.pages, start=1):
            if remaining <= 0:
                break
            text = page.extract_text() or ""
            text = text[:remaining]
            pages.append({"page_number": index, "text": text})
            remaining -= len(text)
        if not pages:
            raise JobExecutionError(
                "source PDF yielded no readable pages",
                code="EMPTY_SOURCE_TEXT",
                retryable=False,
                detail={"path": str(path)},
            )
        return pages


class ResearchContextHydrator:
    """Deterministically expands job payloads from authoritative source/database state."""

    def __init__(
        self,
        *,
        store,
        artifact_root: str | Path,
        text_extractor: Callable[[Path], list[dict[str, Any]]] | None = None,
    ):
        self.store = store
        self.artifact_store = ContentAddressedArtifactStore(artifact_root)
        self.text_extractor = text_extractor or PdfPageTextExtractor()

    def _artifact_context(self, artifact_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, work_version_id, sha256, byte_size, media_type,
                           format, access_class, artifact_uri, metadata
                    FROM source_artifacts WHERE id=%s
                    """,
                    (artifact_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise JobExecutionError(
                        f"source artifact not found: {artifact_id}",
                        code="MISSING_SOURCE_ARTIFACT",
                        retryable=False,
                    )
                cols = [d.name for d in cur.description]
                artifact = dict(zip(cols, row, strict=True))

        path = self.artifact_store.path_for_hash(artifact["sha256"])
        if not path.exists():
            raise JobExecutionError(
                f"source bytes are not present in artifact store: {artifact_id}",
                code="MISSING_SOURCE_BYTES",
                retryable=True,
                detail={"sha256": artifact["sha256"], "path": str(path)},
            )
        return {
            "source_artifact": artifact,
            "pages": self.text_extractor(path),
        }

    def _evidence(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.evidence_type, e.work_id, e.work_version_id,
                           e.source_artifact_id, e.source_level, e.raw_source_form,
                           e.normalized_form, e.verification_state, e.attributes,
                           l.page_number, l.raw_anchor, l.locator_quality,
                           a.sha256 AS source_sha256
                    FROM evidence_objects e
                    LEFT JOIN source_locators l ON l.id=e.source_locator_id
                    LEFT JOIN source_artifacts a ON a.id=e.source_artifact_id
                    WHERE e.id = ANY(%s)
                    ORDER BY e.id
                    """,
                    (ids,),
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def _checkpoint_context(self, checkpoint_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, fingerprint, scope, manifest, record_counts
                    FROM knowledge_checkpoints WHERE id=%s
                    """,
                    (checkpoint_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise JobExecutionError(
                        f"knowledge checkpoint not found: {checkpoint_id}",
                        code="MISSING_KNOWLEDGE_CHECKPOINT",
                        retryable=False,
                    )
        manifest = row[3]
        evidence_ids = [
            item["id"] for item in manifest.get("evidence_objects", [])
        ]
        return {
            "id": row[0],
            "fingerprint": row[1],
            "scope": row[2],
            "record_counts": row[4],
            "evidence": self._evidence(evidence_ids),
        }

    def _run_checkpoint_id(self, run_id: str) -> str:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT knowledge_checkpoint_id FROM research_runs WHERE id=%s",
                    (run_id,),
                )
                row = cur.fetchone()
        if row is None or not row[0]:
            raise JobExecutionError(
                f"research run has no knowledge checkpoint: {run_id}",
                code="RUN_CHECKPOINT_UNAVAILABLE",
                retryable=True,
            )
        return str(row[0])

    def hydrate(self, job: ResearchJob) -> ResearchJob:
        payload = copy.deepcopy(job.payload)
        context: dict[str, Any] = {}

        if job.job_type == JobType.EXTRACT:
            acquisition = payload.get("acquisition") or {}
            artifact_id = acquisition.get("artifact_id")
            if not artifact_id:
                raise JobExecutionError(
                    "EXTRACT job lacks acquisition artifact_id",
                    code="MISSING_ACQUISITION_CONTEXT",
                    retryable=False,
                )
            context.update(self._artifact_context(str(artifact_id)))

        elif job.job_type == JobType.VERIFY:
            extraction = payload.get("extraction") or {}
            materialized = extraction.get("_materialization") or {}
            artifact_id = materialized.get("artifact_id")
            if not artifact_id:
                raise JobExecutionError(
                    "VERIFY job lacks materialized artifact identity",
                    code="MISSING_EXTRACTION_CONTEXT",
                    retryable=False,
                )
            context.update(self._artifact_context(str(artifact_id)))
            evidence_ids = list(materialized.get("evidence_ids") or [])
            context["authoritative_evidence"] = self._evidence(evidence_ids)

        elif job.job_type == JobType.SYNTHESIZE:
            checkpoint_id = str(payload.get("knowledge_checkpoint_id") or "")
            if not checkpoint_id:
                raise JobExecutionError(
                    "SYNTHESIZE job lacks knowledge_checkpoint_id",
                    code="MISSING_KNOWLEDGE_CHECKPOINT",
                    retryable=False,
                )
            context["knowledge_checkpoint"] = self._checkpoint_context(checkpoint_id)

        elif job.job_type in {JobType.GAP_GENERATE, JobType.NOVELTY_ARBITRATE}:
            checkpoint_id = self._run_checkpoint_id(job.run_id)
            context["knowledge_checkpoint"] = self._checkpoint_context(checkpoint_id)
            if job.job_type == JobType.NOVELTY_ARBITRATE:
                gap = payload.get("gap") or {}
                gap_id = gap.get("id")
                if gap_id:
                    with self.store.connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                SELECT id, statement, gap_type, motivating_evidence_ids,
                                       novelty_status, rationale, human_review_state
                                FROM gap_candidates WHERE id=%s
                                """,
                                (gap_id,),
                            )
                            row = cur.fetchone()
                            if row:
                                cols = [d.name for d in cur.description]
                                context["gap_candidate"] = dict(
                                    zip(cols, row, strict=True)
                                )

        if context:
            payload["_research_context"] = context
        return replace(job, payload=payload)


class SemanticResultMaterializer:
    """Crash-safe bridge from structured model output to authoritative research state."""

    def __init__(self, *, store):
        self.store = store

    def cached(self, job_id: str) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, semantic_result, materialized_result
                    FROM semantic_materializations WHERE job_id=%s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
        return None if row is None else (row[0], row[1], row[2])

    def _ensure_pending(
        self, job: ResearchJob, semantic_result: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO semantic_materializations(
                      job_id, job_type, status, semantic_result
                    )
                    VALUES (%s,%s,'PENDING',%s)
                    ON CONFLICT(job_id) DO NOTHING
                    """,
                    (job.id, str(job.job_type), Jsonb(semantic_result)),
                )
                cur.execute(
                    """
                    SELECT status, semantic_result, materialized_result
                    FROM semantic_materializations WHERE job_id=%s
                    """,
                    (job.id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"failed to seed semantic materialization {job.id}")
            conn.commit()
        return row[0], row[1], row[2]

    @staticmethod
    def _compose(
        semantic_result: dict[str, Any],
        materialized: dict[str, Any] | None,
    ) -> dict[str, Any]:
        materialized = materialized or {}
        base = copy.deepcopy(
            materialized.get("canonical_result") or semantic_result
        )
        if "_execution" not in base and "_execution" in semantic_result:
            base["_execution"] = copy.deepcopy(semantic_result["_execution"])
        details = {
            k: copy.deepcopy(v)
            for k, v in materialized.items()
            if k != "canonical_result"
        }
        if details:
            base["_materialization"] = details
        return base

    def materialize(
        self, job: ResearchJob, semantic_result: dict[str, Any]
    ) -> dict[str, Any]:
        status, canonical_semantic, existing_materialized = self._ensure_pending(
            job, semantic_result
        )
        if status == "COMPLETED":
            return self._compose(canonical_semantic, existing_materialized)

        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, semantic_result, materialized_result
                    FROM semantic_materializations
                    WHERE job_id=%s FOR UPDATE
                    """,
                    (job.id,),
                )
                locked = cur.fetchone()
                if locked[0] == "COMPLETED":
                    return self._compose(locked[1], locked[2])
                canonical_semantic = locked[1]

                if job.job_type == JobType.EXTRACT:
                    materialized = self._materialize_extract(
                        cur, job, canonical_semantic
                    )
                elif job.job_type == JobType.VERIFY:
                    materialized = self._materialize_verify(
                        cur, job, canonical_semantic
                    )
                elif job.job_type == JobType.SYNTHESIZE:
                    materialized = self._materialize_synthesis(
                        cur, job, canonical_semantic
                    )
                elif job.job_type == JobType.GAP_GENERATE:
                    materialized = self._materialize_gaps(
                        cur, job, canonical_semantic
                    )
                elif job.job_type == JobType.NOVELTY_ARBITRATE:
                    materialized = self._materialize_arbitration(
                        cur, job, canonical_semantic
                    )
                else:
                    materialized = {}

                cur.execute(
                    """
                    UPDATE semantic_materializations
                    SET status='COMPLETED', materialized_result=%s, updated_at=now()
                    WHERE job_id=%s
                    """,
                    (Jsonb(materialized), job.id),
                )
            conn.commit()
        return self._compose(canonical_semantic, materialized)

    def _materialize_extract(self, cur, job, result):
        from psycopg.types.json import Jsonb

        acquisition = job.payload["acquisition"]
        item = job.payload["item"]
        work_id = str(item["id"])
        artifact_id = str(acquisition["artifact_id"])
        version_id = str(acquisition["work_version_id"])
        run_id = f"extraction:{job.id}"
        execution = result.get("_execution") or {}
        extractor = (
            f"{execution.get('provider','semantic')}:{execution.get('model','unknown')}"
        )

        cur.execute(
            """
            INSERT INTO extraction_runs(
              id,work_version_id,source_artifact_id,source_level,
              extractor_identity,status,completed_at
            )
            VALUES (%s,%s,%s,'FULLTEXT_PDF_TEXT',%s,'COMPLETED',now())
            ON CONFLICT(id) DO NOTHING
            """,
            (run_id, version_id, artifact_id, extractor),
        )

        ids: list[str] = []
        id_map: dict[str, str] = {}
        for index, evidence in enumerate(result.get("evidence", []), start=1):
            evidence_id = f"evidence:{job.id}:{index:04d}"
            locator_id = f"locator:{job.id}:{index:04d}"
            ids.append(evidence_id)
            id_map[str(evidence.get("id", index))] = evidence_id
            raw = str(evidence["raw_source_form"])
            page_number = int(evidence["page_number"])

            cur.execute(
                """
                INSERT INTO source_locators(
                  id,source_artifact_id,source_level,locator_quality,
                  page_number,raw_anchor,attributes
                )
                VALUES (%s,%s,'FULLTEXT_PDF_TEXT','PAGE_QUOTE',%s,%s,%s)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    locator_id,
                    artifact_id,
                    page_number,
                    raw,
                    Jsonb({"semantic_job_id": job.id}),
                ),
            )
            cur.execute(
                """
                INSERT INTO evidence_objects(
                  id,evidence_type,work_id,work_version_id,source_artifact_id,
                  source_locator_id,source_level,raw_source_form,normalized_form,
                  extraction_run_id,extractor_identity,verification_state,attributes
                )
                VALUES (%s,%s,%s,%s,%s,%s,'FULLTEXT_PDF_TEXT',%s,%s,%s,%s,
                        'UNVERIFIED',%s)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    evidence_id,
                    str(evidence["type"]),
                    work_id,
                    version_id,
                    artifact_id,
                    locator_id,
                    raw,
                    str(evidence["normalized_form"]),
                    run_id,
                    extractor,
                    Jsonb(
                        {
                            "model_evidence_id": evidence.get("id"),
                            "confidence": evidence.get("confidence"),
                            "semantic_job_id": job.id,
                        }
                    ),
                ),
            )

        return {
            "extraction_run_id": run_id,
            "artifact_id": artifact_id,
            "work_version_id": version_id,
            "evidence_ids": ids,
            "evidence_id_map": id_map,
        }

    def _materialize_verify(self, cur, job, result):
        from psycopg.types.json import Jsonb

        extraction = job.payload.get("extraction") or {}
        materialized = extraction.get("_materialization") or {}
        id_map = materialized.get("evidence_id_map") or {}
        known = set(materialized.get("evidence_ids") or [])
        accepted: list[str] = []
        for raw_id in result.get("accepted_evidence_ids", []):
            evidence_id = id_map.get(str(raw_id), str(raw_id))
            if evidence_id not in known:
                raise JobExecutionError(
                    f"verification referenced evidence outside extraction: {raw_id}",
                    code="INVALID_VERIFICATION_REFERENCE",
                    retryable=False,
                )
            accepted.append(evidence_id)

        execution = result.get("_execution") or {}
        verifier = f"{execution.get('provider','semantic')}:{execution.get('model','unknown')}"
        for evidence_id in accepted:
            digest = hashlib.sha256(evidence_id.encode()).hexdigest()[:16]
            event_id = f"verification:{job.id}:{digest}"
            cur.execute(
                """
                INSERT INTO verification_events(
                  id,record_type,record_id,previous_state,new_state,
                  verifier_type,verifier_identity,rationale,evidence
                )
                SELECT %s,'evidence_object',e.id,e.verification_state,'MACHINE_CHECKED',
                       'MODEL',%s,%s,%s
                FROM evidence_objects e WHERE e.id=%s
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    event_id,
                    verifier,
                    "Accepted by semantic verification job",
                    Jsonb({"job_id": job.id, "issues": result.get("issues", [])}),
                    evidence_id,
                ),
            )
            cur.execute(
                """
                UPDATE evidence_objects SET verification_state='MACHINE_CHECKED'
                WHERE id=%s AND verification_state='UNVERIFIED'
                """,
                (evidence_id,),
            )

        return {
            "verified": bool(result.get("verified")),
            "accepted_evidence_ids": accepted,
            "verification_event_count": len(accepted),
            "artifact_id": materialized.get("artifact_id"),
        }

    def _run_checkpoint(self, cur, run_id: str) -> str:
        cur.execute(
            "SELECT knowledge_checkpoint_id FROM research_runs WHERE id=%s",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None or not row[0]:
            raise JobExecutionError(
                "research run has no knowledge checkpoint",
                code="RUN_CHECKPOINT_UNAVAILABLE",
                retryable=True,
            )
        return str(row[0])

    def _materialize_synthesis(self, cur, job, result):
        from psycopg.types.json import Jsonb

        checkpoint_id = str(job.payload["knowledge_checkpoint_id"])
        synthesis_id = f"synthesis:{job.id}"
        cur.execute("SELECT topic FROM research_runs WHERE id=%s", (job.run_id,))
        row = cur.fetchone()
        topic = row[0] if row else job.run_id
        body = _without_execution(result)
        cur.execute(
            """
            INSERT INTO synthesis_artifacts(
              id,scope_class,question,knowledge_checkpoint,body,status
            )
            VALUES (%s,'CORPUS',%s,%s,%s,'CURRENT')
            ON CONFLICT(id) DO NOTHING
            """,
            (
                synthesis_id,
                f"Synthesis for {topic}",
                checkpoint_id,
                Jsonb(body),
            ),
        )
        return {"synthesis_artifact_id": synthesis_id}

    def _materialize_gaps(self, cur, job, result):
        from psycopg.types.json import Jsonb

        checkpoint_id = self._run_checkpoint(cur, job.run_id)
        canonical = copy.deepcopy(_without_execution(result))
        cur.execute("SELECT policy FROM research_runs WHERE id=%s", (job.run_id,))
        row = cur.fetchone()
        policy = (row[0] if row else {}) or {}
        canonical["gaps"] = bounded_gap_candidates(canonical, policy)
        gap_ids: list[str] = []
        for index, gap in enumerate(canonical.get("gaps", []), start=1):
            proposed_id = str(gap.get("id") or index)
            gap_id = f"gap:{job.id}:{index:04d}"
            gap["proposed_id"] = proposed_id
            gap["id"] = gap_id
            motivating = list(gap.get("motivating_evidence_ids") or [])
            gap_ids.append(gap_id)
            cur.execute(
                """
                INSERT INTO gap_candidates(
                  id,statement,gap_type,knowledge_checkpoint_id,
                  motivating_evidence_ids,novelty_status,rationale,
                  human_review_state,attributes
                )
                VALUES (%s,%s,%s,%s,%s,'PROPOSED',%s,'NOT_REVIEWED',%s)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    gap_id,
                    str(gap["statement"]),
                    str(gap["gap_type"]),
                    checkpoint_id,
                    motivating,
                    str(gap["rationale"]),
                    Jsonb(
                        {
                            "proposed_id": proposed_id,
                            "research_run_id": job.run_id,
                            "search_queries": gap.get("search_queries", []),
                        }
                    ),
                ),
            )
        return {"gap_candidate_ids": gap_ids, "canonical_result": canonical}

    def _materialize_arbitration(self, cur, job, result):
        from psycopg.types.json import Jsonb

        checkpoint_id = self._run_checkpoint(cur, job.run_id)
        gap = job.payload["gap"]
        gap_id = str(gap["id"])
        status = str(result["status"])
        if status not in {
            "REJECTED_PRIOR_WORK_FOUND",
            "NARROWED",
            "NEEDS_HUMAN_REVIEW",
        }:
            raise JobExecutionError(
                f"invalid automated novelty status: {status}",
                code="INVALID_NOVELTY_STATUS",
                retryable=False,
            )

        provenance = []
        counterevidence = []
        for search in job.payload.get("search_results", []):
            search_result = search.get("result") or {}
            provenance.append(
                {
                    "query": search.get("query"),
                    "execution": search_result.get("_execution"),
                    "notes": search_result.get("notes"),
                }
            )
            counterevidence.extend(search_result.get("counterevidence", []))

        challenge_id = f"challenge:{job.id}"
        cur.execute(
            """
            INSERT INTO novelty_challenges(
              id,gap_candidate_id,knowledge_checkpoint_id,result,
              search_provenance,counterevidence,reasoning,human_approval
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,NULL)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                challenge_id,
                gap_id,
                checkpoint_id,
                status,
                Jsonb({"searches": provenance}),
                Jsonb(counterevidence),
                str(result["reasoning"]),
            ),
        )
        review_state = "REQUIRED" if status == "NEEDS_HUMAN_REVIEW" else "NOT_REQUIRED"
        cur.execute(
            """
            UPDATE gap_candidates
            SET novelty_status=%s, human_review_state=%s
            WHERE id=%s
            """,
            (status, review_state, gap_id),
        )
        canonical_result = copy.deepcopy(result)
        canonical_result["search_count"] = len(job.payload.get("search_results", []))
        return {
            "novelty_challenge_id": challenge_id,
            "gap_candidate_id": gap_id,
            "status": status,
            "canonical_result": canonical_result,
        }


class AuthoritativeSemanticExecutor:
    """Hydrate -> semantic reasoning -> crash-safe authoritative materialization."""

    def __init__(self, *, delegate, hydrator: ResearchContextHydrator, materializer: SemanticResultMaterializer):
        self.delegate = delegate
        self.hydrator = hydrator
        self.materializer = materializer

    def execute(self, job: ResearchJob) -> dict[str, Any]:
        cached = self.materializer.cached(job.id)
        if cached is not None:
            status, semantic_result, materialized = cached
            if status == "COMPLETED":
                return self.materializer._compose(semantic_result, materialized)
            return self.materializer.materialize(job, semantic_result)

        hydrated = self.hydrator.hydrate(job)
        semantic_result = self.delegate.execute(hydrated)
        return self.materializer.materialize(job, semantic_result)
