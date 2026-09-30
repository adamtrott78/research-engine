from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class KnowledgeCheckpoint:
    id: str
    fingerprint: str
    scope: dict[str, Any]
    manifest: dict[str, list[dict[str, str]]]
    record_counts: dict[str, int]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_record(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def checkpoint_fingerprint(scope: dict[str, Any], manifest: dict[str, list[dict[str, str]]]) -> str:
    payload = {"scope": scope, "manifest": manifest}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class KnowledgeCheckpointBuilder:
    """Build a deterministic scientific-state manifest from authoritative DB records."""

    def __init__(self, store):
        self.store = store

    @staticmethod
    def _fetch_dicts(cur, sql: str, params=()) -> list[dict[str, Any]]:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def capture(self, work_ids: list[str] | None = None) -> KnowledgeCheckpoint:
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                if work_ids is None:
                    works = self._fetch_dicts(
                        cur,
                        "SELECT id, title, publication_year, attributes FROM works ORDER BY id",
                    )
                else:
                    selected = sorted(set(work_ids))
                    works = self._fetch_dicts(
                        cur,
                        """
                        SELECT id, title, publication_year, attributes
                        FROM works WHERE id = ANY(%s) ORDER BY id
                        """,
                        (selected,),
                    )

                selected_work_ids = [r["id"] for r in works]
                if not selected_work_ids:
                    raise ValueError("knowledge checkpoint requires at least one work")

                versions = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, work_id, version_label, published_at, attributes
                    FROM work_versions
                    WHERE work_id = ANY(%s)
                    ORDER BY id
                    """,
                    (selected_work_ids,),
                )
                version_ids = [r["id"] for r in versions]

                artifacts = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, work_version_id, sha256, byte_size, media_type, format,
                           access_class, artifact_uri, metadata
                    FROM source_artifacts
                    WHERE work_version_id = ANY(%s)
                    ORDER BY id
                    """,
                    (version_ids,),
                ) if version_ids else []

                evidence = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, evidence_type, work_id, work_version_id, source_artifact_id,
                           source_locator_id, source_level, raw_source_form, normalized_form,
                           extraction_run_id, extractor_identity, verification_state,
                           supersedes_id, attributes
                    FROM evidence_objects
                    WHERE work_id = ANY(%s)
                    ORDER BY id
                    """,
                    (selected_work_ids,),
                )
                evidence_ids = [r["id"] for r in evidence]
                locator_ids = sorted({r["source_locator_id"] for r in evidence if r["source_locator_id"]})
                extraction_ids = sorted({r["extraction_run_id"] for r in evidence if r["extraction_run_id"]})

                locators = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, source_artifact_id, source_level, locator_quality,
                           page_number, pdf_bbox, section_path, raw_anchor, attributes
                    FROM source_locators
                    WHERE id = ANY(%s)
                    ORDER BY id
                    """,
                    (locator_ids,),
                ) if locator_ids else []

                extraction_runs = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, work_version_id, source_artifact_id, source_level,
                           extractor_identity, status, warnings, started_at, completed_at
                    FROM extraction_runs
                    WHERE id = ANY(%s)
                    ORDER BY id
                    """,
                    (extraction_ids,),
                ) if extraction_ids else []

                relations = self._fetch_dicts(
                    cur,
                    """
                    SELECT id, source_id, target_id, relation_type, context_compatibility,
                           verification_state, rationale, supersedes_id, attributes
                    FROM evidence_relations
                    WHERE source_id = ANY(%s) AND target_id = ANY(%s)
                    ORDER BY id
                    """,
                    (evidence_ids, evidence_ids),
                ) if evidence_ids else []

                rows_by_type = {
                    "works": works,
                    "work_versions": versions,
                    "source_artifacts": artifacts,
                    "extraction_runs": extraction_runs,
                    "source_locators": locators,
                    "evidence_objects": evidence,
                    "evidence_relations": relations,
                }
                manifest = {
                    name: [{"id": row["id"], "sha256": _hash_record(row)} for row in rows]
                    for name, rows in rows_by_type.items()
                }
                scope = {"work_ids": selected_work_ids}
                counts = {name: len(rows) for name, rows in rows_by_type.items()}
                fingerprint = checkpoint_fingerprint(scope, manifest)
                checkpoint_id = f"knowledge-checkpoint:sha256:{fingerprint}"

                cur.execute(
                    """
                    INSERT INTO knowledge_checkpoints(id, fingerprint, scope, manifest, record_counts)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        checkpoint_id,
                        fingerprint,
                        Jsonb(scope),
                        Jsonb(manifest),
                        Jsonb(counts),
                    ),
                )
            conn.commit()

        return KnowledgeCheckpoint(
            id=checkpoint_id,
            fingerprint=fingerprint,
            scope=scope,
            manifest=manifest,
            record_counts=counts,
        )
