from __future__ import annotations

from typing import Any

from .domain import EvidenceObject, EvidenceRelation


class CorpusFixtureError(ValueError):
    pass


class ImmutableRecordConflict(CorpusFixtureError):
    pass


def validate_corpus_fixture(data: dict[str, Any]) -> None:
    required = {
        "works",
        "work_versions",
        "source_artifacts",
        "extraction_runs",
        "source_locators",
        "evidence_objects",
        "evidence_relations",
    }
    missing = required - data.keys()
    if missing:
        raise CorpusFixtureError(f"missing fixture keys: {sorted(missing)}")

    work_ids = {x["id"] for x in data["works"]}
    version_ids = {x["id"] for x in data["work_versions"]}
    artifact_ids = {x["id"] for x in data["source_artifacts"]}
    extraction_ids = {x["id"] for x in data["extraction_runs"]}
    locator_ids = {x["id"] for x in data["source_locators"]}

    for row in data["work_versions"]:
        if row["work_id"] not in work_ids:
            raise CorpusFixtureError(f"work_version {row['id']} references missing work")
    for row in data["source_artifacts"]:
        if row.get("work_version_id") and row["work_version_id"] not in version_ids:
            raise CorpusFixtureError(f"source_artifact {row['id']} references missing version")
    for row in data["extraction_runs"]:
        if row.get("work_version_id") and row["work_version_id"] not in version_ids:
            raise CorpusFixtureError(f"extraction_run {row['id']} references missing version")
        if row.get("source_artifact_id") and row["source_artifact_id"] not in artifact_ids:
            raise CorpusFixtureError(f"extraction_run {row['id']} references missing artifact")
    for row in data["source_locators"]:
        if row["source_artifact_id"] not in artifact_ids:
            raise CorpusFixtureError(f"locator {row['id']} references missing artifact")

    evidence: dict[str, EvidenceObject] = {}
    for raw in data["evidence_objects"]:
        if raw["work_id"] not in work_ids:
            raise CorpusFixtureError(f"evidence {raw['id']} references missing work")
        if raw.get("work_version_id") and raw["work_version_id"] not in version_ids:
            raise CorpusFixtureError(f"evidence {raw['id']} references missing version")
        if raw.get("source_artifact_id") and raw["source_artifact_id"] not in artifact_ids:
            raise CorpusFixtureError(f"evidence {raw['id']} references missing artifact")
        if raw.get("source_locator_id") and raw["source_locator_id"] not in locator_ids:
            raise CorpusFixtureError(f"evidence {raw['id']} references missing locator")
        if raw.get("extraction_run_id") and raw["extraction_run_id"] not in extraction_ids:
            raise CorpusFixtureError(f"evidence {raw['id']} references missing extraction run")
        if raw.get("supersedes_id") == raw["id"]:
            raise CorpusFixtureError(f"evidence {raw['id']} cannot supersede itself")
        evidence[raw["id"]] = EvidenceObject.model_validate(raw)

    for raw in data["evidence_relations"]:
        rel = EvidenceRelation.model_validate(raw)
        if rel.source_id not in evidence or rel.target_id not in evidence:
            raise CorpusFixtureError(f"relation {rel.id} references missing evidence")
        if rel.supersedes_id == rel.id:
            raise CorpusFixtureError(f"relation {rel.id} cannot supersede itself")
        rel.validate_direction(evidence)


class PostgresCorpusLoader:
    """Idempotent authoritative import of normalized research-engine corpus fixtures."""

    def __init__(self, store):
        self.store = store

    def load(self, data: dict[str, Any]) -> None:
        validate_corpus_fixture(data)
        from psycopg.types.json import Jsonb

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                for row in data["works"]:
                    cur.execute(
                        """
                        INSERT INTO works(id, title, publication_year, attributes)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT(id) DO UPDATE SET
                          title = EXCLUDED.title,
                          publication_year = EXCLUDED.publication_year,
                          attributes = EXCLUDED.attributes
                        """,
                        (row["id"], row["title"], row.get("publication_year"), Jsonb(row.get("attributes", {}))),
                    )

                for row in data["work_versions"]:
                    cur.execute(
                        """
                        INSERT INTO work_versions(id, work_id, version_label, published_at, attributes)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT(id) DO UPDATE SET
                          work_id = EXCLUDED.work_id,
                          version_label = EXCLUDED.version_label,
                          published_at = EXCLUDED.published_at,
                          attributes = EXCLUDED.attributes
                        """,
                        (
                            row["id"],
                            row["work_id"],
                            row.get("version_label"),
                            row.get("published_at"),
                            Jsonb(row.get("attributes", {})),
                        ),
                    )

                for row in data["source_artifacts"]:
                    cur.execute(
                        """
                        INSERT INTO source_artifacts(
                          id, work_version_id, sha256, byte_size, media_type, format,
                          access_class, artifact_uri, metadata
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(id) DO UPDATE SET
                          work_version_id = EXCLUDED.work_version_id,
                          sha256 = EXCLUDED.sha256,
                          byte_size = EXCLUDED.byte_size,
                          media_type = EXCLUDED.media_type,
                          format = EXCLUDED.format,
                          access_class = EXCLUDED.access_class,
                          artifact_uri = EXCLUDED.artifact_uri,
                          metadata = EXCLUDED.metadata
                        """,
                        (
                            row["id"],
                            row.get("work_version_id"),
                            row["sha256"],
                            row["byte_size"],
                            row["media_type"],
                            row["format"],
                            row["access_class"],
                            row.get("artifact_uri"),
                            Jsonb(row.get("metadata", {})),
                        ),
                    )

                for row in data["extraction_runs"]:
                    cur.execute(
                        """
                        INSERT INTO extraction_runs(
                          id, work_version_id, source_artifact_id, source_level,
                          extractor_identity, status, warnings, started_at, completed_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(id) DO UPDATE SET
                          work_version_id = EXCLUDED.work_version_id,
                          source_artifact_id = EXCLUDED.source_artifact_id,
                          source_level = EXCLUDED.source_level,
                          extractor_identity = EXCLUDED.extractor_identity,
                          status = EXCLUDED.status,
                          warnings = EXCLUDED.warnings,
                          started_at = EXCLUDED.started_at,
                          completed_at = EXCLUDED.completed_at
                        """,
                        (
                            row["id"],
                            row.get("work_version_id"),
                            row.get("source_artifact_id"),
                            row["source_level"],
                            row["extractor_identity"],
                            row.get("status", "COMPLETED"),
                            Jsonb(row.get("warnings", [])),
                            row.get("started_at"),
                            row.get("completed_at"),
                        ),
                    )

                for row in data["source_locators"]:
                    cur.execute(
                        """
                        INSERT INTO source_locators(
                          id, source_artifact_id, source_level, locator_quality,
                          page_number, pdf_bbox, section_path, raw_anchor, attributes
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(id) DO UPDATE SET
                          source_artifact_id = EXCLUDED.source_artifact_id,
                          source_level = EXCLUDED.source_level,
                          locator_quality = EXCLUDED.locator_quality,
                          page_number = EXCLUDED.page_number,
                          pdf_bbox = EXCLUDED.pdf_bbox,
                          section_path = EXCLUDED.section_path,
                          raw_anchor = EXCLUDED.raw_anchor,
                          attributes = EXCLUDED.attributes
                        """,
                        (
                            row["id"],
                            row["source_artifact_id"],
                            row["source_level"],
                            row["locator_quality"],
                            row.get("page_number"),
                            row.get("pdf_bbox"),
                            row.get("section_path"),
                            row.get("raw_anchor"),
                            Jsonb(row.get("attributes", {})),
                        ),
                    )

                for row in data["evidence_objects"]:
                    incoming = (
                        row["evidence_type"],
                        row["work_id"],
                        row.get("work_version_id"),
                        row.get("source_artifact_id"),
                        row.get("source_locator_id"),
                        row["source_level"],
                        row.get("raw_source_form"),
                        row["normalized_form"],
                        row.get("extraction_run_id"),
                        row.get("extractor_identity"),
                        row["verification_state"],
                        row.get("supersedes_id"),
                        row.get("attributes", {}),
                    )
                    cur.execute(
                        """
                        SELECT evidence_type, work_id, work_version_id, source_artifact_id,
                               source_locator_id, source_level, raw_source_form, normalized_form,
                               extraction_run_id, extractor_identity, verification_state,
                               supersedes_id, attributes
                        FROM evidence_objects WHERE id = %s
                        """,
                        (row["id"],),
                    )
                    existing = cur.fetchone()
                    if existing is None:
                        cur.execute(
                            """
                            INSERT INTO evidence_objects(
                              id, evidence_type, work_id, work_version_id, source_artifact_id,
                              source_locator_id, source_level, raw_source_form, normalized_form,
                              extraction_run_id, extractor_identity, verification_state,
                              supersedes_id, attributes
                            )
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (row["id"], *incoming[:-1], Jsonb(incoming[-1])),
                        )
                    elif tuple(existing) != incoming:
                        raise ImmutableRecordConflict(
                            f"evidence id {row['id']} already exists with different authoritative content; "
                            "create a new evidence id and use supersedes_id"
                        )

                for row in data["evidence_relations"]:
                    incoming = (
                        row["source_id"],
                        row["target_id"],
                        row["relation_type"],
                        row.get("context_compatibility"),
                        row["verification_state"],
                        row["rationale"],
                        row.get("supersedes_id"),
                        row.get("attributes", {}),
                    )
                    cur.execute(
                        """
                        SELECT source_id, target_id, relation_type, context_compatibility,
                               verification_state, rationale, supersedes_id, attributes
                        FROM evidence_relations WHERE id = %s
                        """,
                        (row["id"],),
                    )
                    existing = cur.fetchone()
                    if existing is None:
                        cur.execute(
                            """
                            INSERT INTO evidence_relations(
                              id, source_id, target_id, relation_type, context_compatibility,
                              verification_state, rationale, supersedes_id, attributes
                            )
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (row["id"], *incoming[:-1], Jsonb(incoming[-1])),
                        )
                    elif tuple(existing) != incoming:
                        raise ImmutableRecordConflict(
                            f"relation id {row['id']} already exists with different authoritative content; "
                            "create a new relation id and use supersedes_id"
                        )
            conn.commit()
