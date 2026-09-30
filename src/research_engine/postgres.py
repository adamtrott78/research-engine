from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class PostgresStore:
    """Thin authoritative-store boundary. psycopg is imported only at runtime."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    @contextmanager
    def connect(self) -> Iterator[object]:
        import psycopg
        with psycopg.connect(self.dsn) as conn:
            yield conn

    def initialize(self, schema_path: str | Path) -> None:
        sql = Path(schema_path).read_text()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def put_work(self, *, work_id: str, title: str, publication_year: int | None = None, attributes: dict | None = None) -> None:
        from psycopg.types.json import Jsonb
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO works(id, title, publication_year, attributes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(id) DO UPDATE SET
                      title = EXCLUDED.title,
                      publication_year = EXCLUDED.publication_year,
                      attributes = EXCLUDED.attributes
                    """,
                    (work_id, title, publication_year, Jsonb(attributes or {})),
                )
            conn.commit()

    def claim_support(self, claim_id: str) -> list[dict]:
        from .sql_queries import CLAIM_SUPPORT_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CLAIM_SUPPORT_SQL, {"claim_id": claim_id})
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def qualifiers_for_method(self, method_id: str) -> list[dict]:
        from .sql_queries import METHOD_QUALIFIERS_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(METHOD_QUALIFIERS_SQL, {"method_id": method_id})
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def source_level_counts(self) -> dict[str, int]:
        from .sql_queries import SOURCE_LEVEL_COUNTS_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SOURCE_LEVEL_COUNTS_SQL)
                return {row[0]: row[1] for row in cur.fetchall()}

    def provenance_path(self, start_id: str, goal_id: str, max_depth: int = 8) -> list[dict]:
        from .sql_queries import PROVENANCE_PATH_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(PROVENANCE_PATH_SQL, {"start_id": start_id, "goal_id": goal_id, "max_depth": max_depth})
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def record_verification_event(
        self,
        *,
        event_id: str,
        record_type: str,
        record_id: str,
        new_state: str,
        verifier_type: str,
        verifier_identity: str,
        rationale: str | None = None,
        evidence: list | None = None,
    ) -> None:
        from psycopg.types.json import Jsonb

        tables = {
            "evidence_object": "evidence_objects",
            "evidence_relation": "evidence_relations",
        }
        table = tables.get(record_type)
        if table is None:
            raise ValueError(f"unsupported verification record_type: {record_type}")

        requested = (
            record_type,
            record_id,
            new_state,
            verifier_type,
            verifier_identity,
            rationale,
            evidence or [],
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT record_type, record_id, new_state, verifier_type,
                           verifier_identity, rationale, evidence
                    FROM verification_events WHERE id = %s
                    """,
                    (event_id,),
                )
                existing_event = cur.fetchone()
                if existing_event is not None:
                    if tuple(existing_event) != requested:
                        raise ValueError(f"verification event id {event_id} already exists with different content")
                    return

                cur.execute(f"SELECT verification_state FROM {table} WHERE id = %s", (record_id,))
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"{record_type} {record_id} does not exist")
                previous_state = row[0]

                cur.execute(
                    """
                    INSERT INTO verification_events(
                      id, record_type, record_id, previous_state, new_state,
                      verifier_type, verifier_identity, rationale, evidence
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        event_id,
                        record_type,
                        record_id,
                        previous_state,
                        new_state,
                        verifier_type,
                        verifier_identity,
                        rationale,
                        Jsonb(evidence or []),
                    ),
                )
                cur.execute(f"UPDATE {table} SET verification_state = %s WHERE id = %s", (new_state, record_id))
            conn.commit()

    def create_knowledge_checkpoint(self, work_ids: list[str] | None = None):
        from .checkpoints import KnowledgeCheckpointBuilder
        return KnowledgeCheckpointBuilder(self).capture(work_ids=work_ids)
