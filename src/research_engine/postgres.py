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
                    (work_id, title, publication_year, attributes or {}),
                )
            conn.commit()

    def claim_support(self, claim_id: str) -> list[dict]:
        from .sql_queries import CLAIM_SUPPORT_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CLAIM_SUPPORT_SQL, {"claim_id": claim_id})
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def provenance_path(self, start_id: str, goal_id: str, max_depth: int = 8) -> list[dict]:
        from .sql_queries import PROVENANCE_PATH_SQL
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(PROVENANCE_PATH_SQL, {"start_id": start_id, "goal_id": goal_id, "max_depth": max_depth})
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
