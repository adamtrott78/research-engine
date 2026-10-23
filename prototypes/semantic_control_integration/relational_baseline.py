from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nearest_rank(samples: list[int], p: float) -> int:
    if not samples:
        raise ValueError("no timing samples")
    ordered = sorted(samples)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[rank - 1]


def _validate_result(query: dict[str, Any], result: Any) -> None:
    expected = query["expected"]
    query_id = query["query_id"]

    if query_id == "Q1_CLAIM_SUPPORT":
        assert len(result) == expected["row_count"]
        assert [row["result_id"] for row in result] == expected["ordered_result_ids"]
        assert all(row["claim_id"] == expected["claim_id"] for row in result)
        assert all(row["source_sha256"] == expected["source_sha256"] for row in result)
        assert [row["page_number"] for row in result] == expected["ordered_page_numbers"]
        assert all(row["verification_state"] == expected["verification_state"] for row in result)
        return

    if query_id == "Q2_METHOD_QUALIFIERS":
        assert len(result) == expected["row_count"]
        row = result[0]
        for field in [
            "qualifier_id",
            "evidence_type",
            "page_number",
            "source_sha256",
            "verification_state",
        ]:
            assert row[field] == expected[field]
        return

    if query_id == "Q3_SOURCE_LEVEL_COUNTS":
        assert result == expected
        return

    if query_id == "Q4_PROVENANCE_PATH":
        assert len(result) == expected["row_count"]
        row = result[0]
        for field in [
            "source_id",
            "target_id",
            "relation_type",
            "path",
            "depth",
        ]:
            assert row[field] == expected[field]
        return

    raise ValueError(f"unknown query_id: {query_id}")


def _call(store: Any, query: dict[str, Any]) -> Any:
    method = getattr(store, query["store_method"])
    return method(**query["arguments"])


def run_relational_baseline(repo_root: Path) -> dict[str, Any]:
    dsn = os.getenv("RESEARCH_ENGINE_DATABASE_URL")
    if not dsn:
        raise RuntimeError("RESEARCH_ENGINE_DATABASE_URL is required")

    import psycopg
    from research_engine.corpus_loader import PostgresCorpusLoader
    from research_engine.postgres import PostgresStore

    base = repo_root / "benchmarks" / "semantic_control_integration"
    contract = json.loads((base / "relational-baseline-contract.v1.json").read_text())
    fixture_path = repo_root / contract["authoritative_sources"]["fixture"]["path"]
    schema_path = repo_root / contract["authoritative_sources"]["schema"]["path"]
    fixture = json.loads(fixture_path.read_text())

    store = PostgresStore(dsn)
    store.initialize(schema_path)
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE knowledge_checkpoints, verification_events,
                         evidence_relations, evidence_objects, source_locators,
                         extraction_runs, acquisition_attempts, discovery_runs,
                         external_identifiers, source_artifacts, work_versions, works
                CASCADE
                """
            )
        conn.commit()

    PostgresCorpusLoader(store).load(fixture)

    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            postgres_server_version = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM works")
            works = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM evidence_objects")
            evidence_objects = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM evidence_relations")
            evidence_relations = cur.fetchone()[0]

    identity = contract["fixture_identity"]
    assert works == identity["works"]
    assert evidence_objects == identity["evidence_objects"]
    assert evidence_relations == identity["evidence_relations"]

    warmups = contract["timing_methodology"]["warmup_repetitions_per_query"]
    repetitions = contract["timing_methodology"]["measured_repetitions_per_query"]
    query_results: list[dict[str, Any]] = []

    for query in contract["queries"]:
        for _ in range(warmups):
            value = _call(store, query)
            _validate_result(query, value)

        samples: list[int] = []
        result_hashes: list[str] = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            value = _call(store, query)
            elapsed = time.perf_counter_ns() - start
            _validate_result(query, value)
            samples.append(elapsed)
            result_hashes.append(_sha256(value))

        unique_hashes = sorted(set(result_hashes))
        assert len(unique_hashes) == 1
        query_results.append(
            {
                "query_id": query["query_id"],
                "store_method": query["store_method"],
                "warmup_repetitions": warmups,
                "measured_repetitions": repetitions,
                "canonical_result_sha256": unique_hashes[0],
                "answer_correct": True,
                "latency_ns": {
                    "min": min(samples),
                    "mean": int(round(statistics.fmean(samples))),
                    "p50": _nearest_rank(samples, 0.50),
                    "p95": _nearest_rank(samples, 0.95),
                    "max": max(samples),
                },
            }
        )

    body = {
        "schema_version": 1,
        "baseline_id": contract["baseline_id"],
        "repository_checkpoint": contract["repository_checkpoint"],
        "execution_environment": {
            "production_database_access": False,
            "ci_database_url_source": "RESEARCH_ENGINE_DATABASE_URL",
            "postgres_server_version": postgres_server_version,
            "psycopg_version": psycopg.__version__,
            "python_version": platform.python_version(),
        },
        "fixture_identity": {
            **identity,
            "git_blob_sha": contract["authoritative_sources"]["fixture"]["git_blob_sha"],
        },
        "timing_methodology": contract["timing_methodology"],
        "query_results": query_results,
        "all_answers_correct": all(item["answer_correct"] for item in query_results),
        "performance_promotion_claim": False,
        "latency_threshold_applied": False,
        "production_mutation": False,
    }
    return {
        **body,
        "baseline_result_sha256": _sha256(body),
    }
