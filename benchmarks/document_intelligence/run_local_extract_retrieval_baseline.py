#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research_engine.local_extract_retrieval import (
    PreservedExtractRetrievalBenchmark,
)
from research_engine.postgres import PostgresStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure model-free local EXTRACT retrieval against preserved "
            "historical EXTRACT evidence without materializing scientific state."
        )
    )
    parser.add_argument("--run-id")
    parser.add_argument("--expected-papers", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--quote-threshold", type=float, default=0.8)
    parser.add_argument("--max-region-chars", type=int, default=650)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dsn = os.environ["RESEARCH_ENGINE_DATABASE_URL"]
    artifact_root = os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
    service = PreservedExtractRetrievalBenchmark(
        store=PostgresStore(dsn),
        artifact_root=artifact_root,
    )
    result = service.run(
        run_id=args.run_id,
        expected_papers=args.expected_papers,
        top_k=args.top_k,
        quote_threshold=args.quote_threshold,
        max_region_chars=args.max_region_chars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    print(f"OUTPUT={args.output}")
    print("LOCAL_EXTRACT_RETRIEVAL_BASELINE=MEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
