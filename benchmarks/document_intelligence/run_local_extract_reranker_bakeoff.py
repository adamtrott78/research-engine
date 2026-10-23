from __future__ import annotations

import argparse
import json
import pathlib

from research_engine.local_extract_reranker_bakeoff import (
    run_reranker_bakeoff,
)


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--support-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    report = run_reranker_bakeoff(
        fixture=_json(args.fixture),
        contract=_json(args.contract),
        support_fixture=_json(args.support_fixture),
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "decision": report["decision"],
        "promising_arms": report["promising_arms"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
