from __future__ import annotations

import argparse
import json
import pathlib

from research_engine.local_extract_wave7a3_evaluate import evaluate_wave7a3


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=pathlib.Path,required=True)
    parser.add_argument("--contract",type=pathlib.Path,required=True)
    parser.add_argument("--support-fixture",type=pathlib.Path,required=True)
    parser.add_argument("--generated",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    parser.add_argument("--workdir",type=pathlib.Path,required=True)
    args=parser.parse_args()
    report=evaluate_wave7a3(
        fixture=_load(args.fixture),
        contract=_load(args.contract),
        support_fixture=_load(args.support_fixture),
        generated=_load(args.generated),
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "decision":report["wave7a3_decision"],
        "qualifying_pairs":report["qualifying_pairs"],
        "output":str(args.output),
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
