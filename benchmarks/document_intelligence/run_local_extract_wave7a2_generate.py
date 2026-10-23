from __future__ import annotations

import argparse
import json
import os
import pathlib

from research_engine.local_extract_wave7a2_intents import generate_wave7a2_intents


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=pathlib.Path,required=True)
    parser.add_argument("--contract",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    parser.add_argument("--workdir",type=pathlib.Path,required=True)
    args=parser.parse_args()
    expected=[x for x in os.environ.get("P5_GPU_TARGET_UUIDS","").split(",") if x]
    report=generate_wave7a2_intents(
        fixture=_load(args.fixture),
        contract=_load(args.contract),
        workdir=args.workdir,
        model_path=os.environ["P5_MODEL_PATH"],
        expected_gpu_uuids=expected,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "paper_count":report["metrics"]["paper_count"],
        "model_calls":report["metrics"]["model_calls"],
        "output":str(args.output),
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
