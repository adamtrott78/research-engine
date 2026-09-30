from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_store import ContentAddressedArtifactStore
from .validation import FidelitySummary


def main() -> None:
    p = argparse.ArgumentParser(prog="research-engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("artifact-put")
    a.add_argument("path")
    a.add_argument("--root", required=True)

    f = sub.add_parser("check-fidelity")
    f.add_argument("fixture")

    args = p.parse_args()
    if args.cmd == "artifact-put":
        receipt = ContentAddressedArtifactStore(args.root).put_file(args.path)
        print(json.dumps({"artifact_id": receipt.artifact_id, "sha256": receipt.sha256, "byte_size": receipt.byte_size, "path": str(receipt.path)}))
    elif args.cmd == "check-fidelity":
        data = json.loads(Path(args.fixture).read_text())
        s = FidelitySummary(**data["summary"])
        s.assert_promotion_safe()
        print(json.dumps({"result":"PASS","semantic_preservation_rate":s.semantic_preservation_rate}))
