from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_store import ContentAddressedArtifactStore
from .validation import FidelitySummary
from .orchestration import ExecutorTier


def main() -> None:
    p = argparse.ArgumentParser(prog="research-engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("artifact-put")
    a.add_argument("path")
    a.add_argument("--root", required=True)

    f = sub.add_parser("check-fidelity")
    f.add_argument("fixture")

    w = sub.add_parser("semantic-worker")
    w.add_argument(
        "--tiers",
        default="CHEAP,STANDARD,STRONG",
        help="comma-separated model tiers",
    )
    w.add_argument("--worker-id")
    w.add_argument("--poll-seconds", type=float, default=2.0)
    w.add_argument("--lease-seconds", type=int, default=300)
    w.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        help="exit after this many empty polls; omit for daemon mode",
    )

    dw = sub.add_parser("deterministic-worker")
    dw.add_argument("--worker-id")
    dw.add_argument("--poll-seconds", type=float, default=2.0)
    dw.add_argument("--lease-seconds", type=int, default=300)
    dw.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        help="exit after this many empty polls; omit for daemon mode",
    )

    arxiv = sub.add_parser("arxiv-acquire")
    arxiv.add_argument("arxiv_id")
    arxiv.add_argument("--root", required=True)

    args = p.parse_args()
    if args.cmd == "artifact-put":
        receipt = ContentAddressedArtifactStore(args.root).put_file(args.path)
        print(json.dumps({"artifact_id": receipt.artifact_id, "sha256": receipt.sha256, "byte_size": receipt.byte_size, "path": str(receipt.path)}))
    elif args.cmd == "check-fidelity":
        data = json.loads(Path(args.fixture).read_text())
        s = FidelitySummary(**data["summary"])
        s.assert_promotion_safe()
        print(json.dumps({"result":"PASS","semantic_preservation_rate":s.semantic_preservation_rate}))
    elif args.cmd == "semantic-worker":
        from .worker_runtime import run_semantic_worker_loop

        tiers = tuple(
            ExecutorTier(part.strip().upper())
            for part in args.tiers.split(",")
            if part.strip()
        )
        completed = run_semantic_worker_loop(
            tiers=tiers,
            worker_id=args.worker_id,
            poll_seconds=args.poll_seconds,
            lease_seconds=args.lease_seconds,
            max_idle_polls=args.max_idle_polls,
        )
        print(json.dumps({"result": "STOPPED", "jobs_processed": completed}))
    elif args.cmd == "deterministic-worker":
        from .worker_runtime import run_deterministic_worker_loop

        completed = run_deterministic_worker_loop(
            worker_id=args.worker_id,
            poll_seconds=args.poll_seconds,
            lease_seconds=args.lease_seconds,
            max_idle_polls=args.max_idle_polls,
        )
        print(json.dumps({"result": "STOPPED", "jobs_processed": completed}))
    elif args.cmd == "arxiv-acquire":
        from .arxiv import ArxivProvider
        result = ArxivProvider(ContentAddressedArtifactStore(args.root)).acquire_pdf(args.arxiv_id)
        print(json.dumps({
            "arxiv_id": result.arxiv_id,
            "url": result.versioned_pdf_url,
            "resolved_url": result.resolved_url,
            "media_type": result.media_type,
            "artifact_id": result.receipt.artifact_id,
            "sha256": result.receipt.sha256,
            "byte_size": result.receipt.byte_size,
            "path": str(result.receipt.path),
        }))
