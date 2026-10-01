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

    auto = sub.add_parser("autonomous-run")
    auto.add_argument("topic")
    auto.add_argument("--run-id")
    auto.add_argument("--poll-seconds", type=float, default=1.0)
    auto.add_argument("--lease-seconds", type=int, default=300)
    auto.add_argument("--max-cycles", type=int)

    daemon = sub.add_parser("daemon")
    daemon.add_argument("--poll-seconds", type=float, default=2.0)
    daemon.add_argument("--lease-seconds", type=int, default=300)
    daemon.add_argument("--max-ticks", type=int)
    daemon.add_argument("--max-active-runs", type=int, default=4)
    daemon.add_argument("--max-pending-human", type=int, default=8)
    daemon.add_argument("--stale-worker-seconds", type=int, default=900)
    daemon.add_argument("--submit-topic")
    daemon.add_argument("--run-id")

    report = sub.add_parser("overnight-report")
    report.add_argument("--hours", type=float, default=12.0)
    report.add_argument("--max-active-runs", type=int, default=4)
    report.add_argument("--max-pending-human", type=int, default=8)

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
    elif args.cmd == "autonomous-run":
        from .controller import controller_from_env

        controller = controller_from_env(
            lease_seconds=args.lease_seconds,
            poll_seconds=args.poll_seconds,
        )
        result = controller.run(
            topic=args.topic,
            run_id=args.run_id,
            max_cycles=args.max_cycles,
        )
        print(json.dumps({
            "run_id": result.run_id,
            "status": result.status,
            "cycles": result.cycles,
            "jobs_processed": result.jobs_processed,
            "recovered_leases": result.recovered_leases,
            "state": result.state,
            "human_review": result.human_review,
        }, default=str))
    elif args.cmd == "daemon":
        from .daemon import DaemonLimits, daemon_from_env

        limits = DaemonLimits(
            max_active_runs=args.max_active_runs,
            max_pending_human=args.max_pending_human,
            stale_worker_seconds=args.stale_worker_seconds,
        )
        service = daemon_from_env(
            poll_seconds=args.poll_seconds,
            lease_seconds=args.lease_seconds,
            limits=limits,
        )
        submitted = None
        if args.submit_topic:
            submitted = service.submit_run(
                topic=args.submit_topic,
                run_id=args.run_id,
            )
        result = service.run_forever(max_ticks=args.max_ticks)
        result["submitted_run_id"] = submitted
        print(json.dumps(result, default=str))
    elif args.cmd == "overnight-report":
        from .daemon import DaemonLimits, daemon_from_env

        service = daemon_from_env(
            with_workers=False,
            limits=DaemonLimits(
                max_active_runs=args.max_active_runs,
                max_pending_human=args.max_pending_human,
            ),
        )
        print(json.dumps(service.overnight_report(hours=args.hours), default=str))
