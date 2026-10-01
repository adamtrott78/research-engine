from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
    daemon.add_argument(
        "--max-active-runs",
        type=int,
        default=int(os.getenv("RESEARCH_ENGINE_MAX_ACTIVE_RUNS", "4")),
    )
    daemon.add_argument(
        "--max-pending-human",
        type=int,
        default=int(os.getenv("RESEARCH_ENGINE_MAX_PENDING_HUMAN", "8")),
    )
    daemon.add_argument(
        "--stale-worker-seconds",
        type=int,
        default=int(os.getenv("RESEARCH_ENGINE_STALE_WORKER_SECONDS", "900")),
    )
    daemon.add_argument("--submit-topic")
    daemon.add_argument("--run-id")
    daemon.add_argument(
        "--policy-json",
        help="JSON object persisted as research_runs.policy for a submitted run",
    )

    submit = sub.add_parser("submit-run")
    submit.add_argument("topic")
    submit.add_argument("--run-id")
    submit.add_argument(
        "--policy-json",
        help="JSON object persisted as research_runs.policy",
    )
    submit.add_argument(
        "--discovery-json",
        help="JSON object used as the initial DISCOVERY job payload",
    )

    report = sub.add_parser("overnight-report")
    report.add_argument("--hours", type=float, default=12.0)
    report.add_argument(
        "--max-active-runs",
        type=int,
        default=int(os.getenv("RESEARCH_ENGINE_MAX_ACTIVE_RUNS", "4")),
    )
    report.add_argument(
        "--max-pending-human",
        type=int,
        default=int(os.getenv("RESEARCH_ENGINE_MAX_PENDING_HUMAN", "8")),
    )

    review = sub.add_parser("human-review-resolve")
    review.add_argument("request_id")
    decision_group = review.add_mutually_exclusive_group(required=True)
    decision_group.add_argument("--approve", action="store_true")
    decision_group.add_argument("--reject", action="store_true")
    review.add_argument(
        "--decision-json",
        required=True,
        help="JSON object persisted as the durable human review decision",
    )

    ops_snapshot = sub.add_parser("ops-snapshot")
    ops_snapshot.add_argument("--run-id")
    ops_snapshot.add_argument(
        "--windows",
        default="1,5",
        help="comma-separated rolling windows in hours",
    )

    ops_mirror = sub.add_parser("ops-mirror")
    ops_mirror.add_argument("--run-id")
    ops_mirror.add_argument(
        "--windows",
        default="1,5",
        help="comma-separated rolling windows in hours",
    )
    ops_mirror.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )

    hud = sub.add_parser("hud")
    hud.add_argument("--run-id")
    hud.add_argument("--refresh-seconds", type=float, default=2.0)
    hud.add_argument("--once", action="store_true")
    hud.add_argument("--no-color", action="store_true")

    allowance = sub.add_parser("allowance-observe")
    allowance.add_argument(
        "--window",
        required=True,
        choices=["5h", "weekly"],
    )
    percent_group = allowance.add_mutually_exclusive_group(required=True)
    percent_group.add_argument("--used-percent", type=float)
    percent_group.add_argument("--left-percent", type=float)
    reset_group = allowance.add_mutually_exclusive_group()
    reset_group.add_argument("--resets-at")
    reset_group.add_argument(
        "--resets-in",
        help="relative duration such as '3h 20m' or '5d 8h'",
    )
    allowance.add_argument("--note")
    allowance.add_argument("--approximate", action="store_true")

    allowance_paste = sub.add_parser("allowance-observe-paste")
    allowance_paste.add_argument(
        "--note",
        help="note applied to both parsed allowance observations",
    )

    calibrate_verify = sub.add_parser("calibrate-verify")
    calibrate_verify.add_argument("run_id")
    calibrate_verify.add_argument("--source-job-id")
    calibrate_verify.add_argument(
        "--selector",
        choices=["lowest", "median", "highest"],
        default="median",
    )

    extract_audit = sub.add_parser("audit-extract-provenance")
    extract_audit.add_argument("run_id")

    budget_status = sub.add_parser("budget-status")
    budget_status.add_argument("run_id")

    budget_resume = sub.add_parser("budget-resume")
    budget_resume.add_argument("run_id")
    budget_resume.add_argument(
        "--budgets-json",
        required=True,
        help="JSON object merged into policy.budgets before explicit resume",
    )

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
            policy = json.loads(args.policy_json) if args.policy_json else None
            submitted = service.submit_run(
                topic=args.submit_topic,
                run_id=args.run_id,
                policy=policy,
            )
        result = service.run_forever(max_ticks=args.max_ticks)
        result["submitted_run_id"] = submitted
        print(json.dumps(result, default=str))
    elif args.cmd == "submit-run":
        from .daemon import daemon_from_env

        service = daemon_from_env(with_workers=False)
        policy = json.loads(args.policy_json) if args.policy_json else None
        discovery = (
            json.loads(args.discovery_json)
            if args.discovery_json
            else None
        )
        if policy is not None and not isinstance(policy, dict):
            raise ValueError("--policy-json must be a JSON object")
        if discovery is not None and not isinstance(discovery, dict):
            raise ValueError("--discovery-json must be a JSON object")
        run_id = service.submit_run(
            topic=args.topic,
            run_id=args.run_id,
            policy=policy,
            discovery_payload=discovery,
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": service.queue.run_state(run_id)["status"],
                    "admission": service.admission_status(),
                },
                default=str,
            )
        )
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
    elif args.cmd == "human-review-resolve":
        from .orchestration import OrchestratorStore
        from .postgres import PostgresStore

        store = PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"])
        queue = OrchestratorStore(store)
        decision = json.loads(args.decision_json)
        if not isinstance(decision, dict):
            raise ValueError("--decision-json must be a JSON object")
        queue.resolve_human_review(
            args.request_id,
            approved=bool(args.approve),
            decision=decision,
        )
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "status": "APPROVED" if args.approve else "REJECTED",
                    "decision": decision,
                },
                default=str,
            )
        )
    elif args.cmd == "ops-snapshot":
        from .ops import service_from_env

        windows = tuple(
            float(part.strip())
            for part in args.windows.split(",")
            if part.strip()
        )
        service = service_from_env()
        print(
            json.dumps(
                service.snapshot(
                    run_id=args.run_id,
                    windows=windows,
                ),
                default=str,
            )
        )
    elif args.cmd == "ops-mirror":
        from .ops import service_from_env
        from .ops_mirror import build_chat_ops_mirror, render_chat_ops_markdown

        windows = tuple(
            float(part.strip())
            for part in args.windows.split(",")
            if part.strip()
        )
        mirror = build_chat_ops_mirror(
            service_from_env().snapshot(
                run_id=args.run_id,
                windows=windows,
            )
        )
        if args.format == "json":
            print(json.dumps(mirror, default=str, sort_keys=True))
        else:
            print(render_chat_ops_markdown(mirror))
    elif args.cmd == "hud":
        from .ops import hud_loop, service_from_env

        hud_loop(
            service_from_env(),
            run_id=args.run_id,
            refresh_seconds=args.refresh_seconds,
            once=args.once,
            color=not args.no_color,
        )
    elif args.cmd == "allowance-observe":
        from .ops import (
            parse_relative_duration,
            relative_duration_precision_seconds,
            service_from_env,
        )

        observed_at = datetime.now(timezone.utc)
        if args.left_percent is not None:
            percent_used = 100.0 - float(args.left_percent)
        else:
            percent_used = float(args.used_percent)
        if not 0 <= percent_used <= 100:
            raise ValueError("allowance percentage must resolve to [0,100] used")

        percent_quantum = 1.0 if args.left_percent is not None else None
        reset_at = None
        reset_precision_seconds = None
        if args.resets_at:
            reset_at = datetime.fromisoformat(
                args.resets_at.replace("Z", "+00:00")
            )
            if reset_at.tzinfo is None:
                raise ValueError("--resets-at must include a timezone")
        elif args.resets_in:
            reset_at = observed_at + parse_relative_duration(args.resets_in)
            reset_precision_seconds = relative_duration_precision_seconds(
                args.resets_in
            )

        observation = service_from_env().observe_allowance(
            window=args.window,
            percent_used=percent_used,
            reset_at=reset_at,
            note=args.note,
            approximate=(
                args.approximate
                or percent_quantum is not None
                or reset_precision_seconds is not None
            ),
            percent_quantum=percent_quantum,
            reset_precision_seconds=reset_precision_seconds,
            observed_at=observed_at,
        )
        print(json.dumps(observation, default=str))
    elif args.cmd == "allowance-observe-paste":
        import sys

        from .ops import (
            parse_allowance_ui_paste,
            parse_relative_duration,
            relative_duration_precision_seconds,
            service_from_env,
        )

        print(
            "Paste the complete 5-hour + weekly usage-limit block:",
            file=sys.stderr,
            flush=True,
        )
        lines: list[str] = []
        parsed = None
        for line in sys.stdin:
            lines.append(line)
            try:
                parsed = parse_allowance_ui_paste("".join(lines))
            except ValueError:
                continue
            else:
                break
        if parsed is None:
            parsed = parse_allowance_ui_paste("".join(lines))

        observed_at = datetime.now(timezone.utc)
        service = service_from_env()
        observations = []
        for window in ("5h", "weekly"):
            row = parsed[window]
            reset_text = str(row["resets_in"])
            observation = service.observe_allowance(
                window=window,
                percent_used=100.0 - float(row["left_percent"]),
                reset_at=observed_at + parse_relative_duration(reset_text),
                note=args.note,
                approximate=True,
                percent_quantum=1.0,
                reset_precision_seconds=relative_duration_precision_seconds(
                    reset_text
                ),
                observed_at=observed_at,
            )
            observations.append(observation)
        print(json.dumps(observations, default=str))
    elif args.cmd == "calibrate-verify":
        from .calibration import service_from_env as calibration_from_env

        result = calibration_from_env().run(
            run_id=args.run_id,
            source_job_id=args.source_job_id,
            selector=args.selector,
        )
        print(json.dumps(result, default=str))
    elif args.cmd == "audit-extract-provenance":
        from .calibration import extraction_audit_from_env

        result = extraction_audit_from_env().run(run_id=args.run_id)
        print(json.dumps(result, default=str))
    elif args.cmd == "budget-status":
        from .daemon import daemon_from_env

        service = daemon_from_env(with_workers=False)
        print(json.dumps(service.budget_status(args.run_id), default=str))
    elif args.cmd == "budget-resume":
        from .daemon import daemon_from_env

        service = daemon_from_env(with_workers=False)
        budgets = json.loads(args.budgets_json)
        print(json.dumps(service.resume_budget(args.run_id, budgets=budgets), default=str))
