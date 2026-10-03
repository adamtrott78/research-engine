from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .local_inference import local_inference_service_from_env


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.gpu_peak_used_mib: dict[str, int] = {}
        self.gpu_peak_utilization: dict[str, int] = {}
        self.mem_total_kib: int | None = None
        self.mem_min_available_kib: int | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _sample_gpu(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if proc.returncode != 0:
            return
        for line in proc.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            index, used, util = parts
            try:
                used_i = int(used)
                util_i = int(util)
            except ValueError:
                continue
            self.gpu_peak_used_mib[index] = max(
                self.gpu_peak_used_mib.get(index, 0),
                used_i,
            )
            self.gpu_peak_utilization[index] = max(
                self.gpu_peak_utilization.get(index, 0),
                util_i,
            )

    def _sample_memory(self) -> None:
        try:
            lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        values: dict[str, int] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            token = raw.strip().split()[0]
            try:
                values[key] = int(token)
            except ValueError:
                continue
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is not None:
            self.mem_total_kib = total
        if available is not None:
            if self.mem_min_available_kib is None:
                self.mem_min_available_kib = available
            else:
                self.mem_min_available_kib = min(
                    self.mem_min_available_kib,
                    available,
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_gpu()
            self._sample_memory()
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def report(self) -> dict[str, Any]:
        peak_used_kib = None
        if self.mem_total_kib is not None and self.mem_min_available_kib is not None:
            peak_used_kib = self.mem_total_kib - self.mem_min_available_kib
        return {
            "samples": self.samples,
            "gpu_peak_used_mib": dict(sorted(self.gpu_peak_used_mib.items())),
            "gpu_peak_utilization_percent": dict(
                sorted(self.gpu_peak_utilization.items())
            ),
            "system_mem_total_kib": self.mem_total_kib,
            "system_mem_peak_used_kib": peak_used_kib,
            "system_mem_min_available_kib": self.mem_min_available_kib,
        }


def _formula_targets(plan: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in plan.get("cases", []):
        for page in case.get("target_pages", []):
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "source_location": str(case["source_location"]),
                    "page_number": int(page),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _table_targets(plan: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in plan.get("targets", []):
        bbox = target.get("pdf_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        rows.append(
            {
                "case_id": str(target["case_id"]),
                "source_location": str(target["source_location"]),
                "page_number": int(target["page_number"]),
                "table_asset_id": str(target["table_asset_id"]),
                "pdf_bbox": [float(x) for x in bbox],
            }
        )
        if len(rows) >= limit:
            return rows
    return rows


def run_concurrency_benchmark(
    *,
    bundle_root: str | Path,
    formula_plan_path: str | Path,
    table_plan_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    bundle = Path(bundle_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    formula_plan = json.loads(
        Path(formula_plan_path).read_text(encoding="utf-8")
    )
    table_plan = json.loads(
        Path(table_plan_path).read_text(encoding="utf-8")
    )
    formula_targets = _formula_targets(formula_plan)
    table_targets = _table_targets(table_plan)
    if len(formula_targets) < 2:
        raise ValueError("concurrency gate requires at least two formula targets")
    if len(table_targets) < 2:
        raise ValueError("concurrency gate requires at least two table targets")

    service = local_inference_service_from_env(
        log_root=output / "logs",
        require_enabled=True,
    )
    assert service is not None
    sampler = ResourceSampler()
    sampler.start()
    started = time.monotonic()

    def formula_call(target: dict[str, Any], label: str) -> dict[str, Any]:
        source = bundle / target["source_location"]
        result = service.formula_page(
            source_pdf=source,
            page_number=target["page_number"],
            output_dir=output / "requests" / label,
        )
        return {
            "kind": "formula",
            "target": target,
            "server_pid": result["server_pid"],
            "model_load_count": result["model_load_count"],
            "request_seq": result["request_seq"],
            "result": result["result"],
        }

    def table_call(target: dict[str, Any], label: str) -> dict[str, Any]:
        source = bundle / target["source_location"]
        result = service.table_crop(
            source_pdf=source,
            page_number=target["page_number"],
            pdf_bbox=target["pdf_bbox"],
            output_dir=output / "requests" / label,
        )
        return {
            "kind": "table",
            "target": target,
            "server_pid": result["server_pid"],
            "model_load_count": result["model_load_count"],
            "request_seq": result["request_seq"],
            "result": result["result"],
        }

    first_wave: list[dict[str, Any]] = []
    first_wave_started = time.monotonic()
    try:
        futures = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for index, target in enumerate(formula_targets):
                futures.append(
                    pool.submit(
                        formula_call,
                        target,
                        f"formula-wave1-{index}",
                    )
                )
            for index, target in enumerate(table_targets):
                futures.append(
                    pool.submit(
                        table_call,
                        target,
                        f"table-wave1-{index}",
                    )
                )
            for future in as_completed(futures):
                first_wave.append(future.result())
        first_wave_seconds = time.monotonic() - first_wave_started

        warm_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            warm_formula = pool.submit(
                formula_call,
                formula_targets[0],
                "formula-warm-reuse",
            )
            warm_table = pool.submit(
                table_call,
                table_targets[0],
                "table-warm-reuse",
            )
            warm_wave = [warm_formula.result(), warm_table.result()]
        warm_wave_seconds = time.monotonic() - warm_started

        health = service.health()
        elapsed = time.monotonic() - started
    finally:
        sampler.stop()
        service.close()

    formula_rows = [
        row for row in first_wave + warm_wave if row["kind"] == "formula"
    ]
    table_rows = [
        row for row in first_wave + warm_wave if row["kind"] == "table"
    ]
    formula_pids = sorted({int(row["server_pid"]) for row in formula_rows})
    table_pids = sorted({int(row["server_pid"]) for row in table_rows})
    formula_loads = sorted(
        {int(row["model_load_count"]) for row in formula_rows}
    )
    table_loads = sorted(
        {int(row["model_load_count"]) for row in table_rows}
    )

    checks = {
        "formula_single_owner_pid": len(formula_pids) == 1,
        "table_single_owner_pid": len(table_pids) == 1,
        "formula_single_model_load": formula_loads == [1],
        "table_single_model_load": table_loads == [1],
        "formula_spawn_count_one": health["formula"]["spawn_count"] == 1,
        "table_spawn_count_one": health["table"]["spawn_count"] == 1,
        "no_failed_requests": (
            health["formula"]["requests_failed"] == 0
            and health["table"]["requests_failed"] == 0
        ),
        "queues_drained": (
            health["formula"]["queue_depth"] == 0
            and health["table"]["queue_depth"] == 0
        ),
    }
    decision = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "schema_version": 1,
        "mode": "shared_local_document_inference_concurrency",
        "decision": decision,
        "first_wave_seconds": first_wave_seconds,
        "warm_mixed_wave_seconds": warm_wave_seconds,
        "elapsed_seconds": elapsed,
        "formula_targets": formula_targets,
        "table_targets": table_targets,
        "formula_server_pids": formula_pids,
        "table_server_pids": table_pids,
        "formula_model_load_counts": formula_loads,
        "table_model_load_counts": table_loads,
        "checks": checks,
        "service_health_before_close": health,
        "resources": sampler.report(),
        "requests": {
            "first_wave": sorted(
                first_wave,
                key=lambda row: (
                    row["kind"],
                    row["target"]["case_id"],
                    row["target"]["page_number"],
                ),
            ),
            "warm_wave": warm_wave,
        },
    }
    (output / "local-inference-concurrency-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--formula-plan", required=True)
    parser.add_argument("--table-plan", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    report = run_concurrency_benchmark(
        bundle_root=args.bundle_root,
        formula_plan_path=args.formula_plan,
        table_plan_path=args.table_plan,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "first_wave_seconds": report["first_wave_seconds"],
                "warm_mixed_wave_seconds": report["warm_mixed_wave_seconds"],
                "formula_server_pids": report["formula_server_pids"],
                "table_server_pids": report["table_server_pids"],
                "checks": report["checks"],
                "resources": report["resources"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
