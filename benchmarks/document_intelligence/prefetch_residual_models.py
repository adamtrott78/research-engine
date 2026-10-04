#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

GIB = 1024 ** 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
    )


def license_from_info(info: Any, readme: Path) -> str | None:
    card = getattr(info, "card_data", None)
    if card is not None:
        if isinstance(card, dict):
            value = card.get("license")
        else:
            value = getattr(card, "license", None)
        if value:
            return str(value).strip().lower()
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
            if line.lower().startswith("license:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'").lower()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    disk_before = shutil.disk_usage(args.model_root)
    minimum = int(contract["required_state"]["minimum_free_disk_gib_before_download"]) * GIB
    if disk_before.free < minimum:
        raise SystemExit(
            f"insufficient free disk: {disk_before.free / GIB:.2f} GiB < {minimum / GIB:.2f} GiB"
        )

    api = HfApi(token=False)
    rows: list[dict[str, Any]] = []
    decision = "PASS"
    failure: str | None = None

    try:
        for spec in contract["model_artifacts"]:
            requested = spec["requested_revision"]
            info = api.model_info(spec["repo_id"], revision=requested, token=False)
            resolved = str(info.sha)
            if spec["revision_mode"] == "PINNED" and resolved != requested:
                raise RuntimeError(
                    f"{spec['id']}: pinned revision mismatch {requested} != {resolved}"
                )

            destination = args.model_root / spec["id"] / resolved
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=resolved,
                local_dir=destination,
                token=False,
            )
            # Prove the exact resolved revision can be reopened without network.
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=resolved,
                local_dir=destination,
                token=False,
                local_files_only=True,
            )

            files = artifact_files(destination)
            file_rows = []
            total = 0
            for path in files:
                size = path.stat().st_size
                total += size
                file_rows.append(
                    {
                        "path": path.relative_to(destination).as_posix(),
                        "bytes": size,
                        "sha256": sha256_file(path),
                    }
                )
            if not file_rows:
                raise RuntimeError(f"{spec['id']}: no model files materialized")

            expected_license = spec["expected_license"].lower()
            observed_license = license_from_info(info, destination / "README.md")
            if observed_license != expected_license:
                raise RuntimeError(
                    f"{spec['id']}: license {observed_license!r} != {expected_license!r}"
                )

            ceiling = float(spec["maximum_materialized_gib"]) * GIB
            if total > ceiling:
                raise RuntimeError(
                    f"{spec['id']}: {total / GIB:.3f} GiB exceeds {ceiling / GIB:.3f} GiB"
                )

            rows.append(
                {
                    "id": spec["id"],
                    "family": spec["family"],
                    "repo_id": spec["repo_id"],
                    "requested_revision": requested,
                    "revision_mode": spec["revision_mode"],
                    "resolved_revision": resolved,
                    "license": observed_license,
                    "materialized_bytes": total,
                    "file_count": len(file_rows),
                    "files": file_rows,
                    "offline_reopen": True,
                }
            )
    except Exception as exc:
        decision = "REJECT"
        failure = f"{type(exc).__name__}: {exc}"

    disk_after = shutil.disk_usage(args.model_root)
    report = {
        "schema_version": 1,
        "purpose": "bounded_residual_model_artifact_prefetch",
        "decision": decision,
        "candidate_families": sorted({row["family"] for row in rows}),
        "candidate_model_inference_calls": 0,
        "scientific_state_materialized": False,
        "network_use": "MODEL_ARTIFACT_PREFETCH_ONLY",
        "model_root": str(args.model_root),
        "disk_free_bytes_before": disk_before.free,
        "disk_free_bytes_after": disk_after.free,
        "artifacts": rows,
        "failure": failure,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "decision", "candidate_model_inference_calls", "scientific_state_materialized",
        "disk_free_bytes_before", "disk_free_bytes_after", "failure"
    )}, indent=2, sort_keys=True))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
