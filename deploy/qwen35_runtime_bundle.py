#!/usr/bin/env python3
"""Build and verify the exact offline Qwen3.5 runtime dependency bundle.

The trusted artifact is a wheel bundle + reconstruction manifest, not a live venv.
The existing certified venv is read-only source evidence. Build writes only beneath
the requested output directory. Reconstruction writes only beneath its requested
venv/cache directories and never performs model inference.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from typing import Any


RUNTIME_CONTRACT = "qwen35-vllm-cu129-image-native-sampler-v3"
EXPECTED_FREEZE_SHA256 = "198d90f866d032ea8f747dd9d28a1fe5f5ef18c02e5eb64b850124e77a2744d0"
EXPECTED_FREEZE_LINES = 196
EXPECTED_VLLM_DIST = "0.30.0+cu129"
EXPECTED_VLLM_WHEEL_SHA256 = "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
EXPECTED_TORCH = "2.13.0+cu129"
EXPECTED_TORCHVISION = "0.28.0+cu129"
EXPECTED_TORCHCODEC = "0.16.0+cu129"
EXPECTED_CUDA_RUNTIME = "12.9"
PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CU129_INDEX = "https://download.pytorch.org/whl/cu129"
VLLM_WHEEL_URL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/"
    "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)

PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")
DIRECT_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s+@\s+(https://\S+)$")


class BundleError(RuntimeError):
    pass


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=capture,
        env=merged,
    )


def runtime_freeze(runtime: pathlib.Path) -> tuple[list[str], str, str]:
    python = runtime / "bin" / "python"
    if not python.is_file():
        raise BundleError(f"runtime python missing: {python}")
    result = run([str(python), "-m", "pip", "freeze"])
    lines = sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    )
    text = "\n".join(lines) + "\n"
    return lines, text, sha256_bytes(text.encode("utf-8"))


def parse_lock_line(line: str) -> dict[str, str]:
    match = PIN_RE.fullmatch(line)
    if match:
        name, version = match.groups()
        entry = {
            "name": name,
            "canonical_name": canonical_name(name),
            "kind": "pinned",
            "version": version,
            "requirement": line,
        }
        if entry["canonical_name"] == "vllm":
            if version != EXPECTED_VLLM_DIST:
                raise BundleError(f"unexpected pinned vLLM version: {line}")
            entry["download_requirement"] = (
                f"vllm @ {VLLM_WHEEL_URL}"
                f"#sha256={EXPECTED_VLLM_WHEEL_SHA256}"
            )
        return entry

    match = DIRECT_RE.fullmatch(line)
    if match:
        name, url = match.groups()
        allowed_vllm_urls = {
            VLLM_WHEEL_URL,
            f"{VLLM_WHEEL_URL}#sha256={EXPECTED_VLLM_WHEEL_SHA256}",
        }
        if canonical_name(name) != "vllm" or url not in allowed_vllm_urls:
            raise BundleError(
                "unsupported direct requirement; only the exact pinned vLLM "
                f"release wheel is authorized: {line}"
            )
        return {
            "name": name,
            "canonical_name": canonical_name(name),
            "kind": "direct_https",
            "version": EXPECTED_VLLM_DIST,
            "url": url,
            "requirement": line,
            "download_requirement": (
                f"vllm @ {VLLM_WHEEL_URL}"
                f"#sha256={EXPECTED_VLLM_WHEEL_SHA256}"
            ),
        }

    raise BundleError(f"unsupported/unpinned freeze requirement: {line}")


def parse_lock(lines: list[str]) -> list[dict[str, str]]:
    entries = [parse_lock_line(line) for line in lines]
    names = [entry["canonical_name"] for entry in entries]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise BundleError(f"duplicate normalized requirement names: {duplicates}")
    return entries


def read_ready(runtime: pathlib.Path) -> dict[str, Any]:
    path = runtime / ".workspace-control-ready.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleError(f"invalid runtime ready receipt: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError("runtime ready receipt must be an object")

    checks = {
        "runtime_contract_version":
            value.get("runtime_contract_version") == RUNTIME_CONTRACT,
        "vllm_distribution":
            value.get("vllm_distribution") == EXPECTED_VLLM_DIST,
        "vllm_release_wheel_sha256":
            value.get("vllm_release_wheel_sha256") == EXPECTED_VLLM_WHEEL_SHA256,
        "torch": value.get("torch") == EXPECTED_TORCH,
        "torch_cuda_runtime":
            value.get("torch_cuda_runtime") == EXPECTED_CUDA_RUNTIME,
        "torchvision": value.get("torchvision") == EXPECTED_TORCHVISION,
        "torchcodec": value.get("torchcodec") == EXPECTED_TORCHCODEC,
        "sampling_backend": value.get("sampling_backend") == "vllm_native",
        "vllm_flashinfer_sampler":
            value.get("vllm_flashinfer_sampler") is False,
        "prior_cuda_allocation_probe":
            value.get("cuda_allocation_probe") == "PASS",
        "prior_multimodal_import_probe":
            value.get("vllm_multimodal_import_probe") == "PASS",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise BundleError(f"runtime ready identity mismatch: {failed}")
    return value


def runtime_platform(runtime: pathlib.Path) -> dict[str, str]:
    code = r"""
import json, platform, sys, sysconfig
print(json.dumps({
  "python_version": platform.python_version(),
  "implementation": platform.python_implementation(),
  "machine": platform.machine(),
  "soabi": sysconfig.get_config_var("SOABI"),
  "executable": sys.executable,
}, sort_keys=True))
"""
    result = run([str(runtime / "bin" / "python"), "-c", code])
    value = json.loads(result.stdout)
    if value["machine"] != "x86_64":
        raise BundleError(f"unsupported runtime machine: {value['machine']}")
    if not str(value["python_version"]).startswith("3.12."):
        raise BundleError(f"expected Python 3.12 runtime: {value['python_version']}")
    return value


def wheel_metadata(path: pathlib.Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(candidates) != 1:
                raise BundleError(
                    f"wheel must contain exactly one dist-info METADATA: "
                    f"{path.name}: {candidates}"
                )
            raw = archive.read(candidates[0])
    except zipfile.BadZipFile as exc:
        raise BundleError(f"invalid wheel archive: {path}") from exc

    message = email.parser.BytesParser().parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise BundleError(f"wheel METADATA missing Name/Version: {path.name}")

    license_classifiers = [
        value
        for value in message.get_all("Classifier", [])
        if value.startswith("License ::")
    ]
    return {
        "name": str(name),
        "version": str(version),
        "license_expression": (
            str(message.get("License-Expression"))
            if message.get("License-Expression")
            else None
        ),
        "license": (
            str(message.get("License")).strip()
            if message.get("License") and str(message.get("License")).strip()
            else None
        ),
        "license_classifiers": sorted(license_classifiers),
    }


def artifact_record(path: pathlib.Path) -> dict[str, Any]:
    metadata = wheel_metadata(path)
    name = metadata["name"]
    version = metadata["version"]
    return {
        "filename": path.name,
        "name": name,
        "canonical_name": canonical_name(name),
        "version": version,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "license": {
            "expression": metadata["license_expression"],
            "declared": metadata["license"],
            "classifiers": metadata["license_classifiers"],
        },
    }


def verify_lock_against_wheels(
    entries: list[dict[str, str]],
    wheels: list[dict[str, Any]],
) -> None:
    by_name: dict[str, dict[str, Any]] = {}
    for wheel in wheels:
        key = wheel["canonical_name"]
        if key in by_name:
            raise BundleError(f"multiple wheel artifacts for {key}")
        by_name[key] = wheel

    expected = {entry["canonical_name"]: entry for entry in entries}
    missing = sorted(set(expected) - set(by_name))
    extra = sorted(set(by_name) - set(expected))
    if missing or extra:
        raise BundleError(
            f"wheel/lock package-set mismatch missing={missing} extra={extra}"
        )

    mismatches = []
    for key, entry in expected.items():
        wheel = by_name[key]
        if wheel["version"] != entry["version"]:
            mismatches.append(
                f"{key}: expected {entry['version']} got {wheel['version']}"
            )
    if mismatches:
        raise BundleError("wheel version mismatches: " + "; ".join(mismatches))

    vllm = by_name.get("vllm")
    if not vllm or vllm["sha256"] != EXPECTED_VLLM_WHEEL_SHA256:
        raise BundleError(
            "vLLM wheel digest does not match the certified release artifact"
        )


def write_offline_requirements(
    path: pathlib.Path,
    entries: list[dict[str, str]],
    wheels: list[dict[str, Any]],
) -> None:
    by_name = {wheel["canonical_name"]: wheel for wheel in wheels}
    lines = []
    for entry in sorted(entries, key=lambda item: item["canonical_name"]):
        wheel = by_name[entry["canonical_name"]]
        lines.append(
            f"{entry['name']}=={entry['version']} "
            f"--hash=sha256:{wheel['sha256']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plan(runtime: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    ready = read_ready(runtime)
    lines, freeze_text, freeze_sha = runtime_freeze(runtime)
    if len(lines) != EXPECTED_FREEZE_LINES:
        raise BundleError(
            f"freeze line count mismatch: expected {EXPECTED_FREEZE_LINES} "
            f"got {len(lines)}"
        )
    if freeze_sha != EXPECTED_FREEZE_SHA256:
        raise BundleError(
            f"freeze digest mismatch: expected {EXPECTED_FREEZE_SHA256} "
            f"got {freeze_sha}"
        )
    entries = parse_lock(lines)
    platform = runtime_platform(runtime)

    output.mkdir(parents=True, exist_ok=False)
    (output / "source-freeze.txt").write_text(freeze_text, encoding="utf-8")

    plan = {
        "schema_version": 1,
        "runtime_contract_version": RUNTIME_CONTRACT,
        "source_runtime": str(runtime),
        "source_freeze": {
            "line_count": len(lines),
            "sha256": freeze_sha,
        },
        "platform": {
            "python_version": platform["python_version"],
            "implementation": platform["implementation"],
            "machine": platform["machine"],
            "soabi": platform["soabi"],
        },
        "indexes": {
            "primary": PYPI_INDEX,
            "extra": [PYTORCH_CU129_INDEX],
            "vllm_release_wheel": VLLM_WHEEL_URL,
        },
        "environment": {
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "sampling_backend": "vllm_native",
        },
        "certified_identity": {
            "vllm_distribution": EXPECTED_VLLM_DIST,
            "vllm_release_wheel_sha256": EXPECTED_VLLM_WHEEL_SHA256,
            "torch": EXPECTED_TORCH,
            "torch_cuda_runtime": EXPECTED_CUDA_RUNTIME,
            "torchvision": EXPECTED_TORCHVISION,
            "torchcodec": EXPECTED_TORCHCODEC,
        },
        "source_ready_receipt_sha256": sha256_bytes(
            canonical_json(ready)
        ),
        "requirements": entries,
        "model_weights_included": False,
        "scientific_state_included": False,
        "production_credentials_included": False,
    }
    (output / "bundle-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def build_bundle(runtime: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    plan = build_plan(runtime, output)
    entries = plan["requirements"]
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()

    python = runtime / "bin" / "python"
    download_env = {
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }

    for index, entry in enumerate(entries, start=1):
        before = {p.name for p in wheelhouse.iterdir()}
        command = [
            str(python),
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheelhouse),
            "--no-deps",
            "--only-binary=:all:",
            "--index-url",
            PYPI_INDEX,
            "--extra-index-url",
            PYTORCH_CU129_INDEX,
            entry.get("download_requirement", entry["requirement"]),
        ]
        print(
            f"BUNDLE_DOWNLOAD {index}/{len(entries)} "
            f"{entry['canonical_name']}=={entry['version']}",
            flush=True,
        )
        run(command, env=download_env, capture=False)
        after = {p.name for p in wheelhouse.iterdir()}
        new = sorted(after - before)
        if len(new) != 1 or not new[0].endswith(".whl"):
            raise BundleError(
                f"expected exactly one new wheel for {entry['requirement']}; "
                f"got {new}"
            )

    paths = sorted(wheelhouse.glob("*.whl"), key=lambda p: p.name)
    wheels = [artifact_record(path) for path in paths]
    verify_lock_against_wheels(entries, wheels)

    requirements_path = output / "requirements.offline.txt"
    write_offline_requirements(requirements_path, entries, wheels)

    license_inventory = {
        "packages_with_expression": sum(
            1 for item in wheels if item["license"]["expression"]
        ),
        "packages_with_declared_license": sum(
            1 for item in wheels if item["license"]["declared"]
        ),
        "packages_with_license_classifier": sum(
            1 for item in wheels if item["license"]["classifiers"]
        ),
        "packages_without_any_license_metadata": sorted(
            item["canonical_name"]
            for item in wheels
            if not item["license"]["expression"]
            and not item["license"]["declared"]
            and not item["license"]["classifiers"]
        ),
    }

    manifest = {
        "schema_version": 1,
        "runtime_contract_version": RUNTIME_CONTRACT,
        "bundle_format": "python-wheelhouse-offline-v1",
        "source_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "source_freeze_line_count": EXPECTED_FREEZE_LINES,
        "platform": plan["platform"],
        "indexes": plan["indexes"],
        "environment": plan["environment"],
        "certified_identity": plan["certified_identity"],
        "artifacts": wheels,
        "artifact_count": len(wheels),
        "artifact_bytes": sum(item["size"] for item in wheels),
        "license_inventory": license_inventory,
        "license_review_completed": False,
        "offline_requirements_sha256": sha256_file(requirements_path),
        "model_weights_included": False,
        "scientific_state_included": False,
        "production_credentials_included": False,
    }
    manifest["manifest_identity_sha256"] = sha256_bytes(canonical_json(manifest))
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_bundle(output)
    print(
        json.dumps(
            {
                "decision": "PASS",
                "artifact_count": manifest["artifact_count"],
                "artifact_bytes": manifest["artifact_bytes"],
                "manifest_identity_sha256":
                    manifest["manifest_identity_sha256"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return manifest


def load_bundle(bundle: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = bundle / "bundle-manifest.json"
    plan_path = bundle / "bundle-plan.json"
    freeze_path = bundle / "source-freeze.txt"
    requirements_path = bundle / "requirements.offline.txt"

    for path in (manifest_path, plan_path, freeze_path, requirements_path):
        if not path.is_file():
            raise BundleError(f"bundle file missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze_text = freeze_path.read_text(encoding="utf-8")
    lines = [line for line in freeze_text.splitlines() if line]
    entries = parse_lock(lines)

    if sha256_bytes(freeze_text.encode("utf-8")) != EXPECTED_FREEZE_SHA256:
        raise BundleError("bundle source-freeze digest mismatch")
    if len(lines) != EXPECTED_FREEZE_LINES:
        raise BundleError("bundle source-freeze line count mismatch")
    if manifest.get("runtime_contract_version") != RUNTIME_CONTRACT:
        raise BundleError("bundle runtime contract mismatch")
    if manifest.get("source_freeze_sha256") != EXPECTED_FREEZE_SHA256:
        raise BundleError("manifest source freeze mismatch")
    if plan.get("runtime_contract_version") != RUNTIME_CONTRACT:
        raise BundleError("bundle plan runtime contract mismatch")
    if manifest.get("offline_requirements_sha256") != sha256_file(
        requirements_path
    ):
        raise BundleError("offline requirements digest mismatch")

    identity = manifest.get("manifest_identity_sha256")
    without_identity = dict(manifest)
    without_identity.pop("manifest_identity_sha256", None)
    if identity != sha256_bytes(canonical_json(without_identity)):
        raise BundleError("bundle manifest identity mismatch")

    return manifest, entries


def verify_bundle(bundle: pathlib.Path) -> dict[str, Any]:
    manifest, entries = load_bundle(bundle)
    wheelhouse = bundle / "wheelhouse"
    if not wheelhouse.is_dir():
        raise BundleError("wheelhouse missing")

    actual_paths = sorted(wheelhouse.iterdir(), key=lambda p: p.name)
    if any(not p.is_file() or p.suffix != ".whl" for p in actual_paths):
        raise BundleError("wheelhouse contains non-wheel or non-regular entries")

    actual = [artifact_record(path) for path in actual_paths]
    expected = sorted(
        manifest["artifacts"], key=lambda item: item["filename"]
    )
    if actual != expected:
        raise BundleError("wheelhouse artifact manifest mismatch")

    verify_lock_against_wheels(entries, actual)
    if manifest["artifact_count"] != len(actual):
        raise BundleError("bundle artifact_count mismatch")
    if manifest["artifact_bytes"] != sum(item["size"] for item in actual):
        raise BundleError("bundle artifact_bytes mismatch")

    result = {
        "decision": "PASS",
        "runtime_contract_version": RUNTIME_CONTRACT,
        "artifact_count": len(actual),
        "artifact_bytes": sum(item["size"] for item in actual),
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
    }
    return result


def reconstruct(
    bundle: pathlib.Path,
    venv: pathlib.Path,
    *,
    python: str,
) -> dict[str, Any]:
    manifest, _ = load_bundle(bundle)
    verify_bundle(bundle)
    if venv.exists():
        raise BundleError(f"reconstruction destination already exists: {venv}")

    host = run(
        [
            python,
            "-c",
            (
                "import json,platform,sysconfig;"
                "print(json.dumps({'python_version':platform.python_version(),"
                "'machine':platform.machine(),"
                "'soabi':sysconfig.get_config_var('SOABI')},sort_keys=True))"
            ),
        ]
    )
    host_identity = json.loads(host.stdout)
    expected_platform = manifest["platform"]
    if host_identity["machine"] != expected_platform["machine"]:
        raise BundleError("reconstruction machine mismatch")
    if host_identity["soabi"] != expected_platform["soabi"]:
        raise BundleError("reconstruction Python SOABI mismatch")
    if host_identity["python_version"] != expected_platform["python_version"]:
        raise BundleError("reconstruction Python version mismatch")

    run([python, "-m", "venv", str(venv)], capture=False)
    cache_dir = venv.parent / f".{venv.name}-pip-cache"
    cache_dir.mkdir()
    env = {
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PIP_CACHE_DIR": str(cache_dir),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    try:
        run(
            [
                str(venv / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(bundle / "wheelhouse"),
                "--require-hashes",
                "-r",
                str(bundle / "requirements.offline.txt"),
            ],
            env=env,
            capture=False,
        )

        probe = r"""
import importlib.metadata, json, os, sys
import torch

vllm_dist = importlib.metadata.distribution("vllm")
direct = json.loads(vllm_dist.read_text("direct_url.json") or "{}")
payload = {
  "vllm_distribution": vllm_dist.version,
  "vllm_direct_url_hash": direct.get("archive_info", {}).get("hash"),
  "torch": torch.__version__,
  "torch_cuda_runtime": torch.version.cuda,
  "torchvision": importlib.metadata.version("torchvision"),
  "torchcodec": importlib.metadata.version("torchcodec"),
  "flashinfer_sampler_env": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
}
print(json.dumps(payload, sort_keys=True))
"""
        result = run(
            [str(venv / "bin" / "python"), "-c", probe],
            env=env,
        )
        identity = json.loads(result.stdout)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    expected = {
        "vllm_distribution": EXPECTED_VLLM_DIST,
        "torch": EXPECTED_TORCH,
        "torch_cuda_runtime": EXPECTED_CUDA_RUNTIME,
        "torchvision": EXPECTED_TORCHVISION,
        "torchcodec": EXPECTED_TORCHCODEC,
        "flashinfer_sampler_env": "0",
    }
    actual_core = {
        key: identity.get(key)
        for key in expected
    }
    if actual_core != expected:
        raise BundleError(
            f"reconstructed core identity mismatch expected={expected} "
            f"actual={actual_core}"
        )

    return {
        "decision": "PASS",
        "runtime_contract_version": RUNTIME_CONTRACT,
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "network_permitted": False,
        "model_inference_performed": False,
        "core_identity": identity,
        "venv": str(venv),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--runtime", type=pathlib.Path, required=True)
    plan.add_argument("--output", type=pathlib.Path, required=True)

    build = sub.add_parser("build")
    build.add_argument("--runtime", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=pathlib.Path, required=True)

    rebuild = sub.add_parser("reconstruct")
    rebuild.add_argument("--bundle", type=pathlib.Path, required=True)
    rebuild.add_argument("--venv", type=pathlib.Path, required=True)
    rebuild.add_argument("--python", default="/usr/bin/python3.12")

    args = parser.parse_args()

    try:
        if args.command == "plan":
            value = build_plan(args.runtime, args.output)
            print(json.dumps(value, indent=2, sort_keys=True))
        elif args.command == "build":
            build_bundle(args.runtime, args.output)
        elif args.command == "verify":
            print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))
        elif args.command == "reconstruct":
            print(
                json.dumps(
                    reconstruct(
                        args.bundle,
                        args.venv,
                        python=args.python,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            raise AssertionError(args.command)
    except (BundleError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
