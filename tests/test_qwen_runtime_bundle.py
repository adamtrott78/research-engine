import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "deploy" / "qwen35_runtime_bundle.py"
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "qwen35-runtime-bundle-contract.v1.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("qwen35_runtime_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_wheel(
    path: Path,
    *,
    name: str,
    version: str,
    vendored_metadata: bool = False,
):
    dist = name.replace("-", "_")
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist}-{version}.dist-info/METADATA", metadata)
        if vendored_metadata:
            archive.writestr(
                f"{dist}/_vendor/example-1.0.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: example\nVersion: 1.0.0\n\n",
            )


def test_freeze_parser_accepts_only_exact_pins_and_certified_vllm_url():
    bundle = load_module()

    pinned = bundle.parse_lock_line("torch==2.13.0+cu129")
    assert pinned["canonical_name"] == "torch"
    assert pinned["version"] == "2.13.0+cu129"
    assert pinned["kind"] == "pinned"

    direct = bundle.parse_lock_line(
        "vllm @ "
        "https://github.com/vllm-project/vllm/releases/download/v0.30.0/"
        "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    assert direct["canonical_name"] == "vllm"
    assert direct["version"] == "0.30.0+cu129"
    assert direct["kind"] == "direct_https"

    observed_local = bundle.parse_lock_line(
        "vllm @ "
        "file:///home/atrott/adamArchives/Adam/.research-engine/"
        "residual-runtime-wheels/vllm-0.30.0-cu129/"
        "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256="
        "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
    )
    assert observed_local["canonical_name"] == "vllm"
    assert observed_local["version"] == "0.30.0+cu129"
    assert observed_local["kind"] == "direct_file_certified"
    assert observed_local["download_requirement"].startswith(
        "vllm @ https://github.com/vllm-project/vllm/releases/"
    )
    assert bundle.EXPECTED_VLLM_WHEEL_SHA256 in (
        observed_local["download_requirement"]
    )

    for rejected in [
        "transformers",
        "foo>=1.0",
        "-e git+https://example.invalid/repo.git",
        "foo @ file:///tmp/foo.whl",
        "foo @ https://example.invalid/foo.whl",
        (
            "vllm @ file:///tmp/"
            "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
            "#sha256=" + "0" * 64
        ),
        (
            "vllm @ file:///tmp/vllm-wrong.whl#sha256="
            + bundle.EXPECTED_VLLM_WHEEL_SHA256
        ),
    ]:
        with pytest.raises(bundle.BundleError):
            bundle.parse_lock_line(rejected)


def test_lock_rejects_duplicate_normalized_names():
    bundle = load_module()
    with pytest.raises(bundle.BundleError, match="duplicate"):
        bundle.parse_lock(["foo_bar==1.0", "foo-bar==1.0"])


def test_wheel_metadata_ignores_nested_vendored_dist_info(tmp_path: Path):
    bundle = load_module()
    wheel = tmp_path / "setuptools-78.1.0-py3-none-any.whl"
    make_wheel(
        wheel,
        name="setuptools",
        version="78.1.0",
        vendored_metadata=True,
    )

    metadata = bundle.wheel_metadata(wheel)
    assert metadata["name"] == "setuptools"
    assert metadata["version"] == "78.1.0"


def test_wheel_metadata_and_lock_verification_are_exact(tmp_path: Path):
    bundle = load_module()

    torch_wheel = tmp_path / "torch-2.13.0+cu129-py3-none-any.whl"
    vllm_wheel = tmp_path / "vllm-0.30.0+cu129-py3-none-any.whl"
    make_wheel(torch_wheel, name="torch", version="2.13.0+cu129")
    make_wheel(vllm_wheel, name="vllm", version="0.30.0+cu129")

    torch_record = bundle.artifact_record(torch_wheel)
    vllm_record = bundle.artifact_record(vllm_wheel)
    vllm_record["sha256"] = bundle.EXPECTED_VLLM_WHEEL_SHA256

    entries = [
        bundle.parse_lock_line("torch==2.13.0+cu129"),
        bundle.parse_lock_line(
            "vllm @ "
            "https://github.com/vllm-project/vllm/releases/download/v0.30.0/"
            "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
    ]

    bundle.verify_lock_against_wheels(entries, [torch_record, vllm_record])

    wrong = dict(torch_record)
    wrong["version"] = "0"
    with pytest.raises(bundle.BundleError, match="version"):
        bundle.verify_lock_against_wheels(entries, [wrong, vllm_record])


def test_offline_requirements_pin_versions_and_artifact_hashes(tmp_path: Path):
    bundle = load_module()
    entries = [
        bundle.parse_lock_line("torch==2.13.0+cu129"),
        bundle.parse_lock_line(
            "vllm @ "
            "https://github.com/vllm-project/vllm/releases/download/v0.30.0/"
            "vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        ),
    ]
    wheels = [
        {
            "canonical_name": "torch",
            "sha256": "1" * 64,
        },
        {
            "canonical_name": "vllm",
            "sha256": "2" * 64,
        },
    ]
    path = tmp_path / "requirements.offline.txt"
    bundle.write_offline_requirements(path, entries, wheels)
    text = path.read_text()
    assert "torch==2.13.0+cu129 --hash=sha256:" + "1" * 64 in text
    assert "vllm==0.30.0+cu129 --hash=sha256:" + "2" * 64 in text
    assert "https://" not in text


def test_bundle_contract_separates_persistent_wheels_from_ephemeral_venv():
    contract = json.loads(CONTRACT.read_text())
    assert contract["representation"]["bundle_format"] == "python-wheelhouse-offline-v1"
    assert contract["representation"]["persistent_live_venv"] is False
    assert contract["representation"]["wheel_only"] is True
    assert contract["representation"]["source_distributions_allowed"] is False
    assert contract["representation"]["model_weights_included"] is False
    assert contract["reconstruction"]["network"] == "forbidden"
    assert contract["reconstruction"]["destination"] == "ephemeral per-job venv only"
    assert "--no-index" in contract["reconstruction"]["pip_flags"]
    assert "--require-hashes" in contract["reconstruction"]["pip_flags"]


def test_builder_is_wheel_only_and_reconstruction_is_offline():
    text = SCRIPT.read_text()
    assert '"--only-binary=:all:"' in text
    assert '"--no-deps"' in text
    assert '"--no-index"' in text
    assert '"--require-hashes"' in text
    assert 'PIP_NO_INDEX": "1"' in text
    assert "LLM(" not in text
    assert "SamplingParams" not in text
    assert "model inference" in text.lower()
