import inspect
from pathlib import Path

from research_engine.artifact_store import ContentAddressedArtifactStore
from research_engine.document_sidecar import (
    DocumentSidecarCompiler,
    document_sidecar_compiler_from_env,
)
from research_engine.deterministic_executor import DeterministicResearchExecutor
from research_engine import worker_runtime


class _Store:
    pass


class _FakeCompiler(DocumentSidecarCompiler):
    def __init__(self, *, artifact_root: Path, cache_root: Path):
        super().__init__(
            store=_Store(),
            artifact_root=artifact_root,
            cache_root=cache_root,
            repo_root=cache_root,
            native_runtime=cache_root,
            standard_runtime=cache_root,
            standard_model_cache=cache_root,
            local_inference=None,
            specialized_enabled=False,
        )
        self.events = []
        self.builds = 0

    def _record_event(self, **kwargs):
        self.events.append(kwargs)

    def _build_sidecar(
        self,
        *,
        source_artifact_id,
        source_sha256,
        source_pdf,
        work_root,
    ):
        self.builds += 1
        package = work_root / "packages" / "native" / "paper-package-v1"
        package.mkdir(parents=True)
        (package / "package.json").write_text("{}\n", encoding="utf-8")
        standard = work_root / "packages" / "standard" / "paper-package-v1"
        standard.mkdir(parents=True)
        (standard / "package.json").write_text("{}\n", encoding="utf-8")
        return {
            "routing_summary": {
                "authority": "AUTHORITATIVE_PDF",
                "semantic_baseline": "AUTHORITATIVE_PDF_HYDRATOR",
                "native_status": "COMPLETED",
                "standard_status": "COMPLETED",
                "formula_pages": 0,
                "fast_empty_tables": 0,
                "accurate_pages": 0,
                "paddle_fallbacks": 0,
            },
            "local_inference_summary": {"broker_before": {"enabled": False}},
            "native_package_path": str(package / "package.json"),
            "standard_package_path": str(standard / "package.json"),
        }


def test_sidecar_cache_miss_then_hit_is_content_addressed(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    cache_root = tmp_path / "sidecars"
    store = ContentAddressedArtifactStore(artifact_root)
    receipt = store.put_bytes(b"%PDF-1.4\nsidecar-test\n%%EOF\n")
    compiler = _FakeCompiler(
        artifact_root=artifact_root,
        cache_root=cache_root,
    )

    first = compiler.compile(
        source_artifact_id=receipt.artifact_id,
        source_sha256=receipt.sha256,
        source_pdf=receipt.path,
    )
    second = compiler.compile(
        source_artifact_id=receipt.artifact_id,
        source_sha256=receipt.sha256,
        source_pdf=receipt.path,
    )

    assert compiler.builds == 1
    assert first["status"] == "COMPLETED"
    assert first["cache_hit"] is False
    assert first["authority"] == "DERIVED_SIDECAR_ONLY"
    assert first["semantic_baseline"] == "AUTHORITATIVE_PDF_HYDRATOR"
    assert second["status"] == "CACHE_HIT"
    assert second["cache_hit"] is True
    assert Path(first["cache_path"]) == compiler.cache_path(receipt.sha256)
    assert [event["status"] for event in compiler.events] == [
        "COMPLETED",
        "CACHE_HIT",
    ]
    cache_event = compiler.events[1]["local_inference_summary"]
    assert cache_event["cache_hit"] is True
    assert cache_event["formula_requests"] == 0
    assert cache_event["table_requests"] == 0


def test_sidecar_rejects_non_authoritative_source_path(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    cache_root = tmp_path / "sidecars"
    store = ContentAddressedArtifactStore(artifact_root)
    receipt = store.put_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(receipt.path.read_bytes())
    compiler = _FakeCompiler(
        artifact_root=artifact_root,
        cache_root=cache_root,
    )

    try:
        compiler.compile(
            source_artifact_id=receipt.artifact_id,
            source_sha256=receipt.sha256,
            source_pdf=copy,
        )
    except ValueError as exc:
        assert "authoritative content-addressed artifact path" in str(exc)
    else:
        raise AssertionError("non-authoritative path should be rejected")


def test_sidecar_factory_is_explicitly_feature_gated(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RESEARCH_ENGINE_DOCUMENT_SIDECAR_ENABLED", "0")
    assert (
        document_sidecar_compiler_from_env(
            store=_Store(),
            artifact_root=tmp_path,
        )
        is None
    )


def test_deterministic_acquire_sidecar_is_fail_open_by_contract():
    source = inspect.getsource(DeterministicResearchExecutor._acquire)
    assert "if self.document_sidecar is not None" in source
    assert "FAILED_DERIVED_SIDECAR" in source
    assert "AUTHORITATIVE_PDF_UNCHANGED" in source
    assert "raise" not in source.split("if self.document_sidecar is not None", 1)[1]


def test_worker_factory_wires_sidecar_to_shared_local_inference():
    source = inspect.getsource(worker_runtime.deterministic_worker_from_env)
    assert "document_sidecar_compiler_from_env" in source
    assert "local_inference=local_inference" in source
    assert "document_sidecar=document_sidecar" in source


def test_schema_contains_document_sidecar_telemetry():
    schema = (
        Path(__file__).parents[1] / "sql" / "schema.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS document_sidecar_events" in schema
    assert "FAILED_DERIVED_SIDECAR" in schema
    assert "routing_summary JSONB" in schema
    assert "local_inference_summary JSONB" in schema



def test_sidecar_discovers_hash_named_docling_runtime_json(tmp_path: Path):
    output = tmp_path / "raw" / "native"
    output.mkdir(parents=True)
    runtime_json = output / (
        "054aaba60e8b6cc4dfc5a35dab9f1adc6584bb9ef6ea68b2b9ced1cb71d52f22.json"
    )
    runtime_json.write_text("{}\n", encoding="utf-8")

    assert DocumentSidecarCompiler._single_runtime_json(output) == runtime_json


def test_sidecar_runtime_json_discovery_fails_closed_on_ambiguous_outputs(
    tmp_path: Path,
):
    output = tmp_path / "raw"
    output.mkdir()
    (output / "a.json").write_text("{}\n", encoding="utf-8")
    (output / "b.json").write_text("{}\n", encoding="utf-8")

    try:
        DocumentSidecarCompiler._single_runtime_json(output)
    except RuntimeError as exc:
        assert "expected exactly one runtime JSON" in str(exc)
        assert "a.json" in str(exc)
        assert "b.json" in str(exc)
    else:
        raise AssertionError("ambiguous runtime outputs must fail closed")


def test_sidecar_does_not_hardcode_source_json_runtime_name():
    source = inspect.getsource(DocumentSidecarCompiler._build_sidecar)
    assert '"source.json"' not in source
    assert "_single_runtime_json(raw_native)" in source
    assert "_single_runtime_json(raw_standard)" in source
    assert "_single_runtime_json(output_dir)" in source
