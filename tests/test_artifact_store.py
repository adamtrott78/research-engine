from pathlib import Path

from research_engine.artifact_store import ContentAddressedArtifactStore


def test_content_addressed_store_is_deterministic_and_deduplicates(tmp_path: Path):
    store = ContentAddressedArtifactStore(tmp_path)
    a = store.put_bytes(b"scientific evidence")
    b = store.put_bytes(b"scientific evidence")
    assert a.sha256 == b.sha256
    assert a.path == b.path
    assert a.path.read_bytes() == b"scientific evidence"
    assert a.artifact_id.startswith("source-artifact:sha256:")
