from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class ArtifactReceipt:
    sha256: str
    byte_size: int
    path: Path

    @property
    def artifact_id(self) -> str:
        return f"source-artifact:sha256:{self.sha256}"


class ContentAddressedArtifactStore:
    """Immutable SHA-256-addressed byte store with atomic writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for_hash(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return self.root / "sha256" / sha256[:2] / sha256[2:4] / sha256

    def put_bytes(self, data: bytes) -> ArtifactReceipt:
        sha = hashlib.sha256(data).hexdigest()
        dest = self.path_for_hash(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            self._verify_existing(dest, sha, len(data))
            return ArtifactReceipt(sha, len(data), dest)
        fd, tmp_name = tempfile.mkstemp(prefix=".incoming-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        self._verify_existing(dest, sha, len(data))
        return ArtifactReceipt(sha, len(data), dest)

    def put_file(self, source: str | Path) -> ArtifactReceipt:
        source = Path(source)
        h = hashlib.sha256()
        size = 0
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
                size += len(chunk)
        sha = h.hexdigest()
        dest = self.path_for_hash(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            self._verify_existing(dest, sha, size)
            return ArtifactReceipt(sha, size, dest)
        fd, tmp_name = tempfile.mkstemp(prefix=".incoming-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
                for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        self._verify_existing(dest, sha, size)
        return ArtifactReceipt(sha, size, dest)

    def _verify_existing(self, path: Path, expected_hash: str, expected_size: int) -> None:
        if path.stat().st_size != expected_size:
            raise IOError(f"artifact size mismatch: {path}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected_hash:
            raise IOError(f"artifact hash mismatch: {path}")
