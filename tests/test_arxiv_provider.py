from email.message import Message
from pathlib import Path

import pytest

from research_engine.arxiv import ArxivProvider, normalize_arxiv_id, versioned_pdf_url
from research_engine.artifact_store import ContentAddressedArtifactStore


class FakeResponse:
    def __init__(self, body: bytes, *, url: str, content_type: str = "application/pdf"):
        self._body = body
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body

    def geturl(self):
        return self._url


def test_normalize_and_versioned_pdf_url():
    assert normalize_arxiv_id("arXiv:2606.12718v1") == "2606.12718v1"
    assert normalize_arxiv_id("https://arxiv.org/abs/2605.15593v1") == "2605.15593v1"
    assert versioned_pdf_url("2606.12718v1") == "https://arxiv.org/pdf/2606.12718v1"


def test_provider_writes_exact_pdf_to_content_addressed_store(tmp_path: Path):
    pdf = b"%PDF-1.7\nexact-test-bytes\n%%EOF\n"

    def opener(request, timeout):
        assert request.full_url == "https://arxiv.org/pdf/2606.12718v1"
        assert timeout == 120
        return FakeResponse(pdf, url=request.full_url)

    store = ContentAddressedArtifactStore(tmp_path)
    result = ArxivProvider(store, opener=opener).acquire_pdf("2606.12718v1")

    assert result.arxiv_id == "2606.12718v1"
    assert result.receipt.path.read_bytes() == pdf
    assert result.receipt.byte_size == len(pdf)
    assert result.receipt.artifact_id.startswith("source-artifact:sha256:")


def test_provider_rejects_non_pdf(tmp_path: Path):
    def opener(request, timeout):
        return FakeResponse(
            b"<html>blocked</html>",
            url=request.full_url,
            content_type="text/html",
        )

    store = ContentAddressedArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="did not return PDF bytes"):
        ArxivProvider(store, opener=opener).acquire_pdf("2605.15593v1")


def test_provider_rejects_unexpected_redirect_host(tmp_path: Path):
    def opener(request, timeout):
        return FakeResponse(
            b"%PDF-1.7\nbytes\n",
            url="https://example.invalid/paper.pdf",
        )

    store = ContentAddressedArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="unexpected arXiv redirect host"):
        ArxivProvider(store, opener=opener).acquire_pdf("2605.15593v1")
