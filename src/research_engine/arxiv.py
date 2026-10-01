from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .artifact_store import ArtifactReceipt, ContentAddressedArtifactStore


@dataclass(frozen=True)
class ArxivAcquisition:
    arxiv_id: str
    versioned_pdf_url: str
    resolved_url: str
    media_type: str
    receipt: ArtifactReceipt


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arXiv:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    if not value:
        raise ValueError("arXiv id is empty")
    return value


def versioned_pdf_url(arxiv_id: str) -> str:
    aid = normalize_arxiv_id(arxiv_id)
    return f"https://arxiv.org/pdf/{aid}"


class ArxivProvider:
    """Acquire immutable arXiv PDF bytes into the content-addressed artifact store."""

    def __init__(
        self,
        artifact_store: ContentAddressedArtifactStore,
        *,
        opener=urlopen,
        user_agent: str = "research-engine/0.1 exact-source acquisition",
        timeout_seconds: int = 120,
    ):
        self.artifact_store = artifact_store
        self.opener = opener
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def acquire_pdf(self, arxiv_id: str) -> ArxivAcquisition:
        aid = normalize_arxiv_id(arxiv_id)
        url = versioned_pdf_url(aid)
        req = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,*/*;q=0.8",
            },
        )
        with self.opener(req, timeout=self.timeout_seconds) as response:
            body = response.read()
            resolved_url = response.geturl()
            media_type = response.headers.get_content_type()

        if not body.startswith(b"%PDF-"):
            raise ValueError(
                f"arXiv acquisition for {aid} did not return PDF bytes "
                f"(media_type={media_type!r}, resolved_url={resolved_url!r})"
            )

        host = urlparse(resolved_url).hostname or ""
        if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError(f"unexpected arXiv redirect host: {host}")

        receipt = self.artifact_store.put_bytes(body)
        return ArxivAcquisition(
            arxiv_id=aid,
            versioned_pdf_url=url,
            resolved_url=resolved_url,
            media_type=media_type,
            receipt=receipt,
        )
