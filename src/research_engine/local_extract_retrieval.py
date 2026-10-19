from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .orchestration import JobType
from .paper_package_ab import PaperPackageABPlanner, quote_token_recall
from .semantic_pipeline import ResearchContextHydrator


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_EVIDENCE_FACETS = (
    (
        "method",
        "method framework approach model architecture training "
        "algorithm mechanism",
    ),
    (
        "dataset",
        "dataset testbed receiver transmitter device capture sample "
        "collection data",
    ),
    (
        "evaluation",
        "result performance accuracy auroc oscr rejection unknown known "
        "evaluation metric",
    ),
    (
        "robustness",
        "limitation future robustness generalization domain shift "
        "environment channel interference",
    ),
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


def _display(value: str) -> str:
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", str(value))
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(str(value))]
    if drop_stopwords:
        return [token for token in tokens if token not in _STOPWORDS]
    return tokens


def _looks_like_heading(value: str) -> bool:
    text = _display(value)
    if not 4 <= len(text) <= 120:
        return False
    if re.search(r"[.!?][\"')\]]?$", text):
        return False
    words = text.split()
    if not 1 <= len(words) <= 14:
        return False
    alpha = [word for word in words if any(ch.isalpha() for ch in word)]
    if not alpha:
        return False
    uppercase = sum(word.isupper() for word in alpha) / len(alpha)
    title_like = sum(word[:1].isupper() for word in alpha) / len(alpha)
    return uppercase >= 0.6 or title_like >= 0.8


def exact_source_regions(
    pages: list[dict[str, Any]],
    *,
    max_chars: int = 650,
) -> list[dict[str, Any]]:
    """Split authoritative page text into stable exact regions.

    raw_source_form is always a byte-for-string slice of the hydrated page text.
    Section hints are deterministic retrieval metadata only; they never replace
    authoritative source text.
    """
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")

    regions: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text") or "")
        section_hint = ""
        start: int | None = None
        end = 0
        region_section = ""

        for match in re.finditer(r".*(?:\n|$)", text):
            raw_line = match.group(0)
            if not raw_line:
                continue
            display_line = _display(raw_line)
            if _looks_like_heading(display_line):
                section_hint = display_line

            if start is None:
                start = match.start()
                region_section = section_hint
            end = match.end()
            current = text[start:end]
            terminal = bool(re.search(r'[.!?][\"\')\]]?$', current.rstrip()))
            if len(current) < max_chars and not terminal:
                continue

            raw = text[start:end]
            display = _display(raw)
            if len(display) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": display,
                        "section_hint": region_section,
                        "derivation": {
                            "processor": (
                                "local-extract-authoritative-pdf-region-v1"
                            ),
                            "mode": "NATIVE_TEXT",
                        },
                    }
                )
            start = None
            region_section = ""

        if start is not None and end > start:
            raw = text[start:end]
            display = _display(raw)
            if len(display) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": display,
                        "section_hint": region_section,
                        "derivation": {
                            "processor": (
                                "local-extract-authoritative-pdf-region-v1"
                            ),
                            "mode": "NATIVE_TEXT",
                        },
                    }
                )
    return regions


def retrieval_query(
    *,
    topic: str | None,
    item: dict[str, Any],
) -> str:
    """Build the first-wave deterministic evidence-seeking query."""
    values = [
        str(topic or "").strip(),
        str(item.get("title") or "").strip(),
        str(item.get("rationale") or "").strip(),
    ]
    return " ".join(value for value in values if value)


def _bm25_rank_regions(
    query: str,
    regions: list[dict[str, Any]],
    *,
    k1: float,
    b: float,
    section_weight: float,
) -> list[dict[str, Any]]:
    query_tokens = list(dict.fromkeys(_tokens(query)))
    docs = [_tokens(str(region.get("display_text") or "")) for region in regions]
    lengths = [len(doc) for doc in docs]
    avgdl = sum(lengths) / len(lengths) if lengths else 1.0

    document_frequency: Counter[str] = Counter()
    for doc in docs:
        document_frequency.update(set(doc))

    n_docs = len(docs)
    idf = {
        token: math.log(
            1.0 + (n_docs - document_frequency[token] + 0.5)
            / (document_frequency[token] + 0.5)
        )
        for token in query_tokens
    }

    ranked: list[dict[str, Any]] = []
    for region, doc, doc_len in zip(regions, docs, lengths, strict=True):
        counts = Counter(doc)
        score = 0.0
        for token in query_tokens:
            tf = counts[token]
            if tf == 0:
                continue
            norm = k1 * (1.0 - b + b * doc_len / max(avgdl, 1.0))
            score += idf[token] * ((tf * (k1 + 1.0)) / (tf + norm))

        section_tokens = set(_tokens(str(region.get("section_hint") or "")))
        section_score = sum(
            idf[token] for token in query_tokens if token in section_tokens
        )
        score += section_weight * section_score

        row = dict(region)
        row["retrieval_score"] = score
        ranked.append(row)

    ranked.sort(
        key=lambda row: (
            -float(row["retrieval_score"]),
            int(row["page_number"]),
            int(row["start_char"]),
            str(row["region_id"]),
        )
    )
    return ranked


def rank_regions(
    query: str,
    regions: list[dict[str, Any]],
    *,
    top_k: int = 32,
    k1: float = 1.2,
    b: float = 0.75,
    section_weight: float = 0.35,
    evidence_facets: tuple[tuple[str, str], ...] = _EVIDENCE_FACETS,
) -> list[dict[str, Any]]:
    """Rank exact regions with deterministic facet-diversified lexical retrieval.

    The first list is the global scientific query. Additional fixed evidence
    facets broaden recall across method, dataset, evaluation, and robustness
    evidence without looking at teacher labels or model output. Lists are
    interleaved by rank depth with exact-region deduplication.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not regions:
        return []

    rankings: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "global",
            _bm25_rank_regions(
                query,
                regions,
                k1=k1,
                b=b,
                section_weight=section_weight,
            ),
        )
    ]
    rankings.extend(
        (
            f"facet:{name}",
            _bm25_rank_regions(
                f"{query} {terms}",
                regions,
                k1=k1,
                b=b,
                section_weight=section_weight,
            ),
        )
        for name, terms in evidence_facets
    )

    limit = min(top_k, len(regions))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    depth = 0
    while len(selected) < limit:
        added = False
        for channel, ranking in rankings:
            if depth >= len(ranking):
                continue
            row = ranking[depth]
            region_id = str(row["region_id"])
            if region_id in seen:
                continue
            result = dict(row)
            result["retrieval_channel"] = channel
            selected.append(result)
            seen.add(region_id)
            added = True
            if len(selected) >= limit:
                break
        if not added and depth >= max(
            len(ranking) for _, ranking in rankings
        ):
            break
        depth += 1

    for index, row in enumerate(selected, start=1):
        row["rank"] = index
    return selected


def evaluate_teacher_retrieval(
    *,
    teacher_evidence: list[dict[str, Any]],
    ranked_regions: list[dict[str, Any]],
    quote_threshold: float = 0.8,
) -> dict[str, Any]:
    if not 0 <= quote_threshold <= 1:
        raise ValueError("quote_threshold must be in [0,1]")

    eligible = [
        dict(item)
        for item in teacher_evidence
        if item.get("raw_source_form")
        and item.get("page_number") is not None
    ]
    region_pages = {int(region["page_number"]) for region in ranked_regions}
    rows: list[dict[str, Any]] = []
    quote_hits = 0
    page_hits = 0

    for item in eligible:
        page = int(item["page_number"])
        quote = str(item["raw_source_form"])
        same_page = [
            region for region in ranked_regions
            if int(region["page_number"]) == page
        ]
        page_hit = page in region_pages
        best_recall = max(
            (
                quote_token_recall(
                    quote,
                    str(region.get("raw_source_form") or ""),
                )
                for region in same_page
            ),
            default=0.0,
        )
        quote_hit = page_hit and best_recall >= quote_threshold
        page_hits += int(page_hit)
        quote_hits += int(quote_hit)
        rows.append(
            {
                "teacher_id": str(item.get("id") or ""),
                "teacher_type": str(item.get("type") or ""),
                "page_number": page,
                "page_hit": page_hit,
                "best_quote_token_recall": best_recall,
                "quote_region_hit": quote_hit,
            }
        )

    count = len(eligible)
    return {
        "teacher_evidence_count": count,
        "candidate_page_recall": page_hits / count if count else 1.0,
        "teacher_quote_region_recall": quote_hits / count if count else 1.0,
        "quote_threshold": quote_threshold,
        "teacher_rows": rows,
    }


def retrieval_frontier(
    *,
    teacher_evidence: list[dict[str, Any]],
    ranked_regions: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = (5, 8, 12, 16, 24, 32),
    quote_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Measure the recall/source-size frontier for one ranked region list."""
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError("retrieval cutoffs must be positive")
        selected = ranked_regions[:cutoff]
        metrics = evaluate_teacher_retrieval(
            teacher_evidence=teacher_evidence,
            ranked_regions=selected,
            quote_threshold=quote_threshold,
        )
        rows.append(
            {
                "cutoff": cutoff,
                "selected_regions": len(selected),
                "selected_source_chars": sum(
                    len(str(region.get("raw_source_form") or ""))
                    for region in selected
                ),
                "candidate_page_recall": metrics[
                    "candidate_page_recall"
                ],
                "teacher_quote_region_recall": metrics[
                    "teacher_quote_region_recall"
                ],
            }
        )
    return rows


def _historical_evidence(job: Any) -> list[dict[str, Any]]:
    result = dict(job.result or {})
    return [dict(item) for item in result.get("evidence", [])]


class PreservedExtractRetrievalBenchmark:
    """Read-only Phase-1 benchmark over preserved completed EXTRACT jobs."""

    def __init__(
        self,
        *,
        store: Any,
        artifact_root: str | Path,
        planner: Any | None = None,
        hydrator: Any | None = None,
        topic_provider: Callable[[str], str] | None = None,
    ):
        self.store = store
        self.planner = planner or PaperPackageABPlanner(
            store=store,
            artifact_root=artifact_root,
        )
        self.hydrator = hydrator or ResearchContextHydrator(
            store=store,
            artifact_root=artifact_root,
        )
        self.topic_provider = topic_provider or self._run_topic

    def _run_topic(self, run_id: str) -> str:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT topic FROM research_runs WHERE id=%s",
                    (run_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise ValueError(f"research run not found: {run_id}")
        return str(row[0] or "")

    def run(
        self,
        *,
        run_id: str | None = None,
        expected_papers: int = 9,
        top_k: int = 32,
        quote_threshold: float = 0.8,
        max_region_chars: int = 650,
    ) -> dict[str, Any]:
        run_id = run_id or self.planner.select_run(
            expected_papers=expected_papers
        )
        jobs = self.planner.completed_jobs(run_id, JobType.EXTRACT)
        if len(jobs) != expected_papers:
            raise ValueError(
                f"run {run_id} has {len(jobs)} completed EXTRACT jobs; "
                f"expected {expected_papers}"
            )

        topic = self.topic_provider(run_id)
        cases: list[dict[str, Any]] = []
        for job in jobs:
            hydrated = self.hydrator.hydrate(job)
            context = dict(
                hydrated.payload.get("_research_context") or {}
            )
            pages = list(context.get("pages") or [])
            source = dict(context.get("source_artifact") or {})
            item = dict(job.payload.get("item") or {})
            regions = exact_source_regions(
                pages,
                max_chars=max_region_chars,
            )
            query = retrieval_query(topic=topic, item=item)
            ranked = rank_regions(
                query,
                regions,
                top_k=top_k,
            )
            teacher = _historical_evidence(job)
            metrics = evaluate_teacher_retrieval(
                teacher_evidence=teacher,
                ranked_regions=ranked,
                quote_threshold=quote_threshold,
            )
            frontier = retrieval_frontier(
                teacher_evidence=teacher,
                ranked_regions=ranked,
                quote_threshold=quote_threshold,
            )
            cases.append(
                {
                    "job_id": job.id,
                    "item_id": str(item.get("id") or ""),
                    "source_artifact_id": str(source.get("id") or ""),
                    "source_sha256": str(source.get("sha256") or ""),
                    "query": query,
                    "page_count": len(pages),
                    "candidate_region_count": len(regions),
                    "ranked_region_count": len(ranked),
                    "teacher_metrics": metrics,
                    "retrieval_frontier": frontier,
                    "top_regions": [
                        {
                            "region_id": row["region_id"],
                            "rank": row["rank"],
                            "page_number": row["page_number"],
                            "retrieval_score": row["retrieval_score"],
                            "section_hint": row["section_hint"],
                        }
                        for row in ranked
                    ],
                }
            )

        teacher_count = sum(
            int(case["teacher_metrics"]["teacher_evidence_count"])
            for case in cases
        )
        page_hits = sum(
            sum(
                int(row["page_hit"])
                for row in case["teacher_metrics"]["teacher_rows"]
            )
            for case in cases
        )
        quote_hits = sum(
            sum(
                int(row["quote_region_hit"])
                for row in case["teacher_metrics"]["teacher_rows"]
            )
            for case in cases
        )
        candidate_regions = sum(
            int(case["candidate_region_count"])
            for case in cases
        )

        return {
            "schema_version": 1,
            "benchmark": "LOCAL_EXTRACT_DETERMINISTIC_RETRIEVAL_V1",
            "source_run_id": run_id,
            "expected_papers": expected_papers,
            "paper_count": len(cases),
            "top_k": top_k,
            "retrieval_strategy": "FACET_INTERLEAVED_BM25_V1",
            "evidence_facets": [
                {"name": name, "query_extension": terms}
                for name, terms in _EVIDENCE_FACETS
            ],
            "quote_threshold": quote_threshold,
            "max_region_chars": max_region_chars,
            "teacher_evidence_count": teacher_count,
            "candidate_page_recall_at_k": (
                page_hits / teacher_count if teacher_count else 1.0
            ),
            "teacher_quote_region_recall_at_k": (
                quote_hits / teacher_count if teacher_count else 1.0
            ),
            "candidate_region_count_total": candidate_regions,
            "candidate_region_count_per_paper": (
                candidate_regions / len(cases) if cases else 0.0
            ),
            "semantic_model_calls": 0,
            "scientific_state_materialized": False,
            "production_routing_changed": False,
            "cases": cases,
        }
