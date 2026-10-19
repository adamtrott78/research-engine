from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}
EVIDENCE_FACETS = (
    (
        "method",
        "method framework approach model architecture training algorithm "
        "mechanism",
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
ALLOWED_TYPES = {
    "method",
    "dataset",
    "result",
    "limitation",
    "research_gap",
    "other",
}
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


class Wave1Error(RuntimeError):
    pass


def display_text(value: str) -> str:
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", str(value))
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: str, *, drop_stopwords: bool = True) -> list[str]:
    rows = [match.group(0).lower() for match in TOKEN_RE.finditer(str(value))]
    if drop_stopwords:
        return [token for token in rows if token not in STOPWORDS]
    return rows


def looks_like_heading(value: str) -> bool:
    text = display_text(value)
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
            line = display_text(raw_line)
            if looks_like_heading(line):
                section_hint = line

            if start is None:
                start = match.start()
                region_section = section_hint
            end = match.end()
            current = text[start:end]
            terminal = bool(re.search(r'[.!?][\"\')\]]?$', current.rstrip()))
            if len(current) < max_chars and not terminal:
                continue

            raw = text[start:end]
            rendered = display_text(raw)
            if len(rendered) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": rendered,
                        "section_hint": region_section,
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                    }
                )
            start = None
            region_section = ""

        if start is not None and end > start:
            raw = text[start:end]
            rendered = display_text(raw)
            if len(rendered) >= 12:
                regions.append(
                    {
                        "region_id": (
                            f"pdf:p{page_number}:"
                            f"{start:06d}-{end:06d}"
                        ),
                        "page_number": page_number,
                        "start_char": start,
                        "end_char": end,
                        "raw_source_form": raw,
                        "display_text": rendered,
                        "section_hint": region_section,
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                    }
                )
    return regions


def _bm25(
    query: str,
    regions: list[dict[str, Any]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    section_weight: float = 0.35,
) -> list[dict[str, Any]]:
    query_tokens = list(dict.fromkeys(tokens(query)))
    docs = [tokens(str(region.get("display_text") or "")) for region in regions]
    lengths = [len(doc) for doc in docs]
    avgdl = sum(lengths) / len(lengths) if lengths else 1.0
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    n_docs = len(docs)
    idf = {
        token: math.log(
            1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5)
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
        section = set(tokens(str(region.get("section_hint") or "")))
        score += section_weight * sum(
            idf[token] for token in query_tokens if token in section
        )
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
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not regions:
        return []

    channels: list[tuple[str, list[dict[str, Any]]]] = [
        ("global", _bm25(query, regions))
    ]
    channels.extend(
        (
            f"facet:{name}",
            _bm25(f"{query} {extension}", regions),
        )
        for name, extension in EVIDENCE_FACETS
    )

    limit = min(top_k, len(regions))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    depth = 0
    while len(selected) < limit:
        added = False
        for channel, ranking in channels:
            if depth >= len(ranking):
                continue
            row = ranking[depth]
            region_id = str(row["region_id"])
            if region_id in seen:
                continue
            result = dict(row)
            result["retrieval_channel"] = channel
            result["rank"] = len(selected) + 1
            selected.append(result)
            seen.add(region_id)
            added = True
            if len(selected) >= limit:
                break
        if not added and depth >= max(len(ranking) for _, ranking in channels):
            break
        depth += 1
    return selected


def source_urls(arxiv_id: str) -> list[str]:
    values = [
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    ]
    for version in range(1, 11):
        values.append(f"https://arxiv.org/pdf/{arxiv_id}v{version}")
        values.append(f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf")
    return values


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "research-engine-local-extract-wave1/1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_DOWNLOAD_BYTES:
            raise Wave1Error(f"source too large: {url}")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise Wave1Error(f"source exceeded byte ceiling: {url}")
    return payload


def download_exact_pdf(
    paper: dict[str, Any],
    destination: pathlib.Path,
) -> dict[str, Any]:
    expected = str(paper["sha256"])
    errors: list[str] = []
    for url in source_urls(str(paper["arxiv_id"])):
        try:
            payload = _download(url)
        except (OSError, urllib.error.URLError, Wave1Error) as exc:
            errors.append(f"{url}:{type(exc).__name__}")
            continue
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            errors.append(f"{url}:sha256={actual}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {
            "url": url,
            "sha256": actual,
            "bytes": len(payload),
        }
    raise Wave1Error(
        "no arXiv candidate matched historical SHA-256 "
        f"for {paper['item_id']}; attempts={errors}"
    )


def extract_pdf_pages(path: pathlib.Path) -> tuple[list[dict[str, Any]], str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = ""
    try:
        title = str((reader.metadata or {}).title or "")
    except Exception:
        title = ""

    pages = [
        {
            "page_number": index + 1,
            "text": page.extract_text() or "",
        }
        for index, page in enumerate(reader.pages)
    ]
    return pages, display_text(title)


def evidence_page_recall(
    historical_pages: list[int],
    regions: list[dict[str, Any]],
) -> float:
    expected = {int(page) for page in historical_pages}
    if not expected:
        return 1.0
    actual = {int(region["page_number"]) for region in regions}
    return len(expected & actual) / len(expected)


def retrieval_frontier(
    historical_pages: list[int],
    ranked: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = (5, 8, 12, 16, 24, 32),
) -> list[dict[str, Any]]:
    rows = []
    for cutoff in cutoffs:
        selected = ranked[:cutoff]
        rows.append(
            {
                "cutoff": cutoff,
                "selected_regions": len(selected),
                "selected_source_chars": sum(
                    len(str(row["raw_source_form"])) for row in selected
                ),
                "historical_evidence_page_recall": evidence_page_recall(
                    historical_pages,
                    selected,
                ),
            }
        )
    return rows


def build_prompt(
    *,
    topic: str,
    paper: dict[str, Any],
    title: str,
    regions: list[dict[str, Any]],
    max_claims: int,
) -> str:
    candidates = [
        {
            "candidate_id": row["region_id"],
            "page_number": row["page_number"],
            "section_hint": row.get("section_hint") or "",
            "source_text": row["display_text"],
        }
        for row in regions
    ]
    return (
        "You are the atomic-claim normalization stage of a scientific "
        "evidence pipeline. You are NOT discovering source quotes. "
        "Every claim must reference exactly one supplied candidate_id. "
        "Return JSON only with top-level key claims. Each claim must have "
        "exactly candidate_id, type, normalized_form, confidence. "
        "type must be one of method,dataset,result,limitation,research_gap,other. "
        "normalized_form must be one atomic scientifically useful sentence "
        "fully supported by that candidate alone. Do not add numbers, causes, "
        "comparisons, entities, or conclusions absent from the candidate. "
        "Prefer omission to speculation. Emit at most two distinct claims for "
        f"one candidate and at most {max_claims} total claims. "
        "Do not emit raw_source_form; deterministic code owns quotes.\n\n"
        f"RESEARCH_TOPIC:\n{topic}\n\n"
        f"PAPER_ID: {paper['item_id']}\n"
        f"PAPER_TITLE: {title}\n\n"
        "CANDIDATES_JSON:\n"
        + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    )


def parse_json_object(text: str) -> dict[str, Any]:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise Wave1Error("model output did not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise Wave1Error("model output root must be an object")
    return parsed


def guard_claims(
    payload: dict[str, Any],
    regions: list[dict[str, Any]],
    *,
    max_claims: int,
    max_claims_per_region: int,
) -> dict[str, Any]:
    if set(payload) != {"claims"} or not isinstance(payload["claims"], list):
        return {
            "status": "FAIL_CLOSED",
            "reasons": ["INVALID_TOP_LEVEL_SCHEMA"],
            "evidence": [],
        }
    claims = payload["claims"]
    reasons: list[str] = []
    if not claims:
        reasons.append("NO_CLAIMS")
    if len(claims) > max_claims:
        reasons.append(f"TOO_MANY_CLAIMS:{len(claims)}")

    by_id = {str(row["region_id"]): row for row in regions}
    per_region: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    evidence: list[dict[str, Any]] = []

    for index, claim in enumerate(claims[: max_claims + 1]):
        if not isinstance(claim, dict):
            reasons.append(f"CLAIM_{index}:NOT_OBJECT")
            continue
        if set(claim) != {
            "candidate_id",
            "type",
            "normalized_form",
            "confidence",
        }:
            reasons.append(f"CLAIM_{index}:SCHEMA")
            continue

        candidate_id = str(claim["candidate_id"])
        candidate = by_id.get(candidate_id)
        if candidate is None:
            reasons.append(f"CLAIM_{index}:UNKNOWN_CANDIDATE")
            continue

        kind = str(claim["type"]).strip()
        normalized = display_text(str(claim["normalized_form"]))
        confidence = claim["confidence"]
        if kind not in ALLOWED_TYPES:
            reasons.append(f"CLAIM_{index}:INVALID_TYPE")
            continue
        if not 8 <= len(normalized) <= 420:
            reasons.append(f"CLAIM_{index}:INVALID_NORMALIZED_LENGTH")
            continue
        if not isinstance(confidence, (int, float)):
            reasons.append(f"CLAIM_{index}:INVALID_CONFIDENCE")
            continue
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            reasons.append(f"CLAIM_{index}:INVALID_CONFIDENCE")
            continue

        per_region[candidate_id] += 1
        if per_region[candidate_id] > max_claims_per_region:
            reasons.append(f"CLAIM_{index}:REGION_CLAIM_LIMIT")
            continue

        key = (candidate_id, kind, normalized.lower())
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "candidate_id": candidate_id,
                "type": kind,
                "normalized_form": normalized,
                "raw_source_form": str(candidate["raw_source_form"]),
                "page_number": int(candidate["page_number"]),
                "confidence": confidence,
                "source_layer": "AUTHORITATIVE_PDF_TEXT",
                "source_start_char": int(candidate["start_char"]),
                "source_end_char": int(candidate["end_char"]),
            }
        )

    return {
        "status": "PASS_GROUNDED" if not reasons and evidence else "FAIL_CLOSED",
        "reasons": reasons,
        "evidence": evidence if not reasons else [],
        "input_claim_count": len(claims),
        "accepted_claim_count": len(evidence) if not reasons else 0,
    }


def token_f1(left: str, right: str) -> float:
    a = tokens(left, drop_stopwords=False)
    b = tokens(right, drop_stopwords=False)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def teacher_agreement(
    teacher: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for ti, expected in enumerate(teacher):
        for qi, actual in enumerate(evidence):
            candidates.append(
                (
                    token_f1(
                        str(expected["normalized_form"]),
                        str(actual["normalized_form"]),
                    ),
                    ti,
                    qi,
                )
            )
    candidates.sort(reverse=True)
    used_t: set[int] = set()
    used_q: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, ti, qi in candidates:
        if ti in used_t or qi in used_q:
            continue
        used_t.add(ti)
        used_q.add(qi)
        expected = teacher[ti]
        actual = evidence[qi]
        matches.append(
            {
                "teacher_index": ti,
                "qwen_index": qi,
                "normalized_f1": score,
                "passing": score >= threshold,
                "page_equal": (
                    int(expected["page_number"]) == int(actual["page_number"])
                ),
                "type_equal": (
                    str(expected["type"]) == str(actual["type"])
                ),
            }
        )

    passing = [row for row in matches if row["passing"]]
    return {
        "teacher_claim_count": len(teacher),
        "qwen_claim_count": len(evidence),
        "matched_pairs": len(matches),
        "passing_pairs": len(passing),
        "matched_normalized_f1": (
            sum(row["normalized_f1"] for row in matches) / len(matches)
            if matches else 0.0
        ),
        "teacher_claim_coverage_at_f1_0_5": (
            len(passing) / len(teacher) if teacher else 1.0
        ),
        "qwen_precision_at_f1_0_5": (
            len(passing) / len(evidence) if evidence else 0.0
        ),
        "page_agreement_on_passing_matches": (
            sum(int(row["page_equal"]) for row in passing) / len(passing)
            if passing else 0.0
        ),
        "type_agreement_on_passing_matches": (
            sum(int(row["type_equal"]) for row in passing) / len(passing)
            if passing else 0.0
        ),
        "matches": matches,
    }


class GPUMemorySampler:
    def __init__(self, expected_uuids: list[str]):
        self.expected_uuids = list(expected_uuids)
        self.peak = {uuid: 0 for uuid in expected_uuids}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                for line in raw.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 2 or parts[0] not in self.peak:
                        continue
                    self.peak[parts[0]] = max(
                        self.peak[parts[0]],
                        int(parts[1]),
                    )
            except Exception:
                pass
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=3)
        return False


def initialize_llm(
    model_path: str,
    *,
    tensor_parallel_size: int,
):
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        trust_remote_code=False,
        disable_log_stats=True,
        mm_processor_cache_gb=0.25,
    )


def model_claims(
    llm: Any,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    started = time.perf_counter()
    outputs = llm.chat(
        [{"role": "user", "content": prompt}],
        sampling_params=params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    seconds = time.perf_counter() - started
    output = outputs[0]
    generated = output.outputs[0]
    parsed = parse_json_object(generated.text)
    return parsed, {
        "wall_time_seconds": seconds,
        "prompt_tokens": len(getattr(output, "prompt_token_ids", []) or []),
        "output_tokens": len(generated.token_ids),
        "generated_text_sha256": hashlib.sha256(
            generated.text.encode("utf-8")
        ).hexdigest(),
    }


def decide_wave1(report: dict[str, Any]) -> str:
    hard = report["hard_safety"]
    if not all(bool(value) for value in hard.values()):
        return "REJECT"
    metrics = report["metrics"]
    completion = float(metrics["job_local_completion_fraction"])
    page_recall = float(metrics["historical_evidence_page_recall_at_12"])
    if completion < 0.5 or page_recall < 0.5:
        return "REJECT"

    control = report.get("teacher_subset_agreement") or {}
    research_positive = (
        float(control.get("qwen_precision_at_f1_0_5", 0.0)) >= 0.95
        and float(control.get("teacher_claim_coverage_at_f1_0_5", 0.0)) >= 0.80
        and float(
            control.get("page_agreement_on_passing_matches", 0.0)
        ) >= 0.95
        and float(
            control.get("type_agreement_on_passing_matches", 0.0)
        ) >= 0.95
        and completion >= 0.50
        and page_recall >= 0.80
        and float(
            metrics["projected_remote_input_token_reduction"]
        ) >= 0.50
    )
    return "PROMISING" if research_positive else "PARTIAL"


def run_wave1(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
    model_path: str,
    expected_gpu_uuids: list[str],
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_region_chars = int(config["max_region_chars"])
    max_claims = int(config["max_claims_per_paper"])
    max_claims_per_region = int(config["max_claims_per_region"])

    papers: list[dict[str, Any]] = []
    total_expected_pages = 0
    total_hit_pages = 0
    total_selected_chars = 0
    total_candidate_regions = 0
    acquisition_bytes = 0

    for paper in fixture["papers"]:
        paper = dict(paper)
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        acquisition_bytes += int(acquisition["bytes"])
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise Wave1Error(
                f"page-count mismatch for {paper['item_id']}: "
                f"expected {paper['total_pdf_pages']} got {len(pages)}"
            )
        regions = exact_source_regions(pages, max_chars=max_region_chars)
        query = (
            str(fixture["topic"])
            + " "
            + title
            + " "
            + str(paper["item_id"])
        ).strip()
        ranked = rank_regions(query, regions, top_k=32)
        selected = ranked[:top_k]
        expected_pages = set(int(x) for x in paper["historical_evidence_pages"])
        hit_pages = expected_pages & {
            int(row["page_number"]) for row in selected
        }
        total_expected_pages += len(expected_pages)
        total_hit_pages += len(hit_pages)
        selected_chars = sum(
            len(str(row["raw_source_form"])) for row in selected
        )
        total_selected_chars += selected_chars
        total_candidate_regions += len(regions)
        papers.append(
            {
                **paper,
                "title": title,
                "acquisition": acquisition,
                "page_count": len(pages),
                "candidate_region_count": len(regions),
                "query": query,
                "retrieval_frontier": retrieval_frontier(
                    paper["historical_evidence_pages"],
                    ranked,
                ),
                "historical_evidence_page_recall_at_12": (
                    len(hit_pages) / len(expected_pages)
                    if expected_pages else 1.0
                ),
                "selected_source_chars": selected_chars,
                "selected_regions": selected,
                "prompt": build_prompt(
                    topic=str(fixture["topic"]),
                    paper=paper,
                    title=title,
                    regions=selected,
                    max_claims=max_claims,
                ),
            }
        )

    init_started = time.perf_counter()
    with GPUMemorySampler(expected_gpu_uuids) as gpu:
        llm = initialize_llm(
            model_path,
            tensor_parallel_size=int(config["tensor_parallel_size"]),
        )
        model_init_seconds = time.perf_counter() - init_started

        completed = 0
        prompt_tokens = 0
        output_tokens = 0
        inference_seconds = 0.0
        total_claims = 0
        for paper in papers:
            try:
                payload, usage = model_claims(
                    llm,
                    str(paper.pop("prompt")),
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    seed=int(config["seed"]),
                    max_tokens=int(config["max_output_tokens"]),
                )
                guard = guard_claims(
                    payload,
                    paper["selected_regions"],
                    max_claims=max_claims,
                    max_claims_per_region=max_claims_per_region,
                )
                paper["qwen_usage"] = usage
                paper["guard"] = guard
                prompt_tokens += int(usage["prompt_tokens"])
                output_tokens += int(usage["output_tokens"])
                inference_seconds += float(usage["wall_time_seconds"])
                total_claims += int(guard["accepted_claim_count"])
                if guard["status"] == "PASS_GROUNDED":
                    completed += 1
            except Exception as exc:
                paper["guard"] = {
                    "status": "FAIL_CLOSED",
                    "reasons": [f"{type(exc).__name__}:{exc}"],
                    "evidence": [],
                    "accepted_claim_count": 0,
                }
                paper["qwen_usage"] = {
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "wall_time_seconds": 0.0,
                }

    teacher_paper = next(
        paper for paper in papers if paper.get("teacher_claims")
    )
    teacher_control = teacher_agreement(
        list(teacher_paper["teacher_claims"]),
        list(teacher_paper["guard"]["evidence"]),
    )

    historical = dict(fixture["historical"])
    metrics = {
        "paper_count": len(papers),
        "historical_evidence_count": int(
            historical["evidence_object_count"]
        ),
        "historical_evidence_bearing_page_count": total_expected_pages,
        "historical_evidence_page_hits_at_12": total_hit_pages,
        "historical_evidence_page_recall_at_12": (
            total_hit_pages / total_expected_pages
            if total_expected_pages else 1.0
        ),
        "candidate_region_count_total": total_candidate_regions,
        "candidate_region_count_per_paper": (
            total_candidate_regions / len(papers) if papers else 0.0
        ),
        "selected_source_chars_total": total_selected_chars,
        "historical_extraction_context_chars": int(
            historical["extraction_context_chars_total"]
        ),
        "selected_source_char_reduction_fraction": (
            1.0
            - total_selected_chars
            / int(historical["extraction_context_chars_total"])
        ),
        "qwen_prompt_tokens": prompt_tokens,
        "qwen_output_tokens": output_tokens,
        "historical_remote_extract_input_tokens": int(
            historical["extract_input_tokens"]
        ),
        "projected_remote_input_token_reduction": (
            1.0
            - prompt_tokens / int(historical["extract_input_tokens"])
        ),
        "job_local_completion_fraction": (
            completed / len(papers) if papers else 0.0
        ),
        "locally_completed_jobs": completed,
        "accepted_claim_count": total_claims,
        "model_init_seconds": model_init_seconds,
        "inference_seconds_total": inference_seconds,
        "downloaded_source_bytes": acquisition_bytes,
        "peak_gpu_memory_mib_by_uuid": dict(gpu.peak),
    }
    hard_safety = {
        "fabricated_or_unlocatable_quote_count_zero": all(
            all(
                evidence["candidate_id"]
                in {
                    region["region_id"]
                    for region in paper["selected_regions"]
                }
                for evidence in paper["guard"]["evidence"]
            )
            for paper in papers
        ),
        "exact_quote_locatability_one": all(
            evidence["raw_source_form"]
            == next(
                region["raw_source_form"]
                for region in paper["selected_regions"]
                if region["region_id"] == evidence["candidate_id"]
            )
            for paper in papers
            for evidence in paper["guard"]["evidence"]
        ),
        "page_provenance_coverage_one": all(
            isinstance(evidence["page_number"], int)
            for paper in papers
            for evidence in paper["guard"]["evidence"]
        ),
        "scientific_state_materialized_false": True,
        "production_routing_changed_false": True,
    }

    report = {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_WAVE1_QWEN_NORMALIZATION",
        "source_run_id": fixture["source_run_id"],
        "model_role": "ATOMIC_CLAIM_NORMALIZATION_ONLY",
        "model_path": model_path,
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "retrieval_strategy": "FACET_INTERLEAVED_BM25_V1",
        "qwen_config": config,
        "historical": historical,
        "metrics": metrics,
        "teacher_subset_agreement": teacher_control,
        "teacher_control_scope": {
            "full_historical_teacher_claim_count": int(
                historical["evidence_object_count"]
            ),
            "normalized_teacher_claims_available": int(
                teacher_control["teacher_claim_count"]
            ),
            "normalized_teacher_control_fraction": (
                teacher_control["teacher_claim_count"]
                / int(historical["evidence_object_count"])
            ),
            "limitation": (
                "Full normalized Codex teacher text is not staged into the "
                "isolated benchmark runner. Page/count controls cover all nine "
                "papers; semantic F1/type/page control uses the preserved "
                "teacher subset only."
            ),
        },
        "hard_safety": hard_safety,
        "papers": papers,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "public_model_acquisition": False,
        "source_acquisition": "ARXIV_EXACT_SHA_EPHEMERAL",
    }
    report["wave1_decision"] = decide_wave1(report)
    return report
