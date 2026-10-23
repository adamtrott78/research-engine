from __future__ import annotations

import gc
import math
import pathlib
import shutil
import time
from collections import Counter
from typing import Any

from research_engine.local_extract_diagnostics import (
    _best_alignment,
    full_pdf_propositions,
)
from research_engine.local_extract_page_conditioned import (
    page_conditioned_proposition_pool,
)
from research_engine.local_extract_reranker_bakeoff import (
    BGE_ARM_KEYS,
    QWEN_ARM_ID,
    BGEM3Scorer,
    Qwen3RerankerScorer,
    snapshot_download_exact,
)
from research_engine.local_extract_wave1 import (
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    rank_regions,
    tokens,
)


BM25_ARM_ID = "BM25_DIRECT"
QUERY_MODES = ("TARGET_TEXT_ORACLE", "INTENT_QUERY_ORACLE")
KNOWN_MISSES = ("e2.a", "e2.b", "e4.b")


def _normalize(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _teacher_paper(fixture: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in fixture["papers"] if row.get("teacher_claims")]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one teacher paper, got {len(rows)}")
    return dict(rows[0])


def _prepare_teacher_pool(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
) -> dict[str, Any]:
    paper = _teacher_paper(fixture)
    if str(paper["item_id"]) != "arxiv:2403.04036":
        raise RuntimeError("unexpected teacher paper")
    pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
    acquisition = download_exact_pdf(paper, pdf_path)
    pages, title = extract_pdf_pages(pdf_path)
    if len(pages) != int(paper["total_pdf_pages"]):
        raise RuntimeError("teacher page-count mismatch")
    broad_query = (
        str(fixture["topic"]) + " " + title + " " + str(paper["item_id"])
    ).strip()
    regions = exact_source_regions(pages, max_chars=650)
    selected_regions = rank_regions(broad_query, regions, top_k=32)[:12]
    full_props = full_pdf_propositions(pages)
    page_pool = page_conditioned_proposition_pool(
        full_props,
        selected_regions,
    )
    return {
        "item_id": str(paper["item_id"]),
        "title": title,
        "broad_query": broad_query,
        "acquisition": acquisition,
        "selected_page_numbers": sorted(
            {int(row["page_number"]) for row in selected_regions}
        ),
        "selected_region_count": len(selected_regions),
        "page_pool": page_pool,
    }


def _bm25_scores(
    query: str,
    propositions: list[dict[str, Any]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    section_weight: float = 0.35,
) -> list[float]:
    query_tokens = list(dict.fromkeys(tokens(query)))
    docs = [
        tokens(str(row.get("display_text") or row.get("raw_source_form") or ""))
        for row in propositions
    ]
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
    scores: list[float] = []
    for row, doc, doc_len in zip(
        propositions, docs, lengths, strict=True
    ):
        counts = Counter(doc)
        score = 0.0
        for token in query_tokens:
            tf = counts[token]
            if tf == 0:
                continue
            norm = k1 * (1.0 - b + b * doc_len / max(avgdl, 1.0))
            score += idf[token] * ((tf * (k1 + 1.0)) / (tf + norm))
        section = set(tokens(str(row.get("section_hint") or "")))
        score += section_weight * sum(
            idf[token] for token in query_tokens if token in section
        )
        scores.append(float(score))
    return scores


def _rank(
    propositions: list[dict[str, Any]],
    scores: list[float],
) -> list[dict[str, Any]]:
    if len(propositions) != len(scores):
        raise ValueError("score count mismatch")
    ranked = sorted(
        zip(propositions, scores, strict=True),
        key=lambda pair: (-float(pair[1]), str(pair[0]["proposition_id"])),
    )
    result = []
    for rank, (row, score) in enumerate(ranked, start=1):
        item = dict(row)
        item["oracle_rank"] = rank
        item["oracle_score"] = float(score)
        result.append(item)
    return result


def _support_rank(
    atom: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_page = int(atom["page_number"])
    anchor_rows = []
    ranks = []
    proposition_ids: set[str] = set()
    for anchor in atom["required_anchors"]:
        matches = []
        for row in ranked:
            if int(row["page_number"]) != expected_page:
                continue
            text = _normalize(
                str(row.get("raw_source_form") or row.get("display_text") or "")
            )
            if str(anchor) in text:
                matches.append(
                    {
                        "rank": int(row["oracle_rank"]),
                        "proposition_id": str(row["proposition_id"]),
                    }
                )
        first_rank = min((row["rank"] for row in matches), default=None)
        if first_rank is not None:
            ranks.append(int(first_rank))
            proposition_ids.update(
                row["proposition_id"]
                for row in matches
                if int(row["rank"]) <= int(first_rank)
            )
        anchor_rows.append(
            {
                "anchor": str(anchor),
                "first_rank": first_rank,
                "matches": matches[:5],
            }
        )
    supported = len(ranks) == len(atom["required_anchors"])
    support_rank = max(ranks) if supported and ranks else None
    return {
        "support_rank": support_rank,
        "supported_at_1": bool(support_rank is not None and support_rank <= 1),
        "supported_at_3": bool(support_rank is not None and support_rank <= 3),
        "supported_at_12": bool(support_rank is not None and support_rank <= 12),
        "reciprocal_rank": (
            1.0 / float(support_rank) if support_rank is not None else 0.0
        ),
        "supporting_proposition_ids": sorted(proposition_ids),
        "anchors": anchor_rows,
    }


def _evaluate_atom(
    atom: dict[str, Any],
    query: str,
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    support = _support_rank(atom, ranked)
    f1 = {}
    for cutoff in (1, 3, 12):
        best = _best_alignment(
            str(atom["normalized_form"]),
            ranked[:cutoff],
        )
        f1[str(cutoff)] = float(best["f1"])
    return {
        "id": str(atom["id"]),
        "parent_teacher_id": str(atom["parent_teacher_id"]),
        "type": str(atom["type"]),
        "expected_page": int(atom["page_number"]),
        "query": query,
        **support,
        "best_token_f1": f1,
        "top12": [
            {
                "rank": int(row["oracle_rank"]),
                "proposition_id": str(row["proposition_id"]),
                "page_number": int(row["page_number"]),
                "score": float(row["oracle_score"]),
                "raw_source_form": str(row["raw_source_form"]),
            }
            for row in ranked[:12]
        ],
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    broad_baseline: dict[str, Any],
) -> dict[str, Any]:
    count = len(rows)
    def n(key: str) -> int:
        return sum(1 for row in rows if bool(row[key]))
    supported_1 = n("supported_at_1")
    supported_3 = n("supported_at_3")
    supported_12 = n("supported_at_12")
    known = [
        atom_id
        for atom_id in KNOWN_MISSES
        if any(
            row["id"] == atom_id and row["supported_at_12"]
            for row in rows
        )
    ]
    return {
        "atom_count": count,
        "supported_at_1": supported_1,
        "support_recall_at_1": supported_1 / count if count else 0.0,
        "supported_at_3": supported_3,
        "support_recall_at_3": supported_3 / count if count else 0.0,
        "supported_at_12": supported_12,
        "support_recall_at_12": supported_12 / count if count else 0.0,
        "mrr": (
            sum(float(row["reciprocal_rank"]) for row in rows) / count
            if count else 0.0
        ),
        "mean_best_token_f1_at_1": (
            sum(float(row["best_token_f1"]["1"]) for row in rows) / count
            if count else 0.0
        ),
        "mean_best_token_f1_at_3": (
            sum(float(row["best_token_f1"]["3"]) for row in rows) / count
            if count else 0.0
        ),
        "mean_best_token_f1_at_12": (
            sum(float(row["best_token_f1"]["12"]) for row in rows) / count
            if count else 0.0
        ),
        "known_miss_recovered": known,
        "known_miss_recovery_count": len(known),
        "broad_query_supported_at_12": int(
            broad_baseline["supported_at_12"]
        ),
        "delta_supported_at_12_vs_broad": (
            supported_12 - int(broad_baseline["supported_at_12"])
        ),
        "atoms": rows,
    }


def _verdict(intent_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    confirmed = [
        arm_id
        for arm_id, row in intent_results.items()
        if int(row["supported_at_12"]) >= 10
        and int(row["known_miss_recovery_count"]) == 3
    ]
    if confirmed:
        return {
            "verdict": "QUERY_REPRESENTATION_CONFIRMED",
            "qualifying_arms": confirmed,
            "rationale": (
                "At least one frozen ranker supports >=10/13 atoms at top-12 "
                "and recovers all three known misses when given answer-free "
                "atom-specific evidence intents."
            ),
        }
    partial = [
        arm_id
        for arm_id, row in intent_results.items()
        if int(row["supported_at_12"]) >= 7
        or int(row["known_miss_recovery_count"]) == 3
    ]
    if partial:
        return {
            "verdict": "QUERY_REPRESENTATION_PARTIAL",
            "qualifying_arms": partial,
            "rationale": (
                "Precise answer-free intents materially improve retrieval for "
                "at least one arm, but no arm reaches the predeclared "
                "query-representation-confirmed gate."
            ),
        }
    return {
        "verdict": "RANKER_LIMIT_SUPPORTED",
        "qualifying_arms": [],
        "rationale": (
            "Even precise answer-free atom-specific intents fail to produce "
            "the predeclared retrieval gain; generic ranker/domain capability "
            "remains a substantial bottleneck."
        ),
    }


def run_oracle_query_diagnostic(
    *,
    fixture: dict[str, Any],
    contract: dict[str, Any],
    support_fixture: dict[str, Any],
    workdir: pathlib.Path,
) -> dict[str, Any]:
    teacher = _prepare_teacher_pool(fixture=fixture, workdir=workdir)
    propositions = list(teacher["page_pool"])
    atoms = [dict(row) for row in support_fixture["atoms"]]
    if len(atoms) != 13:
        raise RuntimeError("oracle diagnostic requires 13 unique atoms")

    queries = {
        "TARGET_TEXT_ORACLE": {
            str(atom["id"]): str(atom["normalized_form"])
            for atom in atoms
        },
        "INTENT_QUERY_ORACLE": {
            str(key): str(value)
            for key, value in contract["query_modes"][
                "INTENT_QUERY_ORACLE"
            ]["queries"].items()
        },
    }
    atom_ids = {str(atom["id"]) for atom in atoms}
    if set(queries["INTENT_QUERY_ORACLE"]) != atom_ids:
        raise RuntimeError("intent query set does not match frozen atom IDs")

    rankings: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        mode: {
            arm["id"]: {}
            for arm in contract["arms"]
        }
        for mode in QUERY_MODES
    }

    # Deterministic direct BM25 control.
    for mode in QUERY_MODES:
        for atom in atoms:
            atom_id = str(atom["id"])
            query = queries[mode][atom_id]
            rankings[mode][BM25_ARM_ID][atom_id] = _rank(
                propositions,
                _bm25_scores(query, propositions),
            )

    resources: dict[str, Any] = {
        BM25_ARM_ID: {
            "model_download_bytes": 0,
            "model_download_seconds": 0.0,
            "model_load_seconds": 0.0,
            "ranking_inference_seconds": 0.0,
            "peak_gpu_memory_mib": 0.0,
        }
    }

    bge_cfg = contract["model_inputs"]["BGE_M3"]
    bge_dir = workdir / "models" / "bge-m3"
    bge_download = snapshot_download_exact(
        repository=str(bge_cfg["repository"]),
        revision=str(bge_cfg["revision"]),
        destination=bge_dir,
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "colbert_linear.pt",
            "sparse_linear.pt",
        ],
    )
    bge = BGEM3Scorer(
        bge_dir,
        max_passage_length=int(bge_cfg["max_passage_length"]),
        hybrid_weights=[float(v) for v in bge_cfg["hybrid_weights"]],
        batch_size=int(bge_cfg["batch_size"]),
    )
    for mode in QUERY_MODES:
        for atom in atoms:
            atom_id = str(atom["id"])
            scores = bge.score(queries[mode][atom_id], propositions)
            for arm_id in BGE_ARM_KEYS:
                rankings[mode][arm_id][atom_id] = _rank(
                    propositions,
                    scores[arm_id],
                )
    bge_metrics = bge.resource_metrics()
    bge.close()
    for arm_id in BGE_ARM_KEYS:
        resources[arm_id] = {
            "model_download_bytes": int(bge_download["bytes"]),
            "model_download_seconds": float(bge_download["download_seconds"]),
            "model_snapshot_tree_sha256": str(bge_download["tree_sha256"]),
            **bge_metrics,
        }
    shutil.rmtree(bge_dir, ignore_errors=True)

    qwen_cfg = contract["model_inputs"]["QWEN3_RERANKER_0_6B"]
    qwen_dir = workdir / "models" / "qwen3-reranker-0.6b"
    qwen_download = snapshot_download_exact(
        repository=str(qwen_cfg["repository"]),
        revision=str(qwen_cfg["revision"]),
        destination=qwen_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ],
    )
    qwen = Qwen3RerankerScorer(
        qwen_dir,
        instruction=str(qwen_cfg["instruction"]),
        max_length=int(qwen_cfg["max_length"]),
        batch_size=int(qwen_cfg["batch_size"]),
    )
    for mode in QUERY_MODES:
        for atom in atoms:
            atom_id = str(atom["id"])
            scores = qwen.score(queries[mode][atom_id], propositions)
            rankings[mode][QWEN_ARM_ID][atom_id] = _rank(
                propositions,
                scores[QWEN_ARM_ID],
            )
    qwen_metrics = qwen.resource_metrics()
    qwen.close()
    resources[QWEN_ARM_ID] = {
        "model_download_bytes": int(qwen_download["bytes"]),
        "model_download_seconds": float(qwen_download["download_seconds"]),
        "model_snapshot_tree_sha256": str(qwen_download["tree_sha256"]),
        **qwen_metrics,
    }
    shutil.rmtree(qwen_dir, ignore_errors=True)

    results: dict[str, Any] = {}
    broad = contract["evaluation"]["broad_query_support_baselines"]
    for mode in QUERY_MODES:
        results[mode] = {}
        for arm in contract["arms"]:
            arm_id = str(arm["id"])
            atom_rows = []
            for atom in atoms:
                atom_id = str(atom["id"])
                atom_rows.append(
                    _evaluate_atom(
                        atom,
                        queries[mode][atom_id],
                        rankings[mode][arm_id][atom_id],
                    )
                )
            aggregate = _aggregate(
                atom_rows,
                broad_baseline=broad[arm_id],
            )
            aggregate["resources"] = resources[arm_id]
            results[mode][arm_id] = aggregate

    verdict = _verdict(results["INTENT_QUERY_ORACLE"])

    import huggingface_hub
    import torch
    import transformers

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_ORACLE_QUERY_DIAGNOSTIC",
        "benchmark_id": contract["benchmark_id"],
        "baseline_sha": contract["baseline_sha"],
        "teacher_item_id": teacher["item_id"],
        "selected_page_numbers": teacher["selected_page_numbers"],
        "page_pool_proposition_count": len(propositions),
        "query_modes": results,
        **verdict,
        "model_revisions": {
            key: {
                "repository": str(value["repository"]),
                "revision": str(value["revision"]),
            }
            for key, value in contract["model_inputs"].items()
        },
        "runtime_environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "external_model_downloads": 2,
        "qwen35_extract_model_calls": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
        "wave7_started": False,
    }
