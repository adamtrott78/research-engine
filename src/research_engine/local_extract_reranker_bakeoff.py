from __future__ import annotations

import gc
import hashlib
import pathlib
import re
import shutil
import time
from collections.abc import Callable
from typing import Any

from research_engine.local_extract_diagnostics import (
    _best_alignment,
    _reference_rows,
    full_pdf_propositions,
)
from research_engine.local_extract_page_conditioned import (
    page_conditioned_proposition_pool,
    rank_page_conditioned_propositions,
)
from research_engine.local_extract_wave1 import (
    EVIDENCE_FACETS,
    display_text,
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    rank_regions,
)
from research_engine.local_extract_wave4 import exact_propositions


BGE_ARM_KEYS = {
    "BGE_M3_DENSE": "dense",
    "BGE_M3_SPARSE": "sparse",
    "BGE_M3_COLBERT": "colbert",
    "BGE_M3_HYBRID_040_020_040": "colbert+sparse+dense",
}
QWEN_ARM_ID = "QWEN3_RERANKER_0_6B"
CONTROL_ARM_ID = "PAGE_CONDITIONED_BM25_CONTROL"
KNOWN_MISS_ATOMS = ("e2.a", "e2.b", "e4.b")


def channel_queries(query: str) -> list[tuple[str, str]]:
    rows = [("global", query)]
    rows.extend(
        (f"facet:{name}", f"{query} {extension}")
        for name, extension in EVIDENCE_FACETS
    )
    return rows


def interleave_scored_channels(
    propositions: list[dict[str, Any]],
    channel_scores: dict[str, list[float]],
    *,
    arm_id: str,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    proposition_by_id = {
        str(row["proposition_id"]): dict(row) for row in propositions
    }
    expected = len(propositions)
    rankings: dict[str, list[str]] = {}
    for channel, scores in channel_scores.items():
        if len(scores) != expected:
            raise ValueError(
                f"score count mismatch for {channel}: "
                f"expected {expected} got {len(scores)}"
            )
        ranked = sorted(
            zip(propositions, scores, strict=True),
            key=lambda item: (
                -float(item[1]),
                str(item[0]["proposition_id"]),
            ),
        )
        rankings[channel] = [
            str(row["proposition_id"]) for row, _ in ranked
        ]

    channel_order = [name for name, _ in channel_queries("")]
    missing = [name for name in channel_order if name not in rankings]
    if missing:
        raise ValueError(f"missing score channels: {missing}")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    depth = 0
    limit = min(top_k, len(propositions))
    while len(selected) < limit:
        added = False
        for channel in channel_order:
            ranking = rankings[channel]
            if depth >= len(ranking):
                continue
            proposition_id = ranking[depth]
            if proposition_id in seen:
                continue
            row = dict(proposition_by_id[proposition_id])
            row["retrieval_channel"] = channel
            row["retrieval_rank"] = len(selected) + 1
            row["selector_id"] = arm_id
            row["retrieval_score"] = float(
                channel_scores[channel][
                    next(
                        index
                        for index, proposition in enumerate(propositions)
                        if str(proposition["proposition_id"])
                        == proposition_id
                    )
                ]
            )
            selected.append(row)
            seen.add(proposition_id)
            added = True
            if len(selected) >= limit:
                break
        if not added and depth >= expected:
            break
        depth += 1
    return selected


def _normalize_support_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def evaluate_atom_support(
    support_fixture: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for atom in support_fixture["atoms"]:
        page_number = int(atom["page_number"])
        page_propositions = [
            row
            for row in selected
            if int(row["page_number"]) == page_number
        ]
        normalized = [
            (
                str(row["proposition_id"]),
                _normalize_support_text(
                    str(
                        row.get("raw_source_form")
                        or row.get("display_text")
                        or ""
                    )
                ),
            )
            for row in page_propositions
        ]
        anchor_hits: list[dict[str, Any]] = []
        supporting_ids: set[str] = set()
        for anchor in atom["required_anchors"]:
            matches = [
                proposition_id
                for proposition_id, text in normalized
                if str(anchor) in text
            ]
            anchor_hits.append(
                {
                    "anchor": str(anchor),
                    "matched_proposition_ids": matches,
                    "supported": bool(matches),
                }
            )
            supporting_ids.update(matches)
        supported = all(row["supported"] for row in anchor_hits)
        rows.append(
            {
                "id": str(atom["id"]),
                "parent_teacher_id": str(atom["parent_teacher_id"]),
                "page_number": page_number,
                "supported": supported,
                "supporting_proposition_ids": sorted(supporting_ids),
                "anchors": anchor_hits,
            }
        )

    supported_ids = {
        row["id"] for row in rows if bool(row["supported"])
    }
    known_miss_recovered = [
        atom_id for atom_id in KNOWN_MISS_ATOMS if atom_id in supported_ids
    ]
    return {
        "atom_count": len(rows),
        "supported_atom_count": len(supported_ids),
        "atom_support_recall_at_12": (
            len(supported_ids) / len(rows) if rows else 0.0
        ),
        "known_miss_atoms": list(KNOWN_MISS_ATOMS),
        "known_miss_recovered": known_miss_recovered,
        "known_miss_recovery_count": len(known_miss_recovered),
        "atoms": rows,
    }


def evaluate_lexical(
    references: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for reference in references:
        best = _best_alignment(
            str(reference["normalized_form"]),
            selected,
        )
        rows.append(
            {
                "id": str(reference["id"]),
                "page_number": int(reference["page_number"]),
                "type": str(reference["type"]),
                "normalized_form": str(reference["normalized_form"]),
                "best_selected": best,
            }
        )
    mean_f1 = (
        sum(float(row["best_selected"]["f1"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    passing = [
        row for row in rows if float(row["best_selected"]["f1"]) >= 0.5
    ]
    return {
        "reference_count": len(rows),
        "mean_f1": mean_f1,
        "coverage_at_f1_0_5": (
            len(passing) / len(rows) if rows else 0.0
        ),
        "passing_count": len(passing),
        "rows": rows,
    }


def _gap_closure(
    score: float,
    *,
    baseline: float,
    ceiling: float,
) -> float:
    gap = ceiling - baseline
    if gap <= 0:
        return 0.0
    return (score - baseline) / gap


def evaluate_arm(
    *,
    arm_id: str,
    selected: list[dict[str, Any]],
    grouped_references: list[dict[str, Any]],
    atom_references: list[dict[str, Any]],
    support_fixture: dict[str, Any],
    decision_contract: dict[str, Any],
) -> dict[str, Any]:
    grouped = evaluate_lexical(grouped_references, selected)
    atoms = evaluate_lexical(atom_references, selected)
    support = evaluate_atom_support(support_fixture, selected)
    decision = decision_contract["predeclared_decision"]

    grouped_closure = _gap_closure(
        float(grouped["mean_f1"]),
        baseline=float(decision["wave6_grouped_mean_f1"]),
        ceiling=float(decision["selected_page_pool_grouped_ceiling_f1"]),
    )
    atom_closure = _gap_closure(
        float(atoms["mean_f1"]),
        baseline=float(decision["wave6_atom_mean_f1"]),
        ceiling=float(decision["selected_page_pool_atom_ceiling_f1"]),
    )
    return {
        "arm_id": arm_id,
        "selected_proposition_count": len(selected),
        "selected_source_chars": sum(
            len(str(row.get("raw_source_form") or ""))
            for row in selected
        ),
        "grouped_teacher": grouped,
        "unique_atoms": atoms,
        "semantic_support": support,
        "grouped_gap_closure_fraction": grouped_closure,
        "atom_gap_closure_fraction": atom_closure,
    }


def arm_is_promising(
    arm: dict[str, Any],
    *,
    control_support_recall: float,
    contract: dict[str, Any],
) -> bool:
    decision = contract["predeclared_decision"]
    minimum = float(decision["minimum_gap_closure_fraction"])
    return (
        float(arm["grouped_teacher"]["mean_f1"])
        > float(decision["wave6_grouped_mean_f1"])
        and float(arm["unique_atoms"]["mean_f1"])
        > float(decision["wave6_atom_mean_f1"])
        and float(arm["grouped_gap_closure_fraction"]) >= minimum
        and float(arm["atom_gap_closure_fraction"]) >= minimum
        and int(arm["semantic_support"]["known_miss_recovery_count"])
        >= int(decision["minimum_known_miss_recovery_count"])
        and float(arm["semantic_support"]["atom_support_recall_at_12"])
        >= float(control_support_recall)
    )


def _tree_identity(path: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    files = 0
    for file_path in sorted(
        row
        for row in path.rglob("*")
        if row.is_file() and ".cache" not in row.parts
    ):
        relative = file_path.relative_to(path).as_posix()
        payload = file_path.read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        total += len(payload)
        files += 1
    return {
        "file_count": files,
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


def snapshot_download_exact(
    *,
    repository: str,
    revision: str,
    destination: pathlib.Path,
    allow_patterns: list[str] | None = None,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    resolved = snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )
    elapsed = time.perf_counter() - started
    if pathlib.Path(resolved).resolve() != destination.resolve():
        raise RuntimeError("unexpected Hugging Face snapshot path")
    identity = _tree_identity(destination)
    return {
        "repository": repository,
        "revision": revision,
        "download_seconds": elapsed,
        **identity,
    }


class BGEM3Scorer:
    arm_ids = tuple(BGE_ARM_KEYS)

    def __init__(
        self,
        model_path: pathlib.Path,
        *,
        max_passage_length: int,
        hybrid_weights: list[float],
    ) -> None:
        import torch
        from FlagEmbedding import BGEM3FlagModel

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.model = BGEM3FlagModel(
            str(model_path),
            use_fp16=True,
            pooling_method="cls",
            devices="cuda:0",
        )
        self.load_seconds = time.perf_counter() - started
        self.max_passage_length = max_passage_length
        self.hybrid_weights = hybrid_weights
        self._torch = torch
        self.inference_seconds = 0.0

    def score(
        self,
        query: str,
        propositions: list[dict[str, Any]],
    ) -> dict[str, list[float]]:
        pairs = [
            [query, str(row["display_text"])] for row in propositions
        ]
        started = time.perf_counter()
        scores = self.model.compute_score(
            pairs,
            max_passage_length=self.max_passage_length,
            weights_for_different_modes=self.hybrid_weights,
        )
        self.inference_seconds += time.perf_counter() - started
        return {
            arm_id: [float(value) for value in scores[key]]
            for arm_id, key in BGE_ARM_KEYS.items()
        }

    def resource_metrics(self) -> dict[str, Any]:
        return {
            "model_load_seconds": self.load_seconds,
            "ranking_inference_seconds": self.inference_seconds,
            "peak_gpu_memory_mib": (
                self._torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
        }

    def close(self) -> None:
        torch = self._torch
        model = self.model
        self.model = None
        del model
        gc.collect()
        torch.cuda.empty_cache()


class Qwen3RerankerScorer:
    arm_ids = (QWEN_ARM_ID,)

    def __init__(
        self,
        model_path: pathlib.Path,
        *,
        instruction: str,
        max_length: int,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).cuda().eval()
        self.load_seconds = time.perf_counter() - started
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self.inference_seconds = 0.0
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the "
            "Query and the Instruct provided. Note that the answer can only "
            "be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        suffix = (
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(
            prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            suffix, add_special_tokens=False
        )
        if self.true_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Qwen reranker yes token is unknown")
        if self.false_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Qwen reranker no token is unknown")

    def _formatted(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _score_batch(self, pairs: list[str]) -> list[float]:
        torch = self._torch
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(
                self.max_length
                - len(self.prefix_tokens)
                - len(self.suffix_tokens)
            ),
        )
        for index, row in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = (
                self.prefix_tokens + row + self.suffix_tokens
            )
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            true_vector = logits[:, self.true_id]
            false_vector = logits[:, self.false_id]
            pair_logits = torch.stack(
                [false_vector, true_vector],
                dim=1,
            )
            scores = torch.nn.functional.log_softmax(
                pair_logits, dim=1
            )[:, 1].exp()
        return [float(value) for value in scores.cpu().tolist()]

    def score(
        self,
        query: str,
        propositions: list[dict[str, Any]],
    ) -> dict[str, list[float]]:
        formatted = [
            self._formatted(query, str(row["display_text"]))
            for row in propositions
        ]
        scores: list[float] = []
        started = time.perf_counter()
        for start in range(0, len(formatted), self.batch_size):
            scores.extend(
                self._score_batch(
                    formatted[start:start + self.batch_size]
                )
            )
        self.inference_seconds += time.perf_counter() - started
        return {QWEN_ARM_ID: scores}

    def resource_metrics(self) -> dict[str, Any]:
        return {
            "model_load_seconds": self.load_seconds,
            "ranking_inference_seconds": self.inference_seconds,
            "peak_gpu_memory_mib": (
                self._torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
        }

    def close(self) -> None:
        torch = self._torch
        model = self.model
        tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def _selected_for_scorer(
    scorer: Any,
    paper_inputs: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm_id: {} for arm_id in scorer.arm_ids
    }
    for paper in paper_inputs:
        propositions = list(paper["page_pool"])
        per_arm_channel_scores: dict[
            str, dict[str, list[float]]
        ] = {arm_id: {} for arm_id in scorer.arm_ids}
        for channel, channel_query in channel_queries(str(paper["query"])):
            scores = scorer.score(channel_query, propositions)
            for arm_id in scorer.arm_ids:
                per_arm_channel_scores[arm_id][channel] = scores[arm_id]
        for arm_id in scorer.arm_ids:
            result[arm_id][str(paper["item_id"])] = (
                interleave_scored_channels(
                    propositions,
                    per_arm_channel_scores[arm_id],
                    arm_id=arm_id,
                    top_k=12,
                )
            )
    return result


def _prepare_papers(
    *,
    fixture: dict[str, Any],
    workdir: pathlib.Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    config = dict(fixture["qwen_config"])
    if int(config["top_k_regions"]) != 12:
        raise ValueError("reranker bakeoff requires top_k_regions=12")
    if int(config["max_region_chars"]) != 650:
        raise ValueError("reranker bakeoff requires max_region_chars=650")

    papers: list[dict[str, Any]] = []
    grouped_refs: list[dict[str, Any]] = []
    teacher_item_id = ""
    for source_paper in fixture["papers"]:
        paper = dict(source_paper)
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise RuntimeError(
                f"page-count mismatch for {paper['item_id']}"
            )
        query = (
            str(fixture["topic"])
            + " "
            + title
            + " "
            + str(paper["item_id"])
        ).strip()
        regions = exact_source_regions(pages, max_chars=650)
        selected_regions = rank_regions(query, regions, top_k=32)[:12]
        for rank, region in enumerate(selected_regions, start=1):
            region["rank"] = rank
            region["propositions"] = exact_propositions(region)
        full_props = full_pdf_propositions(pages)
        page_pool = page_conditioned_proposition_pool(
            full_props, selected_regions
        )
        control = rank_page_conditioned_propositions(
            query, page_pool, top_k=12
        )
        papers.append(
            {
                "item_id": str(paper["item_id"]),
                "arxiv_id": str(paper["arxiv_id"]),
                "title": title,
                "query": query,
                "acquisition": acquisition,
                "candidate_region_count": len(regions),
                "coarse_selected_region_count": len(selected_regions),
                "selected_page_count": len(
                    {int(row["page_number"]) for row in selected_regions}
                ),
                "page_pool": page_pool,
                "control_selected": control,
            }
        )
        if paper.get("teacher_claims"):
            teacher_item_id = str(paper["item_id"])
            grouped_refs = _reference_rows(
                list(paper["teacher_claims"]),
                [],
            )["claims"]
    if not grouped_refs or not teacher_item_id:
        raise RuntimeError("fixture contains no grouped teacher references")
    return papers, grouped_refs, [], teacher_item_id


def run_reranker_bakeoff(
    *,
    fixture: dict[str, Any],
    contract: dict[str, Any],
    support_fixture: dict[str, Any],
    workdir: pathlib.Path,
) -> dict[str, Any]:
    papers, grouped_refs, _, teacher_item_id = _prepare_papers(
        fixture=fixture,
        workdir=workdir,
    )
    atom_refs = [
        {
            "id": str(row["id"]),
            "parent_teacher_id": str(row["parent_teacher_id"]),
            "page_number": int(row["page_number"]),
            "type": str(row["type"]),
            "normalized_form": str(row["normalized_form"]),
        }
        for row in support_fixture["atoms"]
    ]

    selections: dict[str, dict[str, list[dict[str, Any]]]] = {
        CONTROL_ARM_ID: {
            str(row["item_id"]): list(row["control_selected"])
            for row in papers
        }
    }
    resources: dict[str, Any] = {
        CONTROL_ARM_ID: {
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
        hybrid_weights=[
            float(value) for value in bge_cfg["hybrid_weights"]
        ],
    )
    selections.update(_selected_for_scorer(bge, papers))
    bge_metrics = bge.resource_metrics()
    bge.close()
    for arm_id in BGE_ARM_KEYS:
        resources[arm_id] = {
            "model_download_bytes": int(bge_download["bytes"]),
            "model_download_seconds": float(
                bge_download["download_seconds"]
            ),
            "model_snapshot_tree_sha256": str(
                bge_download["tree_sha256"]
            ),
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
    selections.update(_selected_for_scorer(qwen, papers))
    qwen_metrics = qwen.resource_metrics()
    qwen.close()
    resources[QWEN_ARM_ID] = {
        "model_download_bytes": int(qwen_download["bytes"]),
        "model_download_seconds": float(
            qwen_download["download_seconds"]
        ),
        "model_snapshot_tree_sha256": str(
            qwen_download["tree_sha256"]
        ),
        **qwen_metrics,
    }
    shutil.rmtree(qwen_dir, ignore_errors=True)

    teacher_paper = next(
        row for row in papers if row["item_id"] == teacher_item_id
    )
    arm_results: dict[str, Any] = {}
    control_support_recall = 0.0
    for arm in contract["arms"]:
        arm_id = str(arm["id"])
        selected = selections[arm_id][teacher_item_id]
        evaluated = evaluate_arm(
            arm_id=arm_id,
            selected=selected,
            grouped_references=grouped_refs,
            atom_references=atom_refs,
            support_fixture=support_fixture,
            decision_contract=contract,
        )
        evaluated["resources"] = resources[arm_id]
        evaluated["teacher_selected_propositions"] = [
            {
                "proposition_id": str(row["proposition_id"]),
                "page_number": int(row["page_number"]),
                "retrieval_channel": str(row["retrieval_channel"]),
                "retrieval_rank": int(row["retrieval_rank"]),
                "raw_source_form": str(row["raw_source_form"]),
            }
            for row in selected
        ]
        arm_results[arm_id] = evaluated
        if arm_id == CONTROL_ARM_ID:
            control_support_recall = float(
                evaluated["semantic_support"][
                    "atom_support_recall_at_12"
                ]
            )

    for arm_id, evaluated in arm_results.items():
        evaluated["promising"] = (
            False
            if arm_id == CONTROL_ARM_ID
            else arm_is_promising(
                evaluated,
                control_support_recall=control_support_recall,
                contract=contract,
            )
        )

    promising = [
        arm_id
        for arm_id, row in arm_results.items()
        if bool(row["promising"])
    ]
    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_PROPOSITION_RERANKER_BAKEOFF",
        "benchmark_id": contract["benchmark_id"],
        "baseline_sha": contract["baseline_sha"],
        "source_run_id": fixture["source_run_id"],
        "teacher_item_id": teacher_item_id,
        "paper_count": len(papers),
        "arms": arm_results,
        "promising_arms": promising,
        "decision": "PROMISING" if promising else "NEGATIVE",
        "paper_pool_stats": [
            {
                "item_id": str(row["item_id"]),
                "selected_page_count": int(row["selected_page_count"]),
                "page_pool_proposition_count": len(row["page_pool"]),
                "control_selected_proposition_count": len(
                    row["control_selected"]
                ),
            }
            for row in papers
        ],
        "model_revisions": {
            key: {
                "repository": str(value["repository"]),
                "revision": str(value["revision"]),
            }
            for key, value in contract["model_inputs"].items()
        },
        "external_model_downloads": 2,
        "qwen35_extract_model_calls": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_verify_changed": False,
    }
