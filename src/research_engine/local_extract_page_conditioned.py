from __future__ import annotations

import pathlib
from typing import Any

from research_engine.local_extract_diagnostics import (
    _best_alignment,
    _evaluate_references,
    _reference_rows,
    full_pdf_propositions,
    summarize_reference_rows,
)
from research_engine.local_extract_wave1 import (
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    rank_regions,
)
from research_engine.local_extract_wave4 import exact_propositions
from research_engine.local_extract_wave6 import deterministic_proposition_selections


SELECTOR_ID = "PAGE_CONDITIONED_FACET_INTERLEAVED_BM25_V1"


def page_conditioned_proposition_pool(
    full_propositions: list[dict[str, Any]],
    selected_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_pages = {
        int(region["page_number"]) for region in selected_regions
    }
    return [
        dict(proposition)
        for proposition in full_propositions
        if int(proposition["page_number"]) in selected_pages
    ]


def rank_page_conditioned_propositions(
    query: str,
    propositions: list[dict[str, Any]],
    *,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for proposition in propositions:
        row = dict(proposition)
        row["region_id"] = str(proposition["proposition_id"])
        aliases.append(row)

    ranked = rank_regions(query, aliases, top_k=top_k)
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        proposition = dict(row)
        proposition["proposition_id"] = str(proposition["region_id"])
        proposition["retrieval_rank"] = rank
        proposition["selector_id"] = SELECTOR_ID
        result.append(proposition)
    return result


def _evaluate_page_conditioned(
    references: list[dict[str, Any]],
    *,
    page_pool: list[dict[str, Any]],
    page_selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in references:
        row = dict(reference)
        row["selected_page_pool_best_single_proposition"] = _best_alignment(
            str(reference["normalized_form"]),
            page_pool,
        )
        row["page_conditioned_selected_proposition"] = _best_alignment(
            str(reference["normalized_form"]),
            page_selected,
        )
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "reference_count": 0,
            "selected_page_pool_best_single_proposition_mean_f1": 0.0,
            "page_conditioned_selected_proposition_mean_f1": 0.0,
        }
    return {
        "reference_count": len(rows),
        "selected_page_pool_best_single_proposition_mean_f1": sum(
            float(
                row["selected_page_pool_best_single_proposition"]["f1"]
            )
            for row in rows
        )
        / len(rows),
        "page_conditioned_selected_proposition_mean_f1": sum(
            float(row["page_conditioned_selected_proposition"]["f1"])
            for row in rows
        )
        / len(rows),
    }


def run_page_conditioned_retrieval_diagnostic(
    *,
    fixture: dict[str, Any],
    diagnostic_contract: dict[str, Any],
    parent_diagnostic_contract: dict[str, Any],
    workdir: pathlib.Path,
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_chars = int(config["max_region_chars"])
    if top_k != 12 or max_chars != 650:
        raise ValueError(
            "Page-conditioned diagnostic requires the Wave-6 baseline"
        )

    teacher_atoms = list(parent_diagnostic_contract["teacher_atomization"])
    papers: list[dict[str, Any]] = []
    teacher_detail: dict[str, Any] | None = None

    for source_paper in fixture["papers"]:
        paper = dict(source_paper)
        pdf_path = workdir / "pdfs" / f"{paper['arxiv_id']}.pdf"
        acquisition = download_exact_pdf(paper, pdf_path)
        pages, title = extract_pdf_pages(pdf_path)
        if len(pages) != int(paper["total_pdf_pages"]):
            raise RuntimeError(
                f"page-count mismatch for {paper['item_id']}: "
                f"expected {paper['total_pdf_pages']} got {len(pages)}"
            )

        query = (
            str(fixture["topic"])
            + " "
            + title
            + " "
            + str(paper["item_id"])
        ).strip()

        regions = exact_source_regions(pages, max_chars=max_chars)
        ranked = rank_regions(query, regions, top_k=32)
        selected_regions = ranked[:top_k]
        for rank, region in enumerate(selected_regions, start=1):
            region["rank"] = rank
            region["propositions"] = exact_propositions(region)

        full_props = full_pdf_propositions(pages)
        page_pool = page_conditioned_proposition_pool(
            full_props,
            selected_regions,
        )
        page_selected = rank_page_conditioned_propositions(
            query,
            page_pool,
            top_k=12,
        )

        papers.append(
            {
                "item_id": str(paper["item_id"]),
                "arxiv_id": str(paper["arxiv_id"]),
                "title": title,
                "acquisition": acquisition,
                "candidate_region_count": len(regions),
                "coarse_selected_region_count": len(selected_regions),
                "coarse_selected_page_count": len(
                    {
                        int(row["page_number"])
                        for row in selected_regions
                    }
                ),
                "selected_page_proposition_pool_count": len(page_pool),
                "page_conditioned_selected_proposition_count": len(
                    page_selected
                ),
                "page_conditioned_selected_source_chars": sum(
                    len(str(row["raw_source_form"]))
                    for row in page_selected
                ),
            }
        )

        if paper.get("teacher_claims"):
            current_selected_propositions = [
                proposition
                for region in selected_regions
                for proposition in region["propositions"]
            ]
            wave6_selected = list(
                deterministic_proposition_selections(
                    query,
                    selected_regions,
                ).values()
            )
            refs = _reference_rows(
                list(paper["teacher_claims"]),
                teacher_atoms,
            )
            legacy_rows = _evaluate_page_conditioned(
                refs["claims"],
                page_pool=page_pool,
                page_selected=page_selected,
            )
            atom_rows = _evaluate_page_conditioned(
                refs["atoms"],
                page_pool=page_pool,
                page_selected=page_selected,
            )
            unique_atom_rows = [
                row for row in atom_rows if not row.get("duplicate_of")
            ]

            current_legacy = _evaluate_references(
                refs["claims"],
                full_propositions=full_props,
                selected_regions=selected_regions,
                selected_propositions=current_selected_propositions,
                wave6_selected=wave6_selected,
                sentence_window_2=[],
                sentence_window_3=[],
            )
            current_atoms = _evaluate_references(
                refs["atoms"],
                full_propositions=full_props,
                selected_regions=selected_regions,
                selected_propositions=current_selected_propositions,
                wave6_selected=wave6_selected,
                sentence_window_2=[],
                sentence_window_3=[],
            )
            current_unique_atoms = [
                row
                for row in current_atoms
                if not row.get("duplicate_of")
            ]

            teacher_detail = {
                "paper_id": str(paper["item_id"]),
                "legacy_claims": legacy_rows,
                "legacy_summary": _summary(legacy_rows),
                "legacy_baseline_summary": summarize_reference_rows(
                    current_legacy
                ),
                "atoms": atom_rows,
                "unique_atoms": unique_atom_rows,
                "unique_atom_summary": _summary(unique_atom_rows),
                "unique_atom_baseline_summary": summarize_reference_rows(
                    current_unique_atoms
                ),
            }

    if teacher_detail is None:
        raise RuntimeError("fixture contains no normalized teacher subset")

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_PAGE_CONDITIONED_PROPOSITION_RETRIEVAL",
        "benchmark_id": diagnostic_contract["benchmark_id"],
        "baseline_sha": diagnostic_contract["baseline_sha"],
        "selector_id": SELECTOR_ID,
        "max_selected_propositions": 12,
        "teacher_diagnostics": teacher_detail,
        "papers": papers,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
    }
