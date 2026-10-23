from __future__ import annotations

import itertools
import pathlib
from collections import defaultdict
from typing import Any, Iterable

from research_engine.local_extract_wave1 import (
    display_text,
    download_exact_pdf,
    exact_source_regions,
    extract_pdf_pages,
    rank_regions,
    token_f1,
)
from research_engine.local_extract_wave4 import exact_propositions
from research_engine.local_extract_wave6 import deterministic_proposition_selections


def _page_region(page: dict[str, Any]) -> dict[str, Any]:
    page_number = int(page["page_number"])
    raw = str(page.get("text") or "")
    return {
        "region_id": f"page:p{page_number}",
        "page_number": page_number,
        "start_char": 0,
        "end_char": len(raw),
        "raw_source_form": raw,
        "display_text": display_text(raw),
        "section_hint": "",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
    }


def full_pdf_propositions(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    propositions: list[dict[str, Any]] = []
    for page in pages:
        propositions.extend(exact_propositions(_page_region(page)))
    return propositions


def sentence_windows(
    propositions: list[dict[str, Any]],
    *,
    window_size: int,
) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for proposition in propositions:
        by_page[int(proposition["page_number"])].append(proposition)

    windows: list[dict[str, Any]] = []
    for page_number in sorted(by_page):
        rows = sorted(
            by_page[page_number],
            key=lambda row: (
                int(row["start_char"]),
                int(row["end_char"]),
                str(row["proposition_id"]),
            ),
        )
        for index in range(len(rows)):
            group = rows[index:index + window_size]
            if len(group) != window_size:
                continue
            start = int(group[0]["start_char"])
            end = int(group[-1]["end_char"])
            rendered = display_text(
                " ".join(str(row["display_text"]) for row in group)
            )
            raw = " ".join(str(row["raw_source_form"]) for row in group)
            windows.append(
                {
                    "region_id": (
                        f"sentence-window-{window_size}:p{page_number}:"
                        f"{start:06d}-{end:06d}"
                    ),
                    "page_number": page_number,
                    "start_char": start,
                    "end_char": end,
                    "raw_source_form": raw,
                    "display_text": rendered,
                    "section_hint": str(group[0].get("section_hint") or ""),
                    "source_layer": "AUTHORITATIVE_PDF_TEXT",
                    "window_size": window_size,
                    "proposition_ids": [
                        str(row["proposition_id"]) for row in group
                    ],
                }
            )
    return windows


def select_under_char_budget(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    budget_chars: int,
) -> list[dict[str, Any]]:
    if budget_chars <= 0:
        raise ValueError("budget_chars must be positive")
    if not candidates:
        return []

    ranked = rank_regions(query, candidates, top_k=len(candidates))
    selected: list[dict[str, Any]] = []
    used = 0
    for row in ranked:
        size = len(str(row["raw_source_form"]))
        if used + size > budget_chars:
            continue
        selected.append(row)
        used += size
    return selected


def _best_alignment(
    reference: str,
    candidates: Iterable[dict[str, Any]],
    *,
    text_field: str = "display_text",
) -> dict[str, Any]:
    best_score = 0.0
    best: dict[str, Any] | None = None
    for row in candidates:
        score = token_f1(reference, str(row.get(text_field) or ""))
        if best is None or score > best_score:
            best_score = score
            best = row

    if best is None:
        return {
            "f1": 0.0,
            "candidate_id": None,
            "page_number": None,
            "text": None,
        }
    return {
        "f1": best_score,
        "candidate_id": str(
            best.get("proposition_id")
            or best.get("region_id")
            or best.get("candidate_id")
            or ""
        ),
        "page_number": int(best["page_number"]),
        "text": str(best.get(text_field) or ""),
    }


def _best_bundle_alignment(
    reference: str,
    propositions: list[dict[str, Any]],
    *,
    max_bundle: int,
) -> dict[str, Any]:
    if max_bundle <= 0:
        raise ValueError("max_bundle must be positive")
    if not propositions:
        return {
            "f1": 0.0,
            "candidate_ids": [],
            "page_numbers": [],
            "text": None,
        }

    best_score = -1.0
    best_group: tuple[dict[str, Any], ...] = ()
    upper = min(max_bundle, len(propositions))
    for size in range(1, upper + 1):
        for group in itertools.combinations(propositions, size):
            text = display_text(
                " ".join(str(row["display_text"]) for row in group)
            )
            score = token_f1(reference, text)
            if score > best_score:
                best_score = score
                best_group = group

    text = display_text(
        " ".join(str(row["display_text"]) for row in best_group)
    )
    return {
        "f1": max(best_score, 0.0),
        "candidate_ids": [
            str(row["proposition_id"]) for row in best_group
        ],
        "page_numbers": [
            int(row["page_number"]) for row in best_group
        ],
        "text": text,
    }


def _reference_rows(
    teacher_claims: list[dict[str, Any]],
    teacher_atoms: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    claims = [
        {
            "id": str(row["id"]),
            "parent_teacher_id": str(row["id"]),
            "page_number": int(row["page_number"]),
            "type": str(row["type"]),
            "normalized_form": str(row["normalized_form"]),
        }
        for row in teacher_claims
    ]
    atoms = [
        {
            "id": str(row["id"]),
            "parent_teacher_id": str(row["parent_teacher_id"]),
            "page_number": int(row["page_number"]),
            "type": str(row["type"]),
            "normalized_form": str(row["normalized_form"]),
            **(
                {"duplicate_of": str(row["duplicate_of"])}
                if row.get("duplicate_of")
                else {}
            ),
        }
        for row in teacher_atoms
    ]
    return {"claims": claims, "atoms": atoms}


def _evaluate_references(
    references: list[dict[str, Any]],
    *,
    full_propositions: list[dict[str, Any]],
    selected_regions: list[dict[str, Any]],
    selected_propositions: list[dict[str, Any]],
    wave6_selected: list[dict[str, Any]],
    sentence_window_2: list[dict[str, Any]],
    sentence_window_3: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_pages = {
        int(row["page_number"]) for row in selected_regions
    }
    results: list[dict[str, Any]] = []
    for reference in references:
        expected_page = int(reference["page_number"])
        expected_page_props = [
            row
            for row in full_propositions
            if int(row["page_number"]) == expected_page
        ]
        result = dict(reference)
        result["expected_page_retrieved_current_top12"] = (
            expected_page in selected_pages
        )
        result["parsed_pdf_best_single_proposition"] = _best_alignment(
            str(reference["normalized_form"]),
            full_propositions,
        )
        result[
            "expected_teacher_page_best_single_proposition"
        ] = _best_alignment(
            str(reference["normalized_form"]),
            expected_page_props,
        )
        result["current_top12_best_region"] = _best_alignment(
            str(reference["normalized_form"]),
            selected_regions,
        )
        result["current_top12_best_single_proposition"] = _best_alignment(
            str(reference["normalized_form"]),
            selected_propositions,
        )
        result[
            "current_wave6_selected_single_proposition"
        ] = _best_alignment(
            str(reference["normalized_form"]),
            wave6_selected,
        )
        result["current_top12_best_bundle_2"] = _best_bundle_alignment(
            str(reference["normalized_form"]),
            selected_propositions,
            max_bundle=2,
        )
        result["current_top12_best_bundle_3"] = _best_bundle_alignment(
            str(reference["normalized_form"]),
            selected_propositions,
            max_bundle=3,
        )
        result["sentence_window_2_fixed_budget"] = _best_alignment(
            str(reference["normalized_form"]),
            sentence_window_2,
        )
        result["sentence_window_3_fixed_budget"] = _best_alignment(
            str(reference["normalized_form"]),
            sentence_window_3,
        )
        results.append(result)
    return results


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]["f1"]) for row in rows) / len(rows)


def summarize_reference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "parsed_pdf_best_single_proposition",
        "expected_teacher_page_best_single_proposition",
        "current_top12_best_region",
        "current_top12_best_single_proposition",
        "current_wave6_selected_single_proposition",
        "current_top12_best_bundle_2",
        "current_top12_best_bundle_3",
        "sentence_window_2_fixed_budget",
        "sentence_window_3_fixed_budget",
    )
    return {
        "reference_count": len(rows),
        "expected_page_retrieval_fraction": (
            sum(
                int(bool(row["expected_page_retrieved_current_top12"]))
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
        "mean_f1": {metric: _mean(rows, metric) for metric in metrics},
    }


def _page_recall(
    historical_pages: list[int],
    selected: list[dict[str, Any]],
) -> float:
    expected = {int(value) for value in historical_pages}
    if not expected:
        return 1.0
    actual = {int(row["page_number"]) for row in selected}
    return len(expected & actual) / len(expected)


def run_post_wave6_diagnostics(
    *,
    fixture: dict[str, Any],
    diagnostic_contract: dict[str, Any],
    workdir: pathlib.Path,
) -> dict[str, Any]:
    config = dict(fixture["qwen_config"])
    top_k = int(config["top_k_regions"])
    max_chars = int(config["max_region_chars"])
    if top_k != 12 or max_chars != 650:
        raise ValueError(
            "Post-Wave-6 diagnostics require the Wave-6 retrieval baseline"
        )

    teacher_atoms = list(diagnostic_contract["teacher_atomization"])
    paper_rows: list[dict[str, Any]] = []
    teacher_detail: dict[str, Any] | None = None
    baseline_page_hits = 0
    window2_page_hits = 0
    window3_page_hits = 0
    total_historical_pages = 0

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
        selected = ranked[:top_k]
        for rank, region in enumerate(selected, start=1):
            region["rank"] = rank
            region["propositions"] = exact_propositions(region)

        baseline_budget = sum(
            len(str(row["raw_source_form"])) for row in selected
        )
        all_propositions = full_pdf_propositions(pages)
        windows2 = sentence_windows(all_propositions, window_size=2)
        windows3 = sentence_windows(all_propositions, window_size=3)
        selected_windows2 = select_under_char_budget(
            query, windows2, budget_chars=baseline_budget
        )
        selected_windows3 = select_under_char_budget(
            query, windows3, budget_chars=baseline_budget
        )

        historical_pages = [
            int(value) for value in paper["historical_evidence_pages"]
        ]
        expected = set(historical_pages)
        baseline_pages = {int(row["page_number"]) for row in selected}
        window2_pages = {
            int(row["page_number"]) for row in selected_windows2
        }
        window3_pages = {
            int(row["page_number"]) for row in selected_windows3
        }
        total_historical_pages += len(expected)
        baseline_page_hits += len(expected & baseline_pages)
        window2_page_hits += len(expected & window2_pages)
        window3_page_hits += len(expected & window3_pages)

        paper_row = {
            "item_id": str(paper["item_id"]),
            "arxiv_id": str(paper["arxiv_id"]),
            "title": title,
            "acquisition": acquisition,
            "page_count": len(pages),
            "candidate_region_count": len(regions),
            "baseline_selected_region_count": len(selected),
            "baseline_selected_source_chars": baseline_budget,
            "historical_evidence_page_count": len(expected),
            "current_top12_historical_page_recall": _page_recall(
                historical_pages, selected
            ),
            "sentence_window_2_fixed_budget_historical_page_recall": (
                _page_recall(historical_pages, selected_windows2)
            ),
            "sentence_window_3_fixed_budget_historical_page_recall": (
                _page_recall(historical_pages, selected_windows3)
            ),
            "sentence_window_2_selected_count": len(selected_windows2),
            "sentence_window_2_selected_source_chars": sum(
                len(str(row["raw_source_form"]))
                for row in selected_windows2
            ),
            "sentence_window_3_selected_count": len(selected_windows3),
            "sentence_window_3_selected_source_chars": sum(
                len(str(row["raw_source_form"]))
                for row in selected_windows3
            ),
        }
        paper_rows.append(paper_row)

        if paper.get("teacher_claims"):
            selected_propositions = [
                proposition
                for region in selected
                for proposition in region["propositions"]
            ]
            wave6_selected_by_region = (
                deterministic_proposition_selections(query, selected)
            )
            wave6_selected = list(wave6_selected_by_region.values())
            refs = _reference_rows(
                list(paper["teacher_claims"]),
                teacher_atoms,
            )
            claim_rows = _evaluate_references(
                refs["claims"],
                full_propositions=all_propositions,
                selected_regions=selected,
                selected_propositions=selected_propositions,
                wave6_selected=wave6_selected,
                sentence_window_2=selected_windows2,
                sentence_window_3=selected_windows3,
            )
            atom_rows = _evaluate_references(
                refs["atoms"],
                full_propositions=all_propositions,
                selected_regions=selected,
                selected_propositions=selected_propositions,
                wave6_selected=wave6_selected,
                sentence_window_2=selected_windows2,
                sentence_window_3=selected_windows3,
            )
            unique_atom_rows = [
                row for row in atom_rows if not row.get("duplicate_of")
            ]
            teacher_detail = {
                "paper_id": str(paper["item_id"]),
                "legacy_claims": claim_rows,
                "legacy_claim_summary": summarize_reference_rows(claim_rows),
                "atoms": atom_rows,
                "atom_summary_including_duplicate": summarize_reference_rows(
                    atom_rows
                ),
                "unique_atoms": unique_atom_rows,
                "unique_atom_summary": summarize_reference_rows(
                    unique_atom_rows
                ),
            }

    if teacher_detail is None:
        raise RuntimeError("fixture contains no normalized teacher subset")

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_POST_WAVE6_CODEX_PARITY_DIAGNOSTICS",
        "source_run_id": fixture["source_run_id"],
        "benchmark_id": diagnostic_contract["benchmark_id"],
        "baseline_sha": diagnostic_contract["baseline_sha"],
        "recovery_inventory": diagnostic_contract["recovery_inventory"],
        "fixed_control": diagnostic_contract["fixed_control"],
        "metrics": {
            "paper_count": len(paper_rows),
            "historical_evidence_object_count": int(
                fixture["historical"]["evidence_object_count"]
            ),
            "historical_evidence_bearing_page_count": total_historical_pages,
            "current_top12_historical_page_recall": (
                baseline_page_hits / total_historical_pages
                if total_historical_pages
                else 1.0
            ),
            "sentence_window_2_fixed_budget_historical_page_recall": (
                window2_page_hits / total_historical_pages
                if total_historical_pages
                else 1.0
            ),
            "sentence_window_3_fixed_budget_historical_page_recall": (
                window3_page_hits / total_historical_pages
                if total_historical_pages
                else 1.0
            ),
        },
        "teacher_diagnostics": teacher_detail,
        "papers": paper_rows,
        "scientific_state_materialized": False,
        "production_routing_changed": False,
        "qwen_model_calls": 0,
        "external_model_downloads": 0,
    }
