from __future__ import annotations

import inspect
import json
import pathlib

from research_engine import local_extract_wave3
from research_engine.local_extract_wave3 import (
    decide_wave3,
    exact_source_regions_soft_terminal,
    terminal_at_end,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-wave3-contract.v1.json"
)


def _page(text: str) -> list[dict]:
    return [{"page_number": 1, "text": text}]


def test_wave3_contract_changes_only_region_overflow_behavior() -> None:
    contract = json.loads(CONTRACT.read_text())
    fixed = set(contract["experimental_control"]["fixed"])
    changed = set(contract["experimental_control"]["changed"])
    assert "FACET_INTERLEAVED_BM25_V1 ranking algorithm and facet queries" in fixed
    assert "top_k_regions=12" in fixed
    assert "Wave-2 one-region-per-model-call execution" in fixed
    assert "Wave-2 one-claim-or-skip model output contract" in fixed
    assert len(changed) == 2
    assert (
        contract["region_construction"]["soft_target_raw_chars"] == 650
    )
    assert contract["region_construction"]["hard_max_raw_chars"] == 1200
    assert contract["execution"]["new_privileged_authority_required"] is False
    assert contract["execution"]["gitrunner_substrate_change_required"] is False


def test_terminal_predicate_matches_existing_closer_semantics() -> None:
    assert terminal_at_end("claim.")
    assert terminal_at_end('claim."')
    assert terminal_at_end("claim.)")
    assert terminal_at_end("claim?]")
    assert not terminal_at_end("claim")
    assert not terminal_at_end("10.8")


def test_soft_target_extends_to_next_terminal_line() -> None:
    text = (
        "A" * 35
        + " alpha\n"
        + "B" * 35
        + " beta\n"
        + "C" * 20
        + " end.\n"
    )
    regions = exact_source_regions_soft_terminal(
        _page(text),
        soft_chars=64,
        hard_chars=160,
    )
    assert len(regions) == 1
    region = regions[0]
    assert region["start_char"] == 0
    assert region["end_char"] == len(text)
    assert region["raw_source_form"] == text
    assert region["soft_extended"] is True
    assert region["forced_hard_split"] is False


def test_hard_ceiling_forces_bounded_line_boundary_split() -> None:
    first = "A" * 50 + "\n"
    second = "B" * 55 + "\n"
    tail = "tail sentence.\n"
    text = first + second + tail
    regions = exact_source_regions_soft_terminal(
        _page(text),
        soft_chars=64,
        hard_chars=100,
    )
    assert len(regions) == 2
    assert regions[0]["raw_source_form"] == first + second
    assert regions[0]["soft_extended"] is True
    assert regions[0]["forced_hard_split"] is True
    assert regions[0]["end_char"] <= 120
    assert regions[1]["forced_hard_split"] is False


def test_terminal_before_soft_target_preserves_wave2_boundary() -> None:
    text = "Short scientific claim.\nSecond scientific claim.\n"
    regions = exact_source_regions_soft_terminal(
        _page(text),
        soft_chars=64,
        hard_chars=160,
    )
    assert len(regions) == 2
    assert all(row["soft_extended"] is False for row in regions)
    assert all(row["forced_hard_split"] is False for row in regions)


def test_wave3_module_has_single_experimental_entrypoint() -> None:
    source = inspect.getsource(local_extract_wave3)
    assert source.count("def run_wave3(") == 1
    assert source.count("def exact_source_regions_soft_terminal(") == 1
    assert "LOCAL_EXTRACT_WAVE2_REGION_ISOLATED_NORMALIZATION" not in source


def test_wave3_research_gate_remains_unchanged() -> None:
    report = {
        "hard_safety": {"all_safe": True},
        "metrics": {"job_local_completion_fraction": 1.0},
        "teacher_subset_agreement": {
            "qwen_precision_at_f1_0_5": 0.0,
            "teacher_claim_coverage_at_f1_0_5": 0.0,
            "page_agreement_on_passing_matches": 0.0,
            "type_agreement_on_passing_matches": 0.0,
        },
    }
    assert decide_wave3(report) == "PARTIAL"

    report["metrics"]["job_local_completion_fraction"] = 0.49
    assert decide_wave3(report) == "REJECT"
