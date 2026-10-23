from __future__ import annotations

import json
import pathlib

from research_engine.local_extract_reranker_bakeoff import (
    BGE_ARM_KEYS,
    CONTROL_ARM_ID,
    KNOWN_MISS_ATOMS,
    arm_is_promising,
    channel_queries,
    evaluate_atom_support,
    interleave_scored_channels,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-proposition-reranker-bakeoff.v1.json"
)
SUPPORT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-teacher-atom-support.v1.json"
)


def _prop(pid: str, page: int, text: str) -> dict:
    return {
        "proposition_id": pid,
        "parent_region_id": f"page:p{page}",
        "page_number": page,
        "start_char": 0,
        "end_char": len(text),
        "raw_source_form": text,
        "display_text": text,
        "section_hint": "",
        "source_layer": "AUTHORITATIVE_PDF_TEXT",
    }


def test_channel_queries_reuse_current_five_channel_order() -> None:
    rows = channel_queries("base")
    assert [name for name, _ in rows] == [
        "global",
        "facet:method",
        "facet:dataset",
        "facet:evaluation",
        "facet:robustness",
    ]
    assert rows[0][1] == "base"
    assert all(text.startswith("base") for _, text in rows)


def test_interleave_scored_channels_is_bounded_and_deterministic() -> None:
    props = [
        _prop("p1", 2, "alpha"),
        _prop("p2", 2, "beta"),
        _prop("p3", 2, "gamma"),
    ]
    channels = {
        "global": [0.9, 0.8, 0.7],
        "facet:method": [0.7, 0.95, 0.8],
        "facet:dataset": [0.6, 0.7, 0.99],
        "facet:evaluation": [0.9, 0.8, 0.7],
        "facet:robustness": [0.9, 0.8, 0.7],
    }
    selected = interleave_scored_channels(
        props, channels, arm_id="TEST", top_k=3
    )
    assert [row["proposition_id"] for row in selected] == [
        "p1", "p2", "p3"
    ]
    assert [row["retrieval_rank"] for row in selected] == [1, 2, 3]
    assert all(row["selector_id"] == "TEST" for row in selected)


def test_atom_support_can_compose_required_anchors_across_propositions() -> None:
    fixture = {
        "atoms": [
            {
                "id": "x",
                "parent_teacher_id": "p",
                "page_number": 2,
                "required_anchors": ["alphaone", "betatwo"],
            }
        ]
    }
    selected = [
        _prop("p1", 2, "Alpha one is present."),
        _prop("p2", 2, "Beta two is also present."),
    ]
    result = evaluate_atom_support(fixture, selected)
    assert result["supported_atom_count"] == 1
    assert result["atom_support_recall_at_12"] == 1.0
    assert result["atoms"][0]["supporting_proposition_ids"] == ["p1", "p2"]


def test_known_miss_set_is_frozen() -> None:
    assert KNOWN_MISS_ATOMS == ("e2.a", "e2.b", "e4.b")


def test_promising_gate_requires_both_lexical_gains_and_support() -> None:
    contract = json.loads(CONTRACT.read_text())
    arm = {
        "grouped_teacher": {"mean_f1": 0.34},
        "unique_atoms": {"mean_f1": 0.27},
        "grouped_gap_closure_fraction": 0.30,
        "atom_gap_closure_fraction": 0.31,
        "semantic_support": {
            "known_miss_recovery_count": 2,
            "atom_support_recall_at_12": 0.8,
        },
    }
    assert arm_is_promising(
        arm, control_support_recall=0.7, contract=contract
    )
    arm["semantic_support"]["known_miss_recovery_count"] = 1
    assert not arm_is_promising(
        arm, control_support_recall=0.7, contract=contract
    )


def test_bakeoff_contract_is_exactly_two_models_and_one_run() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert set(contract["model_inputs"]) == {
        "BGE_M3",
        "QWEN3_RERANKER_0_6B",
    }
    assert set(BGE_ARM_KEYS) == {
        "BGE_M3_DENSE",
        "BGE_M3_SPARSE",
        "BGE_M3_COLBERT",
        "BGE_M3_HYBRID_040_020_040",
    }
    assert contract["arms"][0]["id"] == CONTROL_ARM_ID
    assert contract["execution"]["execution_contract_id"] == (
        "gpu-single-public-ci-v1"
    )
    assert contract["execution"]["gpu_count"] == 1
    assert contract["execution"]["model_binding"] is None
    assert contract["execution"]["models_loaded_sequentially"] is True
    assert contract["boundaries"][
        "exactly_one_bounded_bakeoff_run_after_route_install"
    ] is True
    assert contract["boundaries"]["wave7_execution_authorized"] is False


def test_model_revisions_and_bge_reference_are_pinned() -> None:
    contract = json.loads(CONTRACT.read_text())
    bge = contract["model_inputs"]["BGE_M3"]
    qwen = contract["model_inputs"]["QWEN3_RERANKER_0_6B"]
    assert bge["revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert qwen["revision"] == (
        "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    )
    reference = bge["reference_implementation"]
    assert reference["tag"] == "v1.4.2"
    assert reference["commit"] == (
        "fd1a2bdf69488ffebe0327999d4400d8c8058a0b"
    )
    assert bge["runtime_adapter"]["id"] == "LOCAL_MINIMAL_BGE_M3_V1"
    assert bge["runtime_adapter"]["runtime_package_install_required"] is False
    assert bge["batch_size"] == 8
    support = json.loads(SUPPORT.read_text())
    assert len(support["atoms"]) == 13
