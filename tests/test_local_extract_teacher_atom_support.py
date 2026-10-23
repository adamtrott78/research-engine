from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTION = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-proposition-reranker-candidate-selection.v1.json"
)
SUPPORT = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-teacher-atom-support.v1.json"
)


def test_reranker_revisions_are_immutable_and_pinned() -> None:
    contract = json.loads(SELECTION.read_text())
    rows = {row["id"]: row for row in contract["candidate_models"]}
    assert rows["BGE_M3"]["immutable_revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert rows["QWEN3_RERANKER_0_6B"]["immutable_revision"] == (
        "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    )
    assert all(
        row["immutable_revision_required_before_download"] is False
        for row in rows.values()
    )
    assert contract["execution_boundaries"][
        "revision_and_semantic_fixture_frozen"
    ] is True
    assert contract["execution_boundaries"][
        "model_downloads_authorized_by_this_contract"
    ] is False


def test_semantic_support_fixture_is_unique_and_predeclared() -> None:
    fixture = json.loads(SUPPORT.read_text())
    atoms = fixture["atoms"]
    assert len(atoms) == 13
    assert len({row["id"] for row in atoms}) == 13
    assert "e5.a" not in {row["id"] for row in atoms}
    assert fixture["boundaries"]["frozen_before_neural_rankings"] is True
    assert fixture["boundaries"]["learned_evaluator_used"] is False
    assert fixture["boundaries"]["legacy_lexical_metric_replaced"] is False


def test_semantic_support_uses_expected_teacher_pages_and_nonempty_anchors() -> None:
    fixture = json.loads(SUPPORT.read_text())
    expected_pages = {
        "e1.a": 2,
        "e1.b": 2,
        "e1.c": 2,
        "e2.a": 2,
        "e2.b": 2,
        "e3.a": 2,
        "e3.b": 2,
        "e3.c": 2,
        "e4.a": 4,
        "e4.b": 4,
        "e6.a": 6,
        "e6.b": 6,
        "e6.c": 6,
    }
    assert {row["id"]: row["page_number"] for row in fixture["atoms"]} == expected_pages
    assert all(row["required_anchors"] for row in fixture["atoms"])
    assert all(
        anchor.isalnum() and anchor == anchor.lower()
        for row in fixture["atoms"]
        for anchor in row["required_anchors"]
    )


def test_selection_contract_binds_frozen_support_fixture() -> None:
    contract = json.loads(SELECTION.read_text())
    semantic = contract["evaluation"]["semantic_support_track"]
    assert semantic["fixture"].endswith(
        "local-extract-teacher-atom-support.v1.json"
    )
    assert semantic["unique_atom_count"] == 13
    assert any(
        "No learned semantic evaluator" in item
        for item in semantic["design"]
    )
