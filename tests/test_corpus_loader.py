import copy
import json
from pathlib import Path

import pytest

from research_engine.corpus_loader import CorpusFixtureError, validate_corpus_fixture


FIXTURE = Path(__file__).parent / "data" / "rf_osr_five_work.json"


def test_five_work_fixture_shape_and_relation_direction():
    data = json.loads(FIXTURE.read_text())
    validate_corpus_fixture(data)
    assert len(data["works"]) == 5
    assert len(data["evidence_objects"]) == 21
    assert len(data["evidence_relations"]) == 11


def test_fixture_rejects_reversed_support_edge():
    data = json.loads(FIXTURE.read_text())
    bad = copy.deepcopy(data)
    rel = next(r for r in bad["evidence_relations"] if r["relation_type"] == "SUPPORTS")
    rel["source_id"], rel["target_id"] = rel["target_id"], rel["source_id"]
    with pytest.raises(ValueError, match="SUPPORTS"):
        validate_corpus_fixture(bad)


def test_fixture_rejects_missing_locator_reference():
    data = json.loads(FIXTURE.read_text())
    bad = copy.deepcopy(data)
    bad["evidence_objects"][0]["source_locator_id"] = "missing"
    with pytest.raises(CorpusFixtureError, match="missing locator"):
        validate_corpus_fixture(bad)
