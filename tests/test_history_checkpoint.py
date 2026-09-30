import copy
import json
from pathlib import Path

import pytest

from research_engine.checkpoints import checkpoint_fingerprint
from research_engine.corpus_loader import ImmutableRecordConflict, validate_corpus_fixture


FIXTURE = Path(__file__).parent / "data" / "rf_osr_five_work.json"


def test_checkpoint_fingerprint_is_order_stable():
    scope = {"work_ids": ["b", "a"]}
    manifest_a = {"evidence_objects": [{"id": "e1", "sha256": "a" * 64}]}
    manifest_b = {"evidence_objects": [{"sha256": "a" * 64, "id": "e1"}]}
    assert checkpoint_fingerprint(scope, manifest_a) == checkpoint_fingerprint(scope, manifest_b)


def test_supersession_cannot_self_reference():
    data = json.loads(FIXTURE.read_text())
    bad = copy.deepcopy(data)
    bad["evidence_objects"][0]["supersedes_id"] = bad["evidence_objects"][0]["id"]
    with pytest.raises(ValueError, match="cannot supersede itself"):
        validate_corpus_fixture(bad)


def test_immutable_conflict_type_is_a_fixture_error():
    assert issubclass(ImmutableRecordConflict, ValueError)
