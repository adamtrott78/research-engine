import json
from pathlib import Path

from research_engine.corpus_loader import validate_corpus_fixture


DELTA = Path(__file__).parent / "data" / "rf_osr_counterevidence_delta.json"


def test_counterevidence_delta_is_valid_and_bounded():
    data = json.loads(DELTA.read_text())
    validate_corpus_fixture(data)
    assert len(data["works"]) == 2
    assert len(data["evidence_objects"]) == 8
    assert len(data["evidence_relations"]) == 5
    assert {x["source_level"] for x in data["evidence_objects"]} == {"FULLTEXT_STRUCTURED"}
    assert all(x.get("source_artifact_id") is None for x in data["evidence_objects"])
