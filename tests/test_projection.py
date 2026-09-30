import json
from pathlib import Path

from research_engine.fixture_projection import project_curated_fixture


def test_mixed_fidelity_projection_and_cross_work_edges():
    data=json.loads((Path(__file__).parent/'data'/'mini_corpus.json').read_text())
    p=project_curated_fixture(data)
    assert p.exact_pdf_work_count == 3
    assert p.ocr_derived_work_count == 1
    assert p.graph.relations['rel:baye-liu'].relation_type == 'USES'
    assert p.graph.relations['rel:broggi-liu'].relation_type == 'COMPARES_AGAINST'
