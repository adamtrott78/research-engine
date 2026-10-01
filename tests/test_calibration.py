from research_engine.calibration import ExtractionProvenanceAudit


def test_extraction_provenance_clusters_are_deterministic():
    assert ExtractionProvenanceAudit._clusters([]) == []
    assert ExtractionProvenanceAudit._clusters([7, 2, 3, 3, 10, 8]) == [
        {"start_page": 2, "end_page": 3, "page_count": 2},
        {"start_page": 7, "end_page": 8, "page_count": 2},
        {"start_page": 10, "end_page": 10, "page_count": 1},
    ]
