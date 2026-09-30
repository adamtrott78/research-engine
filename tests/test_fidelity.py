from research_engine.validation import FidelitySummary


def test_exact_pdf_promotion_benchmark_passes():
    s=FidelitySummary(checked_anchors=11,exact_match=7,normalization_only=4,ocr_minor_error=0,ocr_semantic_error=0,locator_unresolved=0)
    s.assert_promotion_safe()
    assert s.semantic_preservation_rate == 1.0
