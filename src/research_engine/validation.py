from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FidelitySummary:
    checked_anchors: int
    exact_match: int
    normalization_only: int
    ocr_minor_error: int
    ocr_semantic_error: int
    locator_unresolved: int

    @property
    def semantic_preservation_rate(self) -> float:
        if self.checked_anchors == 0:
            return 0.0
        good = self.exact_match + self.normalization_only + self.ocr_minor_error
        return good / self.checked_anchors

    def assert_promotion_safe(self) -> None:
        if self.ocr_semantic_error or self.locator_unresolved:
            raise ValueError("OCR promotion benchmark contains semantic/unresolved failures")
        if self.semantic_preservation_rate != 1.0:
            raise ValueError("semantic preservation must be 1.0 for selected promotion anchors")
