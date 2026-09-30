from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import EvidenceObject, EvidenceRelation, SourceLevel, VerificationState
from .graph import EvidenceGraph


@dataclass(frozen=True)
class ProjectedCorpus:
    graph: EvidenceGraph
    exact_pdf_work_count: int
    ocr_derived_work_count: int


def project_curated_fixture(data: dict[str, Any]) -> ProjectedCorpus:
    evidence: list[EvidenceObject] = []
    for work in data["works"]:
        level = SourceLevel(work["source_level"])
        for raw in work.get("evidence", []):
            evidence.append(EvidenceObject(
                id=raw["evidence_id"],
                evidence_type=raw["type"],
                work_id=work["work_id"],
                normalized_form=raw["normalized_form"],
                verification_state=VerificationState.MACHINE_CHECKED,
                source_level=level,
                source_artifact_id=(raw.get("exact_pdf_locator") or {}).get("source_artifact_id"),
                attributes=raw.get("attributes", {}),
            ))

    # The first exact-PDF paper's full evidence fixture is loaded independently in
    # production. Tests may inject those objects into a combined fixture.
    for raw in data.get("injected_evidence", []):
        evidence.append(EvidenceObject.model_validate(raw))

    relations = [EvidenceRelation(
        id=r["relation_id"],
        source_id=r["source_id"],
        target_id=r["target_id"],
        relation_type=r["relation_type"],
        verification_state=r["verification_state"],
        rationale=r["rationale"],
        attributes={"context":"cross-work"},
    ) for r in data.get("cross_work_relations", []) if r["source_id"] in {e.id for e in evidence} and r["target_id"] in {e.id for e in evidence}]

    levels = [w["source_level"] for w in data["works"]]
    return ProjectedCorpus(
        graph=EvidenceGraph(evidence, relations),
        exact_pdf_work_count=sum(x == SourceLevel.FULLTEXT_PDF_LAYOUT for x in levels),
        ocr_derived_work_count=sum(x == SourceLevel.OCR_DERIVED  for x in levels),
     )
