from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


class VerificationState(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    MACHINE_CHECKED = "MACHINE_CHECKED"
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class SourceLevel(StrEnum):
    METADATA_ONLY = "METADATA_ONLY"
    ABSTRACT_ONLY = "ABSTRACT_ONLY"
    FULLTEXT_STRUCTURED = "FULLTEXT_STRUCTURED"
    FULLTEXT_PDF_TEXT = "FULLTEXT_PDF_TEXT"
    FULLTEXT_PDF_LAYOUT = "FULLTEXT_PDF_LAYOUT"
    OCR_DERIVED = "OCR_DERIVED"


class EvidenceObject(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    evidence_type: str
    work_id: str
    normalized_form: str
    verification_state: VerificationState
    source_level: SourceLevel
    source_artifact_id: str | None = None
    source_locator_id: str | None = None
    attributes: dict = Field(default_factory=dict)


class EvidenceRelation(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source_id: str
    target_id: str
    relation_type: str
    verification_state: str
    rationale: str
    attributes: dict = Field(default_factory=dict)

    def validate_direction(self, evidence: dict[str, EvidenceObject]) -> None:
        source = evidence[self.source_id]
        target = evidence[self.target_id]
        if self.relation_type == "SUPPORTS" and source.evidence_type != "RESULT":
            raise ValueError("SUPPORTS must originate from RESULT in v1")
        if self.relation_type == "SUPPORTS" and target.evidence_type != "CLAIM":
            raise ValueError("SUPPORTS must target CLAIM in v1")
        if self.relation_type == "QUALIFIES" and source.evidence_type not in {"LIMITATION", "ASSUMPTION", "THREAT_TO_VALIDITY"}:
            raise ValueError("QUALIFIES must originate from a qualifier evidence type")
