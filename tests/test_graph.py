import pytest

from research_engine.domain import EvidenceObject, EvidenceRelation, SourceLevel, VerificationState
from research_engine.graph import EvidenceGraph


def obj(i, t):
    return EvidenceObject(id=i,evidence_type=t,work_id="w",normalized_form=i,verification_state=VerificationState.MACHINE_CHECKED,source_level=SourceLevel.FULLTEXT_PDF_LAYOUT)


def test_result_supports_claim_direction_and_query():
    result=obj("result","RESULT")
    claim=obj("claim","CLAIM")
    rel=EvidenceRelation(id="r",source_id="result",target_id="claim",relation_type="SUPPORTS",verification_state="MACHINE_CHECKED",rationale="reported result supports conclusion")
    graph=EvidenceGraph([result,claim],[rel])
    assert [x.id for x in graph.supporting_results("claim")] == ["result"]


def test_reversed_support_edge_is_rejected():
    result=obj("result","RESULT")
    claim=obj("claim","CLAIM")
    rel=EvidenceRelation(id="r",source_id="claim",target_id="result",relation_type="SUPPORTS",verification_state="MACHINE_CHECKED",rationale="wrong")
    with pytest.raises(ValueError, match="SUPPORTS"):
        EvidenceGraph([result,claim],[rel])
