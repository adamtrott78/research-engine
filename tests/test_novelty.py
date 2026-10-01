import pytest

from research_engine.novelty import CHALLENGE_RESULTS, GAP_STATUSES, NoveltyChallengeService


def test_novelty_state_sets_preserve_human_gate():
    assert "SUPPORTED_AS_OPEN" in GAP_STATUSES
    assert "SUPPORTED_AS_OPEN" in CHALLENGE_RESULTS
    assert "NO_COUNTEREVIDENCE_FOUND" not in CHALLENGE_RESULTS


def test_supported_as_open_requires_human_approval_before_database_access():
    class NoDB:
        def connect(self):
            raise AssertionError("database must not be touched")

    service = NoveltyChallengeService(NoDB())
    with pytest.raises(ValueError, match="explicit human approval"):
        service.record_challenge(
            challenge_id="c",
            gap_candidate_id="g",
            knowledge_checkpoint_id="k",
            proposed_result="SUPPORTED_AS_OPEN",
            search_provenance={},
            counterevidence=[],
            reasoning="nothing found",
        )
