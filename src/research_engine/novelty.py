from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .corpus_loader import ImmutableRecordConflict


GAP_STATUSES = {
    "PROPOSED",
    "UNDER_CHALLENGE",
    "NARROWED",
    "REJECTED_PRIOR_WORK_FOUND",
    "NEEDS_HUMAN_REVIEW",
    "SUPPORTED_AS_OPEN",
}

CHALLENGE_RESULTS = {
    "NARROWED",
    "REJECTED_PRIOR_WORK_FOUND",
    "NEEDS_HUMAN_REVIEW",
    "SUPPORTED_AS_OPEN",
}


@dataclass(frozen=True)
class ChallengeOutcome:
    challenge_id: str
    gap_candidate_id: str
    result: str
    knowledge_checkpoint_id: str


class NoveltyChallengeService:
    """Persist gap hypotheses and adversarial novelty challenges.

    Scientific novelty is not inferred from an empty search result. Automated
    challenges can reject, narrow, or escalate to human review. Only an explicit
    human approval record can produce SUPPORTED_AS_OPEN.
    """

    def __init__(self, store):
        self.store = store

    @staticmethod
    def _checkpoint_exists(cur, checkpoint_id: str) -> None:
        cur.execute("SELECT 1 FROM knowledge_checkpoints WHERE id = %s", (checkpoint_id,))
        if cur.fetchone() is None:
            raise ValueError(f"knowledge checkpoint does not exist: {checkpoint_id}")

    def put_gap_candidate(
        self,
        *,
        candidate_id: str,
        statement: str,
        gap_type: str,
        knowledge_checkpoint_id: str,
        motivating_evidence_ids: list[str],
        rationale: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        from psycopg.types.json import Jsonb

        incoming = (
            statement,
            gap_type,
            knowledge_checkpoint_id,
            motivating_evidence_ids,
            "PROPOSED",
            rationale,
            "NOT_REVIEWED",
            attributes or {},
        )
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                self._checkpoint_exists(cur, knowledge_checkpoint_id)

                if motivating_evidence_ids:
                    cur.execute(
                        "SELECT id FROM evidence_objects WHERE id = ANY(%s)",
                        (motivating_evidence_ids,),
                    )
                    found = {row[0] for row in cur.fetchall()}
                    missing = sorted(set(motivating_evidence_ids) - found)
                    if missing:
                        raise ValueError(f"motivating evidence does not exist: {missing}")

                cur.execute(
                    """
                    SELECT statement, gap_type, knowledge_checkpoint_id,
                           motivating_evidence_ids, novelty_status, rationale,
                           human_review_state, attributes
                    FROM gap_candidates WHERE id = %s
                    """,
                    (candidate_id,),
                )
                existing = cur.fetchone()
                if existing is None:
                    cur.execute(
                        """
                        INSERT INTO gap_candidates(
                          id, statement, gap_type, knowledge_checkpoint_id,
                          motivating_evidence_ids, novelty_status, rationale,
                          human_review_state, attributes
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            candidate_id,
                            statement,
                            gap_type,
                            knowledge_checkpoint_id,
                            motivating_evidence_ids,
                            "PROPOSED",
                            rationale,
                            "NOT_REVIEWED",
                            Jsonb(attributes or {}),
                        ),
                    )
                else:
                    # novelty_status and human_review_state are state fields updated
                    # by challenges; the scientific identity fields must remain fixed.
                    existing_identity = (
                        existing[0],
                        existing[1],
                        existing[2],
                        list(existing[3]),
                        existing[5],
                        existing[7],
                    )
                    incoming_identity = (
                        statement,
                        gap_type,
                        knowledge_checkpoint_id,
                        motivating_evidence_ids,
                        rationale,
                        attributes or {},
                    )
                    if existing_identity != incoming_identity:
                        raise ImmutableRecordConflict(
                            f"gap candidate id {candidate_id} already exists with different scientific content"
                        )
            conn.commit()

    def record_challenge(
        self,
        *,
        challenge_id: str,
        gap_candidate_id: str,
        knowledge_checkpoint_id: str,
        proposed_result: str,
        search_provenance: dict[str, Any],
        counterevidence: list[dict[str, Any]],
        reasoning: str,
        human_approval: dict[str, Any] | None = None,
    ) -> ChallengeOutcome:
        from psycopg.types.json import Jsonb

        result = (
            "NEEDS_HUMAN_REVIEW"
            if proposed_result == "NO_COUNTEREVIDENCE_FOUND"
            else proposed_result
        )
        if result not in CHALLENGE_RESULTS:
            raise ValueError(f"unsupported novelty challenge result: {proposed_result}")

        if result == "SUPPORTED_AS_OPEN":
            if not human_approval or human_approval.get("approved") is not True:
                raise ValueError(
                    "SUPPORTED_AS_OPEN requires explicit human approval; "
                    "an automated empty search is insufficient"
                )
            for key in ("reviewer", "decision_at", "rationale"):
                if not human_approval.get(key):
                    raise ValueError(f"human approval missing required field: {key}")

        requested = (
            gap_candidate_id,
            knowledge_checkpoint_id,
            result,
            search_provenance,
            counterevidence,
            reasoning,
            human_approval,
        )

        with self.store.connect() as conn:
            with conn.cursor() as cur:
                self._checkpoint_exists(cur, knowledge_checkpoint_id)
                cur.execute(
                    "SELECT novelty_status FROM gap_candidates WHERE id = %s",
                    (gap_candidate_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"gap candidate does not exist: {gap_candidate_id}")

                cur.execute(
                    """
                    SELECT gap_candidate_id, knowledge_checkpoint_id, result,
                           search_provenance, counterevidence, reasoning, human_approval
                    FROM novelty_challenges WHERE id = %s
                    """,
                    (challenge_id,),
                )
                existing = cur.fetchone()
                if existing is None:
                    cur.execute(
                        """
                        INSERT INTO novelty_challenges(
                          id, gap_candidate_id, knowledge_checkpoint_id, result,
                          search_provenance, counterevidence, reasoning, human_approval
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            challenge_id,
                            gap_candidate_id,
                            knowledge_checkpoint_id,
                            result,
                            Jsonb(search_provenance),
                            Jsonb(counterevidence),
                            reasoning,
                            Jsonb(human_approval) if human_approval is not None else None,
                        ),
                    )
                else:
                    if tuple(existing) != requested:
                        raise ImmutableRecordConflict(
                            f"novelty challenge id {challenge_id} already exists with different content"
                        )

                if result == "SUPPORTED_AS_OPEN":
                    review_state = "HUMAN_APPROVED_OPEN"
                elif result == "NEEDS_HUMAN_REVIEW":
                    review_state = "REQUIRED"
                else:
                    review_state = "NOT_REQUIRED"

                cur.execute(
                    """
                    UPDATE gap_candidates
                    SET novelty_status = %s, human_review_state = %s
                    WHERE id = %s
                    """,
                    (result, review_state, gap_candidate_id),
                )
            conn.commit()

        return ChallengeOutcome(
            challenge_id=challenge_id,
            gap_candidate_id=gap_candidate_id,
            result=result,
            knowledge_checkpoint_id=knowledge_checkpoint_id,
        )

    def challenge_history(self, gap_candidate_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, knowledge_checkpoint_id, result, search_provenance,
                           counterevidence, reasoning, human_approval, created_at
                    FROM novelty_challenges
                    WHERE gap_candidate_id = %s
                    ORDER BY created_at, id
                    """,
                    (gap_candidate_id,),
                )
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
