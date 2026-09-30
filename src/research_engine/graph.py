from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .domain import EvidenceObject, EvidenceRelation


@dataclass(frozen=True)
class PathStep:
    source_id: str
    relation_type: str
    target_id: str


class EvidenceGraph:
    """In-memory semantic reference implementation for graph invariants/tests."""

    def __init__(self, evidence: list[EvidenceObject], relations: list[EvidenceRelation]):
        self.evidence = {e.id: e for e in evidence}
        self.relations = {r.id: r for r in relations}
        for r in relations:
            if r.source_id not in self.evidence or r.target_id not in self.evidence:
                raise ValueError(f"relation {r.id} references missing evidence")
            r.validate_direction(self.evidence)
        self.outgoing: dict[str, list[EvidenceRelation]] = defaultdict(list)
        self.incoming: dict[str, list[EvidenceRelation]] = defaultdict(list)
        for r in relations:
            self.outgoing[r.source_id].append(r)
            self.incoming[r.target_id].append(r)

    def supporting_results(self, claim_id: str) -> list[EvidenceObject]:
        return [self.evidence[r.source_id] for r in self.incoming[claim_id] if r.relation_type == "SUPPORTS"]

    def qualifiers_for_method(self, method_id: str) -> list[EvidenceObject]:
        return [self.evidence[r.source_id] for r in self.incoming[method_id] if r.relation_type == "QUALIFIES"]

    def shortest_path(self, start: str, goal: str, max_depth: int = 8) -> list[PathStep] | None:
        q = deque([(start, [])])
        seen = {start}
        while q:
            node, path = q.popleft()
            if len(path) >= max_depth:
                continue
            for r in self.outgoing[node]:
                step = PathStep(r.source_id, r.relation_type, r.target_id)
                if r.target_id == goal:
                    return path + [step]
                if r.target_id not in seen:
                    seen.add(r.target_id)
                    q.append((r.target_id, path + [step]))
        return None
