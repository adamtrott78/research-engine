from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable


_ALLOWED_LAYERS = {"EPISTEMIC", "NORMATIVE", "EXECUTION"}
_ALLOWED_ASSERTION_STATES = {"ACTIVE", "SUPERSEDED", "REJECTED", "UNKNOWN"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class WorkspaceSemanticTwin:
    """Read-only deterministic semantic projection for prototype benchmarks."""

    def __init__(self, observations: Iterable[dict[str, Any]]):
        records = [deepcopy(record) for record in observations]
        by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            self._validate_observation(record)
            observation_id = record["observation_id"]
            if observation_id in by_id:
                raise ValueError(f"duplicate observation_id: {observation_id}")
            by_id[observation_id] = record

        self._by_id = by_id
        self._ordered_ids = tuple(sorted(by_id))
        self._manifest_body = {
            "schema_version": 1,
            "observations": [deepcopy(by_id[key]) for key in self._ordered_ids],
        }
        self._manifest_sha256 = _sha256_hex(self._manifest_body)

    @staticmethod
    def _validate_observation(record: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "observation_id",
            "layer",
            "subject",
            "predicate",
            "object",
            "assertion_state",
            "provenance",
            "verification",
            "temporal",
            "supersedes",
            "contradicts",
            "attributes",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"missing observation fields: {sorted(missing)}")
        if record["layer"] not in _ALLOWED_LAYERS:
            raise ValueError(f'invalid layer: {record["layer"]}')
        if record["assertion_state"] not in _ALLOWED_ASSERTION_STATES:
            raise ValueError(f'invalid assertion_state: {record["assertion_state"]}')
        if not record["provenance"]:
            raise ValueError(f'{record["observation_id"]} has no provenance')

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    def observation_ids(self) -> tuple[str, ...]:
        return self._ordered_ids

    def get(self, observation_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self._by_id[observation_id])
        except KeyError as exc:
            raise KeyError(f"unknown observation_id: {observation_id}") from exc

    def all_observations(self) -> list[dict[str, Any]]:
        return [self.get(observation_id) for observation_id in self._ordered_ids]

    def query(
        self,
        *,
        layer: str | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        predicate: str | None = None,
        assertion_state: str | None = None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for observation_id in self._ordered_ids:
            record = self._by_id[observation_id]
            subject = record["subject"]
            if layer is not None and record["layer"] != layer:
                continue
            if subject_type is not None and subject["entity_type"] != subject_type:
                continue
            if subject_id is not None and subject["entity_id"] != subject_id:
                continue
            if predicate is not None and record["predicate"] != predicate:
                continue
            if assertion_state is not None and record["assertion_state"] != assertion_state:
                continue
            matches.append(deepcopy(record))
        return matches

    def trace_provenance(self, observation_id: str) -> list[Any]:
        return deepcopy(self._by_id[observation_id]["provenance"])

    def contradictions(self, observation_id: str) -> list[dict[str, Any]]:
        record = self._by_id[observation_id]
        result: list[dict[str, Any]] = []
        for other_id in sorted(record["contradicts"]):
            if other_id not in self._by_id:
                raise ValueError(
                    f"{observation_id} references missing contradiction {other_id}"
                )
            result.append(deepcopy(self._by_id[other_id]))
        return result

    def supersession_history(
        self,
        *,
        subject_type: str,
        subject_id: str,
        predicate: str,
    ) -> list[dict[str, Any]]:
        records = self.query(
            subject_type=subject_type,
            subject_id=subject_id,
            predicate=predicate,
        )
        return sorted(
            records,
            key=lambda record: (
                _parse_time(record["temporal"].get("valid_from"))
                or _parse_time(record["temporal"].get("observed_at"))
                or datetime.min.replace(tzinfo=timezone.utc),
                record["observation_id"],
            ),
        )

    def replay_as_of(
        self,
        *,
        subject_type: str,
        subject_id: str,
        predicate: str,
        as_of: str,
    ) -> list[dict[str, Any]]:
        instant = _parse_time(as_of)
        assert instant is not None
        result: list[dict[str, Any]] = []
        for record in self.supersession_history(
            subject_type=subject_type,
            subject_id=subject_id,
            predicate=predicate,
        ):
            temporal = record["temporal"]
            valid_from = _parse_time(temporal.get("valid_from"))
            valid_to = _parse_time(temporal.get("valid_to"))
            if valid_from is not None and instant < valid_from:
                continue
            if valid_to is not None and instant >= valid_to:
                continue
            if valid_from is None and valid_to is None:
                observed_at = _parse_time(temporal.get("observed_at"))
                if observed_at is not None and instant < observed_at:
                    continue
            result.append(deepcopy(record))
        return result

    def query_task_scientific_state(
        self,
        *,
        task_id: str,
        experiment_id: str,
    ) -> dict[str, Any]:
        task_status = self.query(
            layer="EXECUTION",
            subject_type="Task",
            subject_id=task_id,
            predicate="hasStatus",
        )
        experiment = self.query(
            layer="EPISTEMIC",
            subject_type="ScientificExperiment",
            subject_id=experiment_id,
            predicate="hasScientificDisposition",
        )
        if len(task_status) != 1:
            raise ValueError(f"expected one task status, found {len(task_status)}")
        if len(experiment) != 1:
            raise ValueError(
                f"expected one scientific disposition, found {len(experiment)}"
            )
        experiment_record = experiment[0]
        return {
            "task_status": task_status[0]["object"]["value"],
            "scientific_disposition": experiment_record["object"]["value"],
            "wave7b_authorized": experiment_record["attributes"]["wave7b_authorized"],
            "evidence_observation_ids": [
                task_status[0]["observation_id"],
                experiment_record["observation_id"],
            ],
        }
