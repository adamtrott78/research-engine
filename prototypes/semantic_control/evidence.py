from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable


_SOURCE_LEVEL_RANK = {
    "METADATA": 0,
    "ABSTRACT": 1,
    "FULLTEXT": 2,
    "AUTHORITATIVE_PRIMARY": 3,
}

_SLICE_PROFILES = {
    "scientific_evidence_gap": {
        "slice_id": "slice:scientific-gap-v1",
        "included_entity_types": ["Claim", "EvidenceRequirement", "Method"],
        "included_predicates": [
            "contradicts",
            "improvesHardwareShiftRobustness",
            "requiresEvidence",
        ],
        "constraints": [
            "preserve provenance",
            "unknown is not false",
            "do not collapse contradictions",
        ],
    }
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dedupe(values: Iterable[Any]) -> list[Any]:
    by_key = {_canonical_json(value): value for value in values}
    return [by_key[key] for key in sorted(by_key)]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_ontology_slice(
    *,
    profile_name: str,
    semantic_manifest_id: str,
    source_schema_refs: list[Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        profile = _SLICE_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown ontology slice profile: {profile_name}") from exc

    entity_types = sorted(profile["included_entity_types"])
    predicates = sorted(profile["included_predicates"])
    deterministic_key = (
        "scientific-gap|"
        + ",".join(entity_types)
        + "|"
        + ",".join(predicates)
    )
    return {
        "schema_version": 1,
        "slice_id": profile["slice_id"],
        "semantic_manifest_id": semantic_manifest_id,
        "context": dict(context or {}),
        "included_entity_types": entity_types,
        "included_predicates": predicates,
        "constraints": list(profile["constraints"]),
        "source_schema_refs": _dedupe(source_schema_refs),
        "deterministic_key": deterministic_key,
    }


def _requirement_id(
    *,
    target_ids: list[str],
    reason_types: list[str],
    observations: dict[str, dict[str, Any]],
) -> str:
    if reason_types == ["CONTRADICTION"] and len(target_ids) == 2:
        left = observations[target_ids[0]]
        right = observations[target_ids[1]]
        if left["subject"] == right["subject"]:
            return (
                "req:resolve-"
                + _slug(left["subject"]["entity_id"])
                + "-contradiction"
            )
    return "req:auto:" + _stable_hash(
        {"target_ids": target_ids, "reason_types": reason_types}
    )[:16]


def _provenance_for(
    target_ids: list[str],
    observations: dict[str, dict[str, Any]],
) -> list[Any]:
    values: list[Any] = []
    for target_id in target_ids:
        values.extend(observations[target_id].get("provenance", []))
    return _dedupe(values)


def _requested_kinds(reason_types: list[str]) -> list[str]:
    mapping = {
        "MISSING_FACET": "MISSING_FACET_EVIDENCE",
        "CONTRADICTION": "INDEPENDENT_EVALUATION",
        "SOURCE_LEVEL_INSUFFICIENT": "HIGHER_SOURCE_LEVEL",
        "VERIFICATION_INSUFFICIENT": "VERIFICATION_EVIDENCE",
        "STALE": "FRESH_EVIDENCE",
        "BENCHMARK_CALIBRATED_INSUFFICIENCY": "BENCHMARK_DISCRIMINATING_EVIDENCE",
    }
    return sorted({mapping[reason] for reason in reason_types})


def _make_requirement(
    *,
    target_ids: list[str],
    reason_types: list[str],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_ids = sorted(target_ids)
    reason_types = sorted(set(reason_types))
    priority = (
        "HIGH"
        if {
            "CONTRADICTION",
            "BENCHMARK_CALIBRATED_INSUFFICIENCY",
        }
        & set(reason_types)
        else "MEDIUM"
    )
    return {
        "schema_version": 1,
        "requirement_id": _requirement_id(
            target_ids=target_ids,
            reason_types=reason_types,
            observations=observations,
        ),
        "target_ids": target_ids,
        "reason_types": reason_types,
        "requested_evidence_kinds": _requested_kinds(reason_types),
        "evidence_basis": target_ids,
        "priority": priority,
        "status": "OPEN",
        "execution_effect": "REQUEST_ONLY",
        "provenance": _provenance_for(target_ids, observations),
    }


def _source_level_insufficient(attributes: dict[str, Any]) -> bool:
    source_level = attributes.get("source_level")
    required = attributes.get("required_source_level")
    if source_level is None or required is None:
        return False
    try:
        return _SOURCE_LEVEL_RANK[source_level] < _SOURCE_LEVEL_RANK[required]
    except KeyError as exc:
        raise ValueError(f"unknown source level: {exc.args[0]}") from exc


def derive_evidence_requirements(
    observations: Iterable[dict[str, Any]],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    records = {record["observation_id"]: record for record in observations}
    if len(records) != len(list(observations)) if not isinstance(observations, list) else False:
        raise ValueError("duplicate observation_id")

    # Re-materialize safely for arbitrary iterables.
    records_list = list(records.values())
    records = {record["observation_id"]: record for record in records_list}
    requirements: list[dict[str, Any]] = []

    seen_pairs: set[tuple[str, str]] = set()
    for observation_id in sorted(records):
        record = records[observation_id]
        if record.get("assertion_state") != "ACTIVE":
            continue
        for other_id in sorted(record.get("contradicts", [])):
            if other_id not in records:
                raise ValueError(
                    f"{observation_id} references missing contradiction {other_id}"
                )
            if records[other_id].get("assertion_state") != "ACTIVE":
                continue
            pair = tuple(sorted((observation_id, other_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            requirements.append(
                _make_requirement(
                    target_ids=list(pair),
                    reason_types=["CONTRADICTION"],
                    observations=records,
                )
            )

    instant = _parse_time(as_of)
    for observation_id in sorted(records):
        record = records[observation_id]
        if record.get("assertion_state") != "ACTIVE":
            continue
        attributes = record.get("attributes", {})
        reasons: list[str] = []

        if attributes.get("missing_facets"):
            reasons.append("MISSING_FACET")
        if _source_level_insufficient(attributes):
            reasons.append("SOURCE_LEVEL_INSUFFICIENT")
        if record.get("verification", {}).get("state") == "UNVERIFIED":
            reasons.append("VERIFICATION_INSUFFICIENT")
        if attributes.get("benchmark_sufficient") is False:
            reasons.append("BENCHMARK_CALIBRATED_INSUFFICIENCY")

        if instant is not None:
            valid_to = _parse_time(record.get("temporal", {}).get("valid_to"))
            if valid_to is not None and instant >= valid_to:
                reasons.append("STALE")

        # model_confidence is intentionally ignored as a standalone signal.
        if reasons:
            requirements.append(
                _make_requirement(
                    target_ids=[observation_id],
                    reason_types=reasons,
                    observations=records,
                )
            )

    return sorted(requirements, key=lambda item: item["requirement_id"])
