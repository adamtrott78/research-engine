from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any


_SOURCE_LEVEL_RANK={
    "METADATA":0,
    "ABSTRACT":1,
    "FULLTEXT":2,
    "AUTHORITATIVE_PRIMARY":3,
}

_REASON_TO_KIND={
    "MISSING_FACET":"MISSING_FACET_EVIDENCE",
    "CONTRADICTION":"INDEPENDENT_EVALUATION",
    "SOURCE_LEVEL_INSUFFICIENT":"HIGHER_SOURCE_LEVEL",
    "VERIFICATION_INSUFFICIENT":"VERIFICATION_EVIDENCE",
    "STALE":"FRESH_EVIDENCE",
    "BENCHMARK_CALIBRATED_INSUFFICIENCY":"BENCHMARK_DISCRIMINATING_EVIDENCE",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None:
        parsed=parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_refs(observations: list[dict[str, Any]]) -> list[Any]:
    refs=[]
    for observation in observations:
        refs.extend(deepcopy(observation.get("provenance",[])))
    keyed={_canonical_json(ref):ref for ref in refs}
    return [keyed[key] for key in sorted(keyed)]


def select_bounded_ontology_slice(
    *,
    observations: list[dict[str, Any]],
    case: dict[str, Any],
    semantic_manifest_id: str,
) -> dict[str, Any]:
    required_types=set(case["required_entity_types"])
    required_predicates=set(case["required_predicates"])
    forbidden_types=set(case.get("forbidden_entity_types",[]))
    maximum=case["max_observations"]

    scored=[]
    for observation in observations:
        subject_type=observation["subject"]["entity_type"]
        object_type=observation.get("object",{}).get("entity_type")
        predicate=observation["predicate"]

        if subject_type in forbidden_types or object_type in forbidden_types:
            continue

        type_match=(subject_type in required_types) or (object_type in required_types)
        predicate_match=predicate in required_predicates
        if not type_match and not predicate_match:
            continue

        score=(
            0 if (type_match and predicate_match) else
            1 if predicate_match else
            2
        )
        scored.append((score,observation["observation_id"],deepcopy(observation)))

    scored.sort(key=lambda item:(item[0],item[1]))
    selected=[item[2] for item in scored[:maximum]]

    included_types=sorted({
        entity_type
        for observation in selected
        for entity_type in [
            observation["subject"]["entity_type"],
            observation.get("object",{}).get("entity_type"),
        ]
        if entity_type is not None and entity_type in required_types
    })
    included_predicates=sorted({
        observation["predicate"]
        for observation in selected
        if observation["predicate"] in required_predicates
    })

    body={
        "schema_version":1,
        "slice_id":f'slice:{case["case_id"].lower()}',
        "semantic_manifest_id":semantic_manifest_id,
        "context":deepcopy(case["context"]),
        "included_entity_types":included_types,
        "included_predicates":included_predicates,
        "constraints":[
            f"max_observations={maximum}",
            "authority_widening_allowed=false",
            "provenance_required=true",
        ],
        "source_schema_refs":_source_refs(selected),
        "deterministic_key":_sha256({
            "case_id":case["case_id"],
            "semantic_manifest_id":semantic_manifest_id,
            "observation_ids":[item["observation_id"] for item in selected],
        }),
        "included_observation_ids":[item["observation_id"] for item in selected],
        "observations":selected,
        "authority_widening_allowed":False,
    }
    return body


def derive_evidence_requirement(
    *,
    target_id: str,
    trigger: dict[str, Any],
    provenance: list[Any],
) -> dict[str, Any] | None:
    reasons=[]

    if trigger.get("contradictions"):
        reasons.append("CONTRADICTION")
    if trigger.get("missing_facets"):
        reasons.append("MISSING_FACET")

    source_level=trigger.get("source_level")
    required_source_level=trigger.get("required_source_level")
    if source_level is not None and required_source_level is not None:
        if source_level not in _SOURCE_LEVEL_RANK:
            raise ValueError(f"unknown source level: {source_level}")
        if required_source_level not in _SOURCE_LEVEL_RANK:
            raise ValueError(f"unknown required source level: {required_source_level}")
        if _SOURCE_LEVEL_RANK[source_level] < _SOURCE_LEVEL_RANK[required_source_level]:
            reasons.append("SOURCE_LEVEL_INSUFFICIENT")

    if trigger.get("verification_state")=="UNVERIFIED":
        reasons.append("VERIFICATION_INSUFFICIENT")

    valid_to=_parse_time(trigger.get("valid_to"))
    as_of=_parse_time(trigger.get("as_of"))
    if valid_to is not None and as_of is not None and as_of>=valid_to:
        reasons.append("STALE")

    if trigger.get("benchmark_sufficient") is False:
        reasons.append("BENCHMARK_CALIBRATED_INSUFFICIENCY")

    # model_confidence is deliberately ignored as a standalone trigger.
    reasons=sorted(set(reasons))
    if not reasons:
        return None

    body={
        "schema_version":1,
        "target_ids":[target_id],
        "reason_types":reasons,
        "requested_evidence_kinds":sorted({_REASON_TO_KIND[reason] for reason in reasons}),
        "evidence_basis":[target_id],
        "priority":"HIGH" if {"CONTRADICTION","BENCHMARK_CALIBRATED_INSUFFICIENCY"} & set(reasons) else "MEDIUM",
        "status":"OPEN",
        "execution_effect":"REQUEST_ONLY",
        "provenance":deepcopy(provenance),
    }
    return {
        "requirement_id":"hardening:req:"+_sha256(body)[:20],
        **body,
    }


def derive_frozen_evidence_case(case: dict[str, Any]) -> dict[str, Any] | None:
    return derive_evidence_requirement(
        target_id=f'target:{case["case_id"]}',
        trigger=case["trigger"],
        provenance=[{
            "source_kind":"SYNTHETIC_HARDENING_FIXTURE",
            "case_id":case["case_id"],
        }],
    )
