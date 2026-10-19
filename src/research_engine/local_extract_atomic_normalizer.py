from __future__ import annotations

import hashlib
import json
import re
from typing import Any


MAX_CLAIMS_PER_SPAN = 4
MAX_NORMALIZED_CHARS = 360
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def claim_id(
    *,
    candidate_id: str,
    candidate_type: str,
    normalized_form: str,
) -> str:
    digest = hashlib.sha256(
        _canonical(
            {
                "candidate_id": candidate_id,
                "type": candidate_type,
                "normalized_form": normalized_form,
            }
        )
    ).hexdigest()[:16]
    return f"local-claim:{candidate_id}:{digest}"


def build_normalization_request(
    region: dict[str, Any],
    *,
    local_context: str = "",
    max_context_chars: int = 1200,
) -> dict[str, Any]:
    """Build a bounded tier-3 request from one already-located exact span."""
    if max_context_chars < 0:
        raise ValueError("max_context_chars must be nonnegative")

    candidate_id = str(region.get("region_id") or "").strip()
    raw = str(region.get("raw_source_form") or "")
    page = region.get("page_number")
    if not candidate_id:
        raise ValueError("region_id is required")
    if not raw.strip():
        raise ValueError("raw_source_form is required")
    if page is None:
        raise ValueError("page_number is required")

    context = str(local_context or "")
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    return {
        "schema_version": 1,
        "operation": "LOCAL_EXTRACT_ATOMIC_CLAIM_NORMALIZATION_V1",
        "candidate": {
            "candidate_id": candidate_id,
            "page_number": int(page),
            "raw_source_form": raw,
            "section_hint": str(region.get("section_hint") or ""),
        },
        "bounded_context": context,
        "output_contract": {
            "claims": [
                {
                    "candidate_id": candidate_id,
                    "type": "snake_case_candidate_type",
                    "normalized_form": "one atomic claim supported by the span",
                    "confidence": "number in [0,1]",
                }
            ]
        },
        "rules": [
            "Do not discover or quote new source text.",
            "Use only the supplied candidate span and bounded context.",
            "Every claim must reference the supplied candidate_id exactly.",
            "Return zero claims when the span does not support a useful atomic scientific claim.",
            "Split materially distinct facts into separate atomic claims.",
            "Do not emit raw_source_form, page_number, offsets, or source identifiers; deterministic code owns provenance.",
        ],
    }


def build_normalization_prompt(request: dict[str, Any]) -> str:
    """Render the narrow request as a deterministic JSON-first prompt."""
    candidate = dict(request["candidate"])
    contract = {
        "task": (
            "Normalize the supplied already-located source span into zero or "
            "more atomic scientific claims. You are not performing retrieval."
        ),
        "candidate": candidate,
        "bounded_context": request.get("bounded_context") or "",
        "rules": list(request["rules"]),
        "return_json_only": {
            "claims": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "type": "snake_case_candidate_type",
                    "normalized_form": "atomic supported claim",
                    "confidence": 0.0,
                }
            ]
        },
    }
    return json.dumps(
        contract,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def _normalize_type(value: Any) -> str:
    candidate_type = str(value or "").strip().lower().replace("-", "_")
    candidate_type = re.sub(r"\s+", "_", candidate_type)
    return candidate_type


def guard_and_canonicalize_normalization(
    region: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed and copy all provenance from deterministic region data."""
    candidate_id = str(region.get("region_id") or "").strip()
    raw = str(region.get("raw_source_form") or "")
    page = region.get("page_number")
    if not candidate_id or not raw.strip() or page is None:
        raise ValueError("invalid deterministic source region")

    reasons: list[str] = []
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return {
            "status": "FALLTHROUGH_CODEX",
            "reasons": ["CLAIMS_NOT_LIST"],
            "canonical_claims": None,
        }
    if len(claims) > MAX_CLAIMS_PER_SPAN:
        reasons.append(f"TOO_MANY_CLAIMS:{len(claims)}")

    canonical: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    for index, row in enumerate(claims[:MAX_CLAIMS_PER_SPAN]):
        if not isinstance(row, dict):
            reasons.append(f"CLAIM_{index}:NOT_OBJECT")
            continue

        referenced = str(row.get("candidate_id") or "").strip()
        if referenced != candidate_id:
            reasons.append(f"CLAIM_{index}:CANDIDATE_ID_MISMATCH")

        candidate_type = _normalize_type(row.get("type"))
        if not _TYPE_RE.fullmatch(candidate_type):
            reasons.append(f"CLAIM_{index}:INVALID_TYPE")

        normalized = re.sub(
            r"\s+",
            " ",
            str(row.get("normalized_form") or ""),
        ).strip()
        if not normalized:
            reasons.append(f"CLAIM_{index}:EMPTY_NORMALIZED_FORM")
        elif len(normalized) > MAX_NORMALIZED_CHARS:
            reasons.append(f"CLAIM_{index}:NORMALIZED_FORM_TOO_LONG")
        elif "\n" in normalized:
            reasons.append(f"CLAIM_{index}:MULTILINE_NORMALIZED_FORM")

        confidence = row.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            reasons.append(f"CLAIM_{index}:INVALID_CONFIDENCE")
            confidence = 0.0

        if (
            referenced != candidate_id
            or not _TYPE_RE.fullmatch(candidate_type)
            or not normalized
            or len(normalized) > MAX_NORMALIZED_CHARS
        ):
            continue

        key = (candidate_type, normalized)
        prior = seen.get(key)
        if prior is not None:
            prior["confidence"] = min(
                float(prior["confidence"]),
                float(confidence),
            )
            continue

        claim = {
            "id": claim_id(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                normalized_form=normalized,
            ),
            "candidate_id": candidate_id,
            "type": candidate_type,
            "normalized_form": normalized,
            "raw_source_form": raw,
            "page_number": int(page),
            "confidence": float(confidence),
            "source_layer": str(
                region.get("source_layer") or "AUTHORITATIVE_PDF_TEXT"
            ),
            "source_region": {
                "start_char": region.get("start_char"),
                "end_char": region.get("end_char"),
                "section_hint": str(region.get("section_hint") or ""),
            },
        }
        seen[key] = claim
        canonical.append(claim)

    return {
        "status": "PASS_GROUNDED" if not reasons else "FALLTHROUGH_CODEX",
        "reasons": reasons,
        "canonical_claims": canonical if not reasons else None,
        "candidate_id": candidate_id,
        "model_provenance_fields_accepted": False,
        "scientific_state_materialized": False,
    }
