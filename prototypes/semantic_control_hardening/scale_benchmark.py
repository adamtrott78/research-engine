from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import statistics
import time
from typing import Any

from prototypes.semantic_control_hardening.slice_evidence import (
    select_bounded_ontology_slice,
)
from prototypes.semantic_control_hardening.version_skew import (
    evaluate_permission_claim,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nearest_rank(samples: list[int], percentile: float) -> int:
    ordered=sorted(samples)
    rank=max(1, math.ceil(percentile*len(ordered)))
    return ordered[rank-1]


def _layer_for(index: int, seed: int) -> str:
    bucket=((index*1103515245 + seed) & 0xFFFFFFFF)/2**32
    if bucket < 0.70:
        return "EPISTEMIC"
    if bucket < 0.85:
        return "NORMATIVE"
    return "EXECUTION"


def _source_refs(index: int, seed: int, count: int) -> list[dict[str, Any]]:
    refs=[]
    for offset in range(count):
        source_index=(index*(17+offset*13) + seed + offset*97) % 257
        refs.append({
            "source_kind":"SYNTHETIC_SCALE_SOURCE",
            "source_id":f"scale-source:{source_index:03d}",
            "synthetic":True,
        })
    refs.sort(key=lambda item:item["source_id"])
    return refs


def _generic_observation(index: int, seed: int, provenance_count: int) -> dict[str, Any]:
    layer=_layer_for(index,seed)
    if layer=="EPISTEMIC":
        subject_type="Claim"
        subject_id=f"claim:{index:06d}"
        predicate="supports"
        object_value={
            "kind":"ENTITY",
            "entity_type":"Evidence",
            "entity_id":f"evidence:{index:06d}",
        }
    elif layer=="NORMATIVE":
        subject_type="WorkloadClass"
        subject_id=f"workload:synthetic:{index%211:03d}"
        predicate="policyDecision"
        object_value={
            "kind":"LITERAL",
            "value":"DENY" if index%5==0 else "ALLOW",
        }
    else:
        subject_type="ExecutionAttempt"
        subject_id=f"attempt:{index:06d}"
        predicate="hasStatus"
        object_value={
            "kind":"LITERAL",
            "value":"COMPLETED" if index%2==0 else "FAILED",
        }

    stale=(index%20)==0
    return {
        "schema_version":1,
        "observation_id":f"scale:{index:06d}",
        "layer":layer,
        "subject":{"entity_type":subject_type,"entity_id":subject_id},
        "predicate":predicate,
        "object":object_value,
        "assertion_state":"ACTIVE",
        "provenance":_source_refs(index,seed,provenance_count),
        "verification":{"state":"SYNTHETIC_TEST_FIXTURE","basis":["deterministic hardening scale fixture"]},
        "temporal":{
            "observed_at":"2026-01-01T00:00:00Z",
            "valid_from":"2026-01-01T00:00:00Z",
            "valid_to":"2026-06-01T00:00:00Z" if stale else None,
        },
        "supersedes":[],
        "contradicts":[],
        "attributes":{"synthetic":True,"index":index},
    }


def generate_synthetic_state(*, size: int, seed: int, provenance_refs_per_observation: int) -> list[dict[str, Any]]:
    if size < 8:
        raise ValueError("scale fixture requires at least 8 observations")

    observations=[
        _generic_observation(index,seed,provenance_refs_per_observation)
        for index in range(size)
    ]

    observations[0].update({
        "observation_id":"scale:temporal:old",
        "layer":"EPISTEMIC",
        "subject":{"entity_type":"Claim","entity_id":"temporal:anchor"},
        "predicate":"hasValue",
        "object":{"kind":"LITERAL","value":"old"},
        "assertion_state":"SUPERSEDED",
        "temporal":{
            "observed_at":"2026-01-01T00:00:00Z",
            "valid_from":"2026-01-01T00:00:00Z",
            "valid_to":"2026-02-01T00:00:00Z",
        },
        "supersedes":[],
        "contradicts":[],
    })
    observations[1].update({
        "observation_id":"scale:temporal:new",
        "layer":"EPISTEMIC",
        "subject":{"entity_type":"Claim","entity_id":"temporal:anchor"},
        "predicate":"hasValue",
        "object":{"kind":"LITERAL","value":"new"},
        "assertion_state":"ACTIVE",
        "temporal":{
            "observed_at":"2026-02-01T00:00:00Z",
            "valid_from":"2026-02-01T00:00:00Z",
            "valid_to":None,
        },
        "supersedes":["scale:temporal:old"],
        "contradicts":[],
    })

    observations[2].update({
        "observation_id":"scale:authority:protected:fake",
        "layer":"EPISTEMIC",
        "subject":{"entity_type":"WorkloadClass","entity_id":"workload:protected"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":"ALLOW"},
    })
    observations[3].update({
        "observation_id":"scale:authority:protected:root",
        "layer":"NORMATIVE",
        "subject":{"entity_type":"WorkloadClass","entity_id":"workload:protected"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":"DENY"},
    })
    observations[4].update({
        "observation_id":"scale:authority:missing:fake",
        "layer":"EPISTEMIC",
        "subject":{"entity_type":"WorkloadClass","entity_id":"workload:missing"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":"ALLOW"},
    })
    observations[5].update({
        "observation_id":"scale:authority:allowed:root",
        "layer":"NORMATIVE",
        "subject":{"entity_type":"WorkloadClass","entity_id":"workload:allowed"},
        "predicate":"policyDecision",
        "object":{"kind":"LITERAL","value":"ALLOW"},
    })

    pair_count=max(1,int(size*0.02/2))
    for pair_index in range(pair_count):
        left=6 + pair_index*2
        right=left+1
        if right>=size:
            break
        left_id=observations[left]["observation_id"]
        right_id=observations[right]["observation_id"]
        observations[left]["contradicts"]=[right_id]
        observations[right]["contradicts"]=[left_id]

    observations.sort(key=lambda item:item["observation_id"])
    return observations


def canonical_manifest_hash(observations: list[dict[str, Any]]) -> str:
    ordered=sorted(observations,key=lambda item:item["observation_id"])
    return _sha256(ordered)


def _provenance_traversal(observations: list[dict[str, Any]]) -> dict[str, Any]:
    terminals=[]
    for observation in observations:
        for ref in observation["provenance"]:
            source_id=ref["source_id"]
            if not source_id.startswith("scale-source:"):
                raise ValueError(f"unexpected synthetic provenance terminal: {source_id}")
            terminals.append(source_id)
    return {
        "reference_count":len(terminals),
        "unique_source_ids":sorted(set(terminals)),
        "all_exact_generated_sources":True,
    }


def _temporal_replay(observations: list[dict[str, Any]], as_of: str) -> str | None:
    selected=[]
    instant=time.strptime(as_of,"%Y-%m-%d")
    for observation in observations:
        if observation["subject"]["entity_id"]!="temporal:anchor":
            continue
        valid_from=observation["temporal"]["valid_from"][:10]
        valid_to=observation["temporal"]["valid_to"]
        valid_from_time=time.strptime(valid_from,"%Y-%m-%d")
        if instant < valid_from_time:
            continue
        if valid_to is not None and instant >= time.strptime(valid_to[:10],"%Y-%m-%d"):
            continue
        selected.append(observation)
    if len(selected)!=1:
        raise ValueError(f"expected one temporal answer, found {len(selected)}")
    return selected[0]["object"]["value"]


def _decision_lookup(observations: list[dict[str, Any]]) -> dict[str, str]:
    targets=["workload:protected","workload:missing","workload:allowed"]
    results={}
    for target in targets:
        matches=[
            item for item in observations
            if item["subject"]["entity_id"]==target
            and item["predicate"]=="policyDecision"
        ]
        normative=[item for item in matches if item["layer"]=="NORMATIVE"]
        if len(normative)>1:
            raise ValueError(f"multiple normative policies for {target}")
        if len(normative)==1:
            observation=normative[0]
            decision=evaluate_permission_claim(
                observation_layer=observation["layer"],
                claimed_policy_decision=observation["object"]["value"],
            )["decision"]
        elif matches:
            observation=sorted(matches,key=lambda item:item["observation_id"])[0]
            decision=evaluate_permission_claim(
                observation_layer=observation["layer"],
                claimed_policy_decision=observation["object"]["value"],
            )["decision"]
        else:
            decision="ABSTAIN"
        results[target]=decision
    return results


def _slice_case() -> dict[str, Any]:
    return {
        "case_id":"SCALE_BOUNDED_AUTHORITY_SLICE",
        "context":{"purpose":"scale-benchmark-authority-slice"},
        "max_observations":12,
        "required_entity_types":["WorkloadClass"],
        "required_predicates":["policyDecision"],
        "forbidden_entity_types":["ScientificExperiment"],
        "authority_widening_allowed":False,
    }


def _measure(operation, repetitions: int) -> dict[str, int]:
    samples=[]
    for _ in range(repetitions):
        started=time.perf_counter_ns()
        operation()
        samples.append(time.perf_counter_ns()-started)
    return {
        "min":min(samples),
        "mean":int(round(statistics.fmean(samples))),
        "p50":_nearest_rank(samples,0.50),
        "p95":_nearest_rank(samples,0.95),
        "max":max(samples),
    }


def run_scale_benchmark(contract: dict[str, Any]) -> dict[str, Any]:
    generator=contract["generator"]
    results=[]

    for size in generator["sizes"]:
        observations=generate_synthetic_state(
            size=size,
            seed=generator["seed"],
            provenance_refs_per_observation=generator["provenance_refs_per_observation"],
        )
        manifest_hash=canonical_manifest_hash(observations)
        second_hash=canonical_manifest_hash(generate_synthetic_state(
            size=size,
            seed=generator["seed"],
            provenance_refs_per_observation=generator["provenance_refs_per_observation"],
        ))
        if manifest_hash!=second_hash:
            raise ValueError("deterministic manifest hash mismatch")

        slice_result=select_bounded_ontology_slice(
            observations=observations,
            case=_slice_case(),
            semantic_manifest_id=manifest_hash,
        )
        if len(slice_result["observations"])>12:
            raise ValueError("bounded slice exceeded frozen maximum")

        provenance_result=_provenance_traversal(observations)
        if not provenance_result["all_exact_generated_sources"]:
            raise ValueError("provenance traversal failed")

        old_value=_temporal_replay(observations,"2026-01-15")
        new_value=_temporal_replay(observations,"2026-02-15")
        if (old_value,new_value)!=("old","new"):
            raise ValueError("temporal replay mismatch")

        decisions=_decision_lookup(observations)
        expected_decisions={
            "workload:protected":"DENY",
            "workload:missing":"ABSTAIN",
            "workload:allowed":"ALLOW",
        }
        if decisions!=expected_decisions:
            raise ValueError(f"decision lookup mismatch: {decisions}")
        authorization_escape_count=sum(
            1 for target in ["workload:protected","workload:missing"]
            if decisions[target]=="ALLOW"
        )
        if authorization_escape_count!=0:
            raise ValueError("authorization escape in scale fixture")

        repetitions=contract["repetitions"][str(size)]
        operations={
            "canonical_manifest_rebuild":lambda: canonical_manifest_hash(observations),
            "bounded_ontology_slice":lambda: select_bounded_ontology_slice(
                observations=observations,
                case=_slice_case(),
                semantic_manifest_id=manifest_hash,
            ),
            "provenance_traversal":lambda: _provenance_traversal(observations),
            "temporal_replay":lambda: (
                _temporal_replay(observations,"2026-01-15"),
                _temporal_replay(observations,"2026-02-15"),
            ),
            "decision_lookup":lambda: _decision_lookup(observations),
        }
        measurements={
            name:_measure(operation,repetitions)
            for name,operation in operations.items()
        }

        results.append({
            "size":size,
            "repetitions":repetitions,
            "semantic_manifest_sha256":manifest_hash,
            "deterministic_rebuild":manifest_hash==second_hash,
            "slice_observation_count":len(slice_result["observations"]),
            "slice_bounded":len(slice_result["observations"])<=12,
            "provenance_reference_count":provenance_result["reference_count"],
            "provenance_unique_source_count":len(provenance_result["unique_source_ids"]),
            "provenance_exact_generated_sources":provenance_result["all_exact_generated_sources"],
            "temporal_replay":{"2026-01-15":old_value,"2026-02-15":new_value},
            "decision_lookup":decisions,
            "authorization_escape_count":authorization_escape_count,
            "latency_ns":measurements,
        })

    return {
        "schema_version":1,
        "benchmark_id":contract["benchmark_id"],
        "fixture_kind":contract["fixture_kind"],
        "generator":deepcopy(generator),
        "results":results,
        "all_correct":all(
            item["deterministic_rebuild"]
            and item["slice_bounded"]
            and item["provenance_exact_generated_sources"]
            and item["temporal_replay"]=={"2026-01-15":"old","2026-02-15":"new"}
            and item["authorization_escape_count"]==0
            for item in results
        ),
        "pass_fail_latency_threshold":None,
        "performance_promotion_claim":False,
        "production_mutation":False,
        "scientific_evidence":False,
    }
