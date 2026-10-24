import json
from pathlib import Path
import warnings

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"


def test_scale_generator_is_deterministic_and_benchmark_correct():
    from prototypes.semantic_control_hardening.scale_benchmark import (
        canonical_manifest_hash,
        generate_synthetic_state,
        run_scale_benchmark,
    )
    contract=json.loads((BASE/"scale-contract.v1.json").read_text())
    first=generate_synthetic_state(
        size=1000,
        seed=contract["generator"]["seed"],
        provenance_refs_per_observation=contract["generator"]["provenance_refs_per_observation"],
    )
    second=generate_synthetic_state(
        size=1000,
        seed=contract["generator"]["seed"],
        provenance_refs_per_observation=contract["generator"]["provenance_refs_per_observation"],
    )
    assert canonical_manifest_hash(first)==canonical_manifest_hash(second)

    result=run_scale_benchmark(contract)
    assert result["all_correct"] is True
    assert [item["size"] for item in result["results"]]==[1000,10000,100000]
    assert all(item["slice_bounded"] for item in result["results"])
    assert all(item["provenance_exact_generated_sources"] for item in result["results"])
    assert all(item["authorization_escape_count"]==0 for item in result["results"])
    assert all(item["decision_lookup"]=={
        "workload:protected":"DENY",
        "workload:missing":"ABSTAIN",
        "workload:allowed":"ALLOW",
    } for item in result["results"])
    assert result["pass_fail_latency_threshold"] is None
    assert result["performance_promotion_claim"] is False
    assert result["production_mutation"] is False
    assert result["scientific_evidence"] is False

    warnings.warn(
        "SEMANTIC_CONTROL_HARDENING_SCALE_V1="
        + json.dumps(result,sort_keys=True,separators=(",",":")),
        UserWarning,
        stacklevel=1,
    )


def test_scale_fixture_never_claims_authority_or_scientific_evidence():
    from prototypes.semantic_control_hardening.scale_benchmark import (
        generate_synthetic_state,
    )
    observations=generate_synthetic_state(
        size=1000,
        seed=20261023,
        provenance_refs_per_observation=2,
    )
    assert all(item["attributes"]["synthetic"] is True for item in observations)
    assert all(item["verification"]["state"]=="SYNTHETIC_TEST_FIXTURE" for item in observations)
    assert all(
        ref["synthetic"] is True
        for item in observations
        for ref in item["provenance"]
    )
