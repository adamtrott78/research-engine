import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks" / "semantic_control" / "local-reasoner-architecture.v1.json"


def load():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_local_reasoner_architecture_core_bounds():
    c = load()
    assert c["status"] == "FROZEN_EXPERIMENTAL_DESIGN"
    assert c["serving_architecture"]["physical_owner_count"] == 1
    assert c["serving_architecture"]["max_concurrency"] == 1
    assert c["resource_gates"]["forty_minute_endurance"]["wall_seconds"] == 2400
    assert c["resource_gates"]["forty_minute_endurance"]["external_llm_calls"] == 0
    assert c["resource_gates"]["forty_minute_endurance"]["cloud_model_fallbacks"] == 0
    assert c["resource_gates"]["forty_minute_endurance"]["scientific_state_materialized"] is False


def test_first_wave_has_baseline_control_and_reasoning_distillation():
    c = load()
    arms = {row["id"]: row for row in c["candidate_ladder"]["first_wave"]}
    assert set(arms) == {
        "QWEN35_4B_BASELINE",
        "QWEN35_9B_CONTROL",
        "QWYTHOS_9B_V2_REASONER",
    }
    assert arms["QWEN35_4B_BASELINE"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert arms["QWEN35_9B_CONTROL"]["model"] == "Qwen/Qwen3.5-9B"
    assert arms["QWYTHOS_9B_V2_REASONER"]["model"] == "nvcky/Qwythos"


def test_existing_verify_is_not_replaced_by_design():
    c = load()
    inherited = c["inherited_facts"]["bounded_verify_model"]
    assert inherited["disposition"] == "PRESERVE_UNCHANGED_DURING_BAKEOFF"
    assert "No Qwen3.5-4B VERIFY replacement." in c["explicit_non_authorizations"]


def test_experimental_iterations_do_not_require_independent_review():
    c = load()
    policy = c["iteration_policy"]
    assert policy["experimental_architecture"]["independent_review_required"] is False
    assert policy["promotion_boundary"]["independent_review_required_by_this_contract"] is False
    assert policy["promotion_boundary"]["requires_explicit_user_authorization"] is True


def test_resource_policy_does_not_assume_co_residency_or_dual_gpu():
    c = load()
    gpu = c["serving_architecture"]["gpu_policy"]
    gate = c["resource_gates"]["first_wave_single_gpu"]
    assert gpu["co_residency"] == "NOT_AUTHORIZED_UNTIL_MEASURED"
    assert gpu["dual_gpu_reasoner"] == "SECOND_WAVE_ONLY_IF_9B_CLASS_FAILS"
    assert gate["hard_peak_gpu_memory_mib_lte"] == 16384
    assert gate["no_duplicate_model_owner"] is True
    assert gate["no_second_gpu_visibility"] is True


def test_bakeoff_uses_actual_failure_modes():
    c = load()
    suite_ids = {row["id"] for row in c["benchmark_suite"]["suites"]}
    assert {
        "BOUNDED_VERIFY_CONTROL",
        "ATOMIC_FACT_AND_OBLIGATION_REASONING",
        "SEMANTIC_CONTROL",
        "RESEARCH_GRAPH_REASONING",
        "ADVERSARIAL_CRITIQUE",
    } <= suite_ids
    hard = set(c["scoring"]["hard_failures"])
    assert "fabricated or unlocatable evidence" in hard
    assert "false acceptance on frozen known-negative support/review cases" in hard


def test_gitrunner_plan_is_measurement_only():
    c = load()
    phases = c["gitrunner_execution_plan"]
    assert [row["phase"] for row in phases] == list(range(6))
    assert all(row["production_change"] is False for row in phases)
    assert phases[0]["runner"] == "lambda-research-engine CPU"
    assert phases[2]["runner"] == "lambda-research-engine-gpu / selected-GPU isolation"
