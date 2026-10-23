import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control_integration"


def load(path):
    return json.loads((BASE / path).read_text())


def test_integration_contract_is_read_only_and_production_closed():
    contract = load("integration-contract.v1.json")
    boundary = contract["research_engine_read_boundary"]
    assert boundary["allowed_read_methods"] == [
        "claim_support",
        "qualifiers_for_method",
        "source_level_counts",
        "provenance_path",
    ]
    assert set(boundary["adapter_forbidden_methods"]) == {
        "initialize",
        "put_work",
        "record_verification_event",
        "create_knowledge_checkpoint",
    }
    production = contract["production_boundary"]
    assert all(
        production[key] is False
        for key in [
            "production_mutation_authorized",
            "database_schema_change_authorized",
            "policy_change_authorized",
            "extract_verify_change_authorized",
            "lambda_or_executor_routing_change_authorized",
        ]
    )
    assert production["next_promotion_requires_separate_task"] is True


def test_extract_binding_is_explicitly_unresolved():
    contract = load("integration-contract.v1.json")
    gate = contract["extract_binding_gate"]
    assert gate["state"] == "UNRESOLVED"
    assert gate["current_evidence_role"] == "DEPENDENCY_STATE_ONLY"
    assert any("accepted EXTRACT architecture/output contract" in item for item in gate["accepted_binding_requires"])
    assert any("WAVE7A3_REJECT" in item for item in gate["forbidden_until_resolved"])


def test_relational_baseline_has_four_real_store_queries_and_no_threshold():
    contract = load("relational-baseline-contract.v1.json")
    assert [item["query_id"] for item in contract["queries"]] == [
        "Q1_CLAIM_SUPPORT",
        "Q2_METHOD_QUALIFIERS",
        "Q3_SOURCE_LEVEL_COUNTS",
        "Q4_PROVENANCE_PATH",
    ]
    timing = contract["timing_methodology"]
    assert timing["warmup_repetitions_per_query"] == 5
    assert timing["measured_repetitions_per_query"] == 50
    assert timing["pass_fail_latency_threshold"] is None
    assert timing["performance_claim_allowed"] is False
    assert contract["environment"]["production_database_access"] is False


def test_expected_answers_match_frozen_five_work_fixture():
    contract = load("relational-baseline-contract.v1.json")
    fixture = json.loads((ROOT / "tests" / "data" / "rf_osr_five_work.json").read_text())
    evidence = {row["id"]: row for row in fixture["evidence_objects"]}
    relations = fixture["evidence_relations"]

    assert len(fixture["works"]) == contract["fixture_identity"]["works"]
    assert len(fixture["evidence_objects"]) == contract["fixture_identity"]["evidence_objects"]
    assert len(relations) == contract["fixture_identity"]["evidence_relations"]

    counts = {}
    for row in fixture["evidence_objects"]:
        counts[row["source_level"]] = counts.get(row["source_level"], 0) + 1
    assert counts == contract["fixture_identity"]["source_level_counts"]

    q1 = contract["queries"][0]
    support_ids = sorted(
        rel["source_id"]
        for rel in relations
        if rel["relation_type"] == "SUPPORTS"
        and rel["target_id"] == q1["arguments"]["claim_id"]
    )
    assert support_ids == q1["expected"]["ordered_result_ids"]
    assert all(evidence[item]["evidence_type"] == "RESULT" for item in support_ids)

    q2 = contract["queries"][1]
    qualifiers = [
        rel["source_id"]
        for rel in relations
        if rel["relation_type"] == "QUALIFIES"
        and rel["target_id"] == q2["arguments"]["method_id"]
    ]
    assert qualifiers == [q2["expected"]["qualifier_id"]]
    assert evidence[qualifiers[0]]["evidence_type"] == q2["expected"]["evidence_type"]

    q4 = contract["queries"][3]
    edges = {(rel["source_id"], rel["target_id"], rel["relation_type"]) for rel in relations}
    expected_path = q4["expected"]["path"]
    assert (
        expected_path[0],
        expected_path[1],
        "DERIVED_FROM",
    ) in edges
    assert (
        expected_path[1],
        expected_path[2],
        q4["expected"]["relation_type"],
    ) in edges


def test_workspace_control_source_manifest_is_pinned_and_authority_typed():
    manifest = load("workspace-control-source-manifest.v1.json")
    assert manifest["source_blobs_verified_identical_across_commits"] is True
    roles = {item["role"] for item in manifest["sources"]}
    assert "NORMATIVE_AUTHORITY" in roles
    assert "RECEIPT_AND_RUNTIME_CONTRACT" in roles
    assert "DEPENDENCY_STATE_ONLY" in roles
    for item in manifest["sources"]:
        assert len(item["git_blob_sha"]) == 40
        int(item["git_blob_sha"], 16)
    extract = next(item for item in manifest["sources"] if item["source_id"] == "wc-extract-task")
    assert extract["extract_binding_authority"] is False


def test_contract_pins_exact_prototype_promotion_without_production_authority():
    contract = load("integration-contract.v1.json")
    result = contract["prototype_result"]
    assert result["verdict"] == "PROMOTE_TO_INTEGRATION"
    assert result["production_integration_authorized"] is False
    assert result["commit_sha"] == "4317830c4f3779cebb8a5cf0d7eeec369b85d5ea"
