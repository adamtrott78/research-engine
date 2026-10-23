import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks" / "semantic_control"


def load(name):
    return json.loads((BASE / name).read_text())


def test_contract_set_contains_all_five_v1_interfaces():
    contracts = load("contracts.v1.json")
    assert set(contracts["contracts"]) == {
        "SemanticObservationV1",
        "EvidenceRequirementV1",
        "OntologySliceV1",
        "CandidateActionV1",
        "DecisionEvaluationV1",
    }


def test_fixture_manifest_pins_exact_authoritative_sources():
    manifest = load("fixture-manifest.v1.json")
    assert len(manifest["sources"]) == 7
    roles = {source["role"] for source in manifest["sources"]}
    assert "scientific_result" in roles
    assert "normative_authority" in roles
    assert "capability_and_route_state" in roles
    for source in manifest["sources"]:
        assert len(source["commit_sha"]) == 40
        assert len(source["git_blob_sha"]) == 40
        int(source["commit_sha"], 16)
        int(source["git_blob_sha"], 16)


def test_candidate_actions_are_structurally_non_authoritative():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    assert snapshot["candidate_actions"]
    assert all(
        action["authorization_state"] == "UNDECIDED"
        for action in snapshot["candidate_actions"]
    )


def test_evidence_requirements_are_request_only():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    assert snapshot["evidence_requirements"]
    assert all(
        requirement["execution_effect"] == "REQUEST_ONLY"
        for requirement in snapshot["evidence_requirements"]
    )


def test_authority_adversaries_fail_closed():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    decisions = {
        evaluation["evaluation_id"]: evaluation["decision"]
        for evaluation in snapshot["decision_evaluations"]
    }
    assert decisions["eval:gpu-heavy-deny"] == "DENY"
    assert decisions["eval:wave7b-abstain"] == "ABSTAIN"
    assert "ALLOW" not in decisions.values()


def test_contradictions_are_preserved_bidirectionally():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    by_id = {obs["observation_id"]: obs for obs in snapshot["observations"]}
    assert by_id["obs:synthetic-claim-a"]["contradicts"] == ["obs:synthetic-claim-b"]
    assert by_id["obs:synthetic-claim-b"]["contradicts"] == ["obs:synthetic-claim-a"]


def test_supersession_preserves_history():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    by_id = {obs["observation_id"]: obs for obs in snapshot["observations"]}
    assert by_id["obs:synthetic-old"]["assertion_state"] == "SUPERSEDED"
    assert by_id["obs:synthetic-new"]["supersedes"] == ["obs:synthetic-old"]


def test_wave7a3_fixture_is_reject_not_accepted_extract_architecture():
    snapshot = load("fixtures/authoritative-snapshot.v1.json")
    obs = next(
        obs for obs in snapshot["observations"]
        if obs["observation_id"] == "obs:wave7a3-verdict"
    )
    assert obs["object"]["value"] == "WAVE7A3_REJECT"
    assert obs["attributes"]["wave7b_authorized"] is False
    assert obs["attributes"]["production_extract_changed"] is False
    assert obs["attributes"]["qwen_verify_changed"] is False


def test_benchmark_suite_freezes_p1_through_p8():
    suite = load("fixtures/benchmark-cases.v1.json")
    families = {case["family"] for case in suite["cases"]}
    assert families == {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
    ids = [case["case_id"] for case in suite["cases"]]
    assert len(ids) == len(set(ids))
