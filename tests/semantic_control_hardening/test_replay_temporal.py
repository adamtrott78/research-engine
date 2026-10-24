import json
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"benchmarks"/"semantic_control_hardening"

def load(name):
    return json.loads((BASE/name).read_text())

def test_frozen_historical_decisions_replay_exactly():
    from prototypes.semantic_control_hardening.replay_temporal import replay_historical_cases
    cases=load("historical-replay-cases.v1.json")["replay_cases"]
    results=replay_historical_cases(cases)
    assert [r["decision"] for r in results]==["ALLOW","DENY","ABSTAIN"]
    assert all(r["matches_expected"] for r in results)
    assert all(r["current_state_used_for_decision"] is False for r in results)
    assert all(r["execution_receipt_id"] is None for r in results)

def test_divergent_current_state_cannot_change_historical_replay():
    from prototypes.semantic_control_hardening.replay_temporal import replay_historical_decision
    cases=load("historical-replay-cases.v1.json")["replay_cases"]
    for case in cases:
        first=replay_historical_decision(case)
        changed=deepcopy(case)
        changed["divergent_current_state"]={
            "fixture_provenance":"SYNTHETIC_ADVERSARY_ONLY",
            "policy_epoch":999999,
            "decision":"ALLOW" if first["decision"]!="ALLOW" else "DENY",
            "arbitrary_extra_state":{"should":"never matter"},
        }
        second=replay_historical_decision(changed)
        assert first["decision"]==second["decision"]
        assert first["historical_input_sha256"]==second["historical_input_sha256"]
        assert first["divergent_current_state_sha256"]!=second["divergent_current_state_sha256"]

def test_supersession_temporal_replay_uses_validity_intervals_not_current_assertion_state():
    from prototypes.semantic_control_hardening.replay_temporal import replay_temporal_case
    case=load("temporal-version-skew-adversaries.v1.json")["temporal_cases"][0]
    result=replay_temporal_case(case)
    assert result["matches_expected"] is True
    assert result["answers"]=={
        "2026-01-15T00:00:00Z":["old"],
        "2026-02-15T00:00:00Z":["new"],
    }
    assert result["history_preserved"] is True

def test_contradiction_is_preserved_without_global_truth_resolution():
    from prototypes.semantic_control_hardening.replay_temporal import contradiction_view
    case=load("temporal-version-skew-adversaries.v1.json")["temporal_cases"][1]
    result=contradiction_view(case)
    assert result["active_observation_ids"]==["t2:false","t2:true"]
    assert set(result["values"])=={False,True}
    assert result["global_truth_resolution"]=="NONE"
    assert result["contradictions_preserved"] is True

def test_unknown_is_not_coerced_to_false():
    from prototypes.semantic_control_hardening.replay_temporal import unknown_semantics
    case=load("temporal-version-skew-adversaries.v1.json")["temporal_cases"][2]
    result=unknown_semantics(case)
    assert result["assertion_state"]=="UNKNOWN"
    assert result["value"] is None
    assert result["coerced_to_false"] is False

def test_replay_module_has_no_external_runtime_or_current_workspace_reads():
    text=(ROOT/"prototypes"/"semantic_control_hardening"/"replay_temporal.py").read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "github","workspace-control","workspace_control_adapter",
        "research_engine.postgres",
    ]:
        assert token not in text
