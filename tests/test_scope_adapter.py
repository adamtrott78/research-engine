import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "policies" / "workspace-control-scope-adapter.v1.json"
README_PATH = ROOT / "README.md"

EXPECTED_WORKSPACE_CONTROL = "1295477546c802029d7c52755beeacdef31b5346"


def load_policy():
    return json.loads(POLICY_PATH.read_text())


def test_scope_adapter_pins_shared_contract():
    policy = load_policy()
    shared = policy["shared_contract"]
    assert shared["repository"] == "adamtrott78/workspace-control"
    assert shared["checkpoint_sha"] == EXPECTED_WORKSPACE_CONTROL
    assert shared["procedure"] == "procedures/scope_closure.md"
    assert shared["envelope_schema"] == "schemas/scope_envelope.v1.schema.json"
    assert shared["control_cli"] == "scripts/scope_control.py"


def test_scope_adapter_derives_authority_from_owning_task():
    policy = load_policy()
    derivation = policy["envelope_derivation"]
    assert derivation["authority"] == "owning_workspace_control_task_capsule"
    assert derivation["task_id_source"] == "task_id"
    assert derivation["next_safe_action_source"] == "next_safe_action"
    assert "objective" not in policy
    assert "next_safe_action" not in policy


def test_scope_adapter_defaults_defer_adjacent_research_work():
    policy = load_policy()
    deferred = "\n".join(policy["defer_by_default"])
    for phrase in (
        "model-family comparisons",
        "extraction redesigns",
        "runner hardening",
        "benchmark variants",
        "scientific experiments",
        "deployment scripts",
    ):
        assert phrase in deferred


def test_scope_adapter_preserves_research_authority():
    policy = load_policy()
    protected = "\n".join(policy["protected_authority"])
    for phrase in (
        "scientific source bytes and provenance",
        "production routing and materialization",
        "daemon and Codex execution authority",
        "model, prompt, threshold",
        "workspace-control task state",
    ):
        assert phrase in protected


def test_scope_closure_is_fail_closed():
    policy = load_policy()
    closure = policy["closure_policy"]
    assert closure == {
        "stop_when_objective_satisfied": True,
        "defer_nonblocking_findings": True,
        "require_validation_pass": True,
        "require_durable_checkpoint": True,
    }


def test_allowed_and_deferred_categories_do_not_duplicate():
    policy = load_policy()
    allowed = set(policy["allowed_now"])
    deferred = set(policy["defer_by_default"])
    assert allowed
    assert deferred
    assert allowed.isdisjoint(deferred)


def test_research_engine_does_not_vendor_shared_scope_engine():
    assert not (ROOT / "scripts" / "scope_control.py").exists()
    assert not (ROOT / "schemas" / "scope_envelope.v1.schema.json").exists()


def test_root_readme_routes_to_scope_adapter_and_stop_gate():
    text = README_PATH.read_text()
    assert "policies/workspace-control-scope-adapter.v1.json" in text
    assert EXPECTED_WORKSPACE_CONTROL in text
    assert "procedures/scope_closure.md" in text
    assert "CHECKPOINT_AND_STOP" in text
