import pytest

from research_engine.daemon import BudgetPolicyError, validate_budget_policy


def test_budget_policy_accepts_token_and_tier_limits():
    policy = validate_budget_policy(
        {
            "budgets": {
                "max_input_tokens": 1000,
                "max_output_tokens": 500,
                "max_semantic_jobs": 20,
                "max_jobs_by_tier": {
                    "CHEAP": 10,
                    "STANDARD": 8,
                    "STRONG": 2,
                },
            }
        }
    )
    assert policy["budgets"]["max_jobs_by_tier"]["STRONG"] == 2


def test_budget_policy_rejects_negative_or_unknown_limits():
    with pytest.raises(BudgetPolicyError):
        validate_budget_policy({"budgets": {"max_input_tokens": -1}})
    with pytest.raises(BudgetPolicyError):
        validate_budget_policy(
            {"budgets": {"max_jobs_by_tier": {"HUMAN": 1}}}
        )
