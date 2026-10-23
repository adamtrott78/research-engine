import json
import os
from pathlib import Path
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_measure_frozen_relational_baseline_and_emit_machine_readable_result():
    if not os.getenv("RESEARCH_ENGINE_DATABASE_URL"):
        pytest.skip("RESEARCH_ENGINE_DATABASE_URL not set")

    from prototypes.semantic_control_integration.relational_baseline import (
        run_relational_baseline,
    )

    result = run_relational_baseline(ROOT)
    assert result["all_answers_correct"] is True
    assert result["performance_promotion_claim"] is False
    assert result["latency_threshold_applied"] is False
    assert result["production_mutation"] is False
    assert len(result["query_results"]) == 4
    assert all(item["measured_repetitions"] == 50 for item in result["query_results"])
    assert all(item["warmup_repetitions"] == 5 for item in result["query_results"])
    assert all(item["latency_ns"]["min"] > 0 for item in result["query_results"])

    warnings.warn(
        "SEMANTIC_CONTROL_RELATIONAL_BASELINE_V1="
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        UserWarning,
        stacklevel=1,
    )


def test_relational_baseline_runner_is_explicitly_non_production():
    text = (
        ROOT
        / "prototypes"
        / "semantic_control_integration"
        / "relational_baseline.py"
    ).read_text()
    lowered = text.lower()
    assert "production_database_access" in text
    assert '"production_mutation": False' in text
    for forbidden in [
        "openai",
        "requests",
        "subprocess",
        "lambda",
        "gpu",
    ]:
        assert forbidden not in lowered
