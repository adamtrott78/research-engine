import os

from research_engine.daemon import DaemonLimits, worker_counts_from_env
from research_engine.orchestration import ExecutorTier


def test_worker_counts_are_tier_configurable(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENGINE_DETERMINISTIC_WORKERS", "3")
    monkeypatch.setenv("RESEARCH_ENGINE_CHEAP_WORKERS", "7")
    monkeypatch.setenv("RESEARCH_ENGINE_STANDARD_WORKERS", "4")
    monkeypatch.setenv("RESEARCH_ENGINE_STRONG_WORKERS", "2")
    counts = worker_counts_from_env()
    assert counts == {
        ExecutorTier.DETERMINISTIC: 3,
        ExecutorTier.CHEAP: 7,
        ExecutorTier.STANDARD: 4,
        ExecutorTier.STRONG: 2,
    }


def test_daemon_limits_have_bounded_defaults():
    limits = DaemonLimits()
    assert limits.max_active_runs > 0
    assert limits.max_pending_human > 0
    assert limits.stale_worker_seconds > 0
