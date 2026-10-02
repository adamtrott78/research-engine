import subprocess
from pathlib import Path


def test_operator_shortcuts_are_bash_syntax_valid():
    root = Path(__file__).parents[1]
    paths = [
        root / "deploy" / "operator" / "START",
        root / "deploy" / "operator" / "STOP",
        root / "deploy" / "operator" / "HUD",
        root / "deploy" / "operator" / "STATUS",
        root / "deploy" / "install-operator-shortcuts.sh",
        root / "deploy" / "deploy-hud-paused.sh",
        root / "deploy" / "run-verify-calibration.sh",
        root / "deploy" / "run-verify-ab-batch.sh",
        root / "deploy" / "publish-ops-mirror.sh",
        root / "deploy" / "run-document-native-fast-bakeoff.sh",
    ]
    for path in paths:
        assert path.exists(), path
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, f"{path}: {proc.stderr}"


def test_stop_shortcut_verifies_process_is_dead_before_success():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "operator" / "STOP").read_text()
    assert 'systemctl --user stop "$unit"' in text
    assert '[ "$active" = "inactive" ]' in text
    assert '[ "$sub" = "dead" ]' in text
    assert '[ "${pid:-0}" = "0" ]' in text
    assert 'echo "STOPPED"' in text


def test_paused_hud_deployer_uses_usage_ui_shape():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "deploy-hud-paused.sh").read_text()
    assert "LEFT percent" in text
    assert "resets in [e.g. 3h 20m" in text
    assert "--left-percent" in text
    assert "--resets-in" in text
    assert "--used-percent" not in text


def test_verify_calibration_runner_never_starts_daemon():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "run-verify-calibration.sh").read_text()
    assert "calibrate-verify" in text
    assert "--selector lowest" in text
    assert "--selector highest" in text
    assert "audit-extract-provenance" in text
    assert "allowance-observe-paste" in text
    assert "Highlight/copy the entire 5-hour + weekly usage-limit section" in text
    assert "pre low/high VERIFY replay batch" in text
    assert "post low/high VERIFY replay batch" in text
    assert "DAEMON_MUST_REMAIN_STOPPED_FOR_VERIFY_CALIBRATION" in text
    assert "systemctl --user start" not in text
    assert " START" not in text


def test_verify_ab_batch_is_bounded_non_materializing_operator_flow():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "run-verify-ab-batch.sh").read_text()
    assert "allowance-observe-paste" in text
    assert "RESEARCH_ENGINE_VERIFY_PAGE_RADIUS=\"$radius\"" in text
    assert "calibrate-verify" in text
    assert "--source-job-id" in text
    assert "REPEATS_TOO_LARGE_MAX_12" in text
    assert "TOTAL_SEMANTIC_CALLS=$((2 * repeats))" in text
    assert "DAEMON_MUST_REMAIN_STOPPED_FOR_VERIFY_AB_BATCH" in text
    assert "systemctl --user start" not in text
    assert " START" not in text


def test_ops_mirror_publisher_uses_github_issue_not_git_commits_or_daemon_control():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "publish-ops-mirror.sh").read_text()
    assert "research-engine\" ops-mirror" in text
    assert "gh issue create" in text
    assert "gh issue edit" in text
    assert "gh issue comment" in text
    assert "gh auth status" in text
    assert "ops-snapshot" not in text
    assert "git commit" not in text
    assert "git push" not in text
    assert "systemctl --user start" not in text
    assert "systemctl --user stop" not in text


def test_document_native_fast_bakeoff_is_stopped_daemon_and_lightweight_only():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "run-document-native-fast-bakeoff.sh").read_text()
    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "systemctl --user start" not in text
    assert "docling-slim[cli,format-pdf]==2.131.0" in text
    assert "mineru>=4.0,<5" in text
    assert "docling \\" not in text
    assert "document-benchmark-external" in text
    assert "docling" in text
    assert "native" in text
    assert "mineru" in text
    assert "flash_native" in text
    assert "marker" not in text.lower()
    assert "paddle" not in text.lower()
    assert "--use_llm" not in text
    assert "DOCUMENT_NATIVE_FAST_BAKEOFF=PASS" in text
