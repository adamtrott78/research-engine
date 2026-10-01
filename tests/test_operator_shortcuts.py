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
    assert "DAEMON_MUST_REMAIN_STOPPED_FOR_VERIFY_CALIBRATION" in text
    assert "systemctl --user start" not in text
    assert " START" not in text
