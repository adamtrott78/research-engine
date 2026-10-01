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
