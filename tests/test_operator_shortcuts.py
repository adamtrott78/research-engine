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
        root / "deploy" / "run-docling-structured-normalization.sh",
        root / "deploy" / "run-document-docling-standard-no-ocr-bakeoff.sh",
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
    assert "docling-slim[cli,convert-core,format-pdf]==2.131.0" in text
    assert "mineru>=4.0,<5" in text
    assert "document-benchmark-external" in text
    assert "docling" in text
    assert "native" in text
    assert "mineru" in text
    assert "flash_native" in text
    assert "marker" not in text.lower()
    assert "paddle" not in text.lower()
    assert "--use_llm" not in text
    assert "DOCLING NATIVE SDK PREFLIGHT" in text
    assert "docling_native.py" in text
    assert "--self-test" in text
    assert "DOCUMENT_NATIVE_FAST_BAKEOFF=PASS" in text
    assert "\\${" not in text
    assert 'exit "${2:-1}"' in text
    assert '[ "${pid:-0}" != "0" ]' in text


def test_docling_structured_normalization_replay_is_stopped_daemon_and_nonsemantic():
    root = Path(__file__).parents[1]
    text = (root / "deploy" / "run-docling-structured-normalization.sh").read_text()
    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "document-normalize-docling-benchmark" in text
    assert "docling-native" in text
    assert "systemctl --user start" not in text
    assert "OPENAI_API_KEY" not in text
    assert "codex" not in text.lower()
    assert "marker" not in text.lower()
    assert "paddle" not in text.lower()
    assert "DOCLING_STRUCTURED_NORMALIZATION=PASS" in text
    assert "\\${" not in text


def test_docling_standard_no_ocr_bakeoff_is_stopped_local_and_bounded():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-document-docling-standard-no-ocr-bakeoff.sh"
    ).read_text()
    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "systemctl --user start" not in text
    assert "docling-slim[convert-core,format-pdf,models-local]==2.131.0" in text
    assert "--prefetch-models" in text
    assert "docling-tools" not in text
    assert "models download" not in text
    assert "docling-standard-no-ocr-core-cu126-v3" in text
    assert "torch==2.13.0 torchvision==0.28.0" in text
    assert "download.pytorch.org/whl/cu126" in text
    assert "SCIPY_VERSION=" in text
    assert "NO_USABLE_CUDA_DEVICE" in text
    assert "torch.cuda.get_device_name(0)" in text
    assert "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode,persistence_mode" in text
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in text
    assert "--query-gpu=index,uuid,memory.free" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "SELECTED_PHYSICAL_GPU_UUID" in text
    assert "SELECTED_PHYSICAL_GPU_INDEX" in text
    assert "allocation_probe_mib" in text
    assert "256 * 1024 * 1024" in text
    assert 'RESEARCH_ENGINE_DOCLING_DEVICE="cuda"' in text
    assert "nvidia-smi --gpu-reset" not in text
    assert "kill " not in text
    assert "pkill" not in text
    assert "RESEARCH_ENGINE_DOCLING_ARTIFACTS" in text
    assert 'RESEARCH_ENGINE_DOCLING_BIN="$runtime/bin/python"' in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "standard_no_ocr" in text
    assert "DOCLING_STANDARD_NO_OCR_SDK_SELF_TEST=PASS" not in text
    assert "--self-test" in text
    assert "marker" not in text.lower()
    assert "paddle" not in text.lower()
    assert "mineru" not in text.lower()
    assert "OPENAI_API_KEY" not in text
    assert "DOCLING_STANDARD_NO_OCR_STRUCTURE_BAKEOFF=PASS" in text
    assert "\\${" not in text
