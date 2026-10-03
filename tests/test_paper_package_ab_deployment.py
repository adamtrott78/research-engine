from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_extract_verify_paper_package_ab_is_fail_closed_and_reuses_runtimes():
    script = (
        ROOT / "deploy" / "run-extract-verify-paper-package-ab.sh"
    ).read_text()

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in script
    assert "workspace-control-tmp" in script
    assert "paper-package-ab-plan" in script
    assert "paper-package-ab-preflight" in script
    assert "PAPER_PACKAGE_ZERO_MODEL_PREFLIGHT_FAILED" in script
    assert "paper-package-ab-replay" in script
    assert "EXPECTED_EXACTLY_NINE_SOURCE_ARTIFACTS" in script
    assert "Scientific state is not materialized" in script
    assert "PAPER_PACKAGE_AB_GATE=MEASURED" in script

    assert "python3 -m venv" not in script
    assert "pip install docling" not in script
    assert "pip install torch" not in script
    assert "--prefetch-models" not in script
    assert "run-document-native-fast-bakeoff" not in script
    assert "run-document-docling-standard-no-ocr-bakeoff" not in script


def test_paper_package_ab_batch_shims_load_each_converter_once():
    native = (
        ROOT
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_native_batch.py"
    ).read_text()
    standard = (
        ROOT
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_standard_no_ocr_batch.py"
    ).read_text()

    assert native.count("converter = build_converter()") == 1
    assert standard.count("converter = build_converter()") == 1
    assert 'for item in plan.get("source_artifacts", [])' in native
    assert 'for item in plan.get("source_artifacts", [])' in standard



def test_paper_package_ab_resume_gate_reuses_normalized_packages_only():
    script = (
        ROOT / "deploy" / "resume-extract-verify-paper-package-ab.sh"
    ).read_text()

    assert "RESUME_ROOT_REQUIRED" in script
    assert "RESUME_ROOT_MUST_BE_UNDER_VISIBLE_WORKSPACE_CONTROL_TMP" in script
    assert "EXPECTED_NINE_NATIVE_PACKAGES" in script
    assert "EXPECTED_NINE_STANDARD_PACKAGES" in script
    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in script
    assert "pg_isready" in script
    assert "paper-package-ab-preflight" in script
    assert "paper-package-ab-replay" in script
    assert "PAPER_PACKAGE_AB_RESUME_GATE=MEASURED" in script

    assert "docling_native_batch" not in script
    assert "docling_standard_no_ocr_batch" not in script
    assert "normalize_docling_document" not in script
    assert "pip install" not in script
    assert "python3 -m venv" not in script
