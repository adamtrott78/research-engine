import json
from pathlib import Path

from research_engine.orchestration import JOB_TIERS, JobType


ROOT = Path(__file__).parents[1]
MATRIX = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "localization-matrix.v2.json"
)


def _matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_localization_matrix_v2_covers_every_research_job_type_once():
    data = _matrix()
    rows = data["job_type_decisions"]
    names = [row["job_type"] for row in rows]

    assert sorted(names) == sorted(str(job_type) for job_type in JobType)
    assert len(names) == len(set(names))


def test_localization_matrix_v2_matches_scheduler_tiers():
    data = _matrix()
    rows = {
        JobType(row["job_type"]): row
        for row in data["job_type_decisions"]
    }

    for job_type, tier in JOB_TIERS.items():
        assert rows[job_type]["executor_tier"] == str(tier)


def test_extract_verify_remain_authoritative_pdf_standard_semantic():
    data = _matrix()
    rows = {
        row["job_type"]: row
        for row in data["job_type_decisions"]
    }

    extract = rows["EXTRACT"]
    verify = rows["VERIFY"]

    assert extract["executor_tier"] == "STANDARD"
    assert extract["localization_decision"] == "RETAIN_BASELINE"
    assert "AUTHORITATIVE_PDF" in extract["production_placement"]

    assert verify["executor_tier"] == "STANDARD"
    assert verify["localization_decision"] == "PROMOTE_BOUNDED_LOCAL_VERIFY"
    assert verify["production_placement"] == (
        "BOUNDED_LOCAL_QWEN_PRIMARY_WITH_REMOTE_CODEX_FALLTHROUGH"
    )
    assert "authoritative" in verify["authority"].lower()


def test_measured_component_policy_locks_document_routing_boundaries():
    data = _matrix()
    components = {
        row["component"]: row
        for row in data["component_decisions"]
    }

    assert (
        components["paper_package_cache"]["decision"]
        == "VALIDATED_DERIVED_CACHE_NOT_SEMANTIC_BASELINE"
    )
    assert (
        components["chunking_and_context_assembly"]["decision"]
        == "VALIDATED_CURRENT_PDF_HYDRATOR"
    )
    assert components["ocr"]["decision"] == "DEFER_DISABLED"
    assert components["general_document_vlm"]["decision"] == "DEFER_DISABLED"
    assert (
        components["extract_semantic_offload"]["decision"]
        == "REJECT_FOR_CURRENT_BASELINE"
    )
    assert (
        components["verify_semantic_offload"]["decision"]
        == "PROMOTE_BOUNDED_LOCAL_QWEN_VERIFY"
    )


def test_localization_matrix_retains_web_and_strong_reasoning_boundaries():
    data = _matrix()
    rows = {
        row["job_type"]: row
        for row in data["job_type_decisions"]
    }

    assert rows["DISCOVERY"]["production_placement"] == (
        "REMOTE_SEMANTIC_WITH_EXTERNAL_WEB"
    )
    assert rows["NOVELTY_SEARCH"]["production_placement"] == (
        "REMOTE_SEMANTIC_WITH_EXTERNAL_WEB"
    )
    assert rows["SYNTHESIZE"]["executor_tier"] == "STRONG"
    assert rows["GAP_GENERATE"]["executor_tier"] == "STRONG"
    assert rows["NOVELTY_ARBITRATE"]["executor_tier"] == "STRONG"


def test_localization_matrix_localizes_only_validated_deterministic_jobs():
    data = _matrix()
    rows = {
        row["job_type"]: row
        for row in data["job_type_decisions"]
    }

    assert rows["ACQUIRE"]["executor_tier"] == "DETERMINISTIC"
    assert rows["ACQUIRE"]["localization_decision"] == "VALIDATED_LOCAL"
    assert rows["CHECKPOINT"]["executor_tier"] == "DETERMINISTIC"
    assert rows["CHECKPOINT"]["localization_decision"] == "VALIDATED_LOCAL"


def test_localization_matrix_requires_future_benchmarks_before_semantic_retier():
    data = _matrix()
    policy = "\n".join(data["fallback_policy"])

    assert "SCREEN/EXTRACT/VERIFY" in policy
    assert "controlled quality benchmark" in policy
    assert "before changing JOB_TIERS" in policy
