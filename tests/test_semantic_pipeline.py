from pathlib import Path

from research_engine.openai_executor import result_schema
from research_engine.orchestration import JobType


def test_extract_schema_requires_auditable_page_quote():
    item = result_schema(JobType.EXTRACT)["properties"]["evidence"]["items"]
    required = set(item["required"])
    assert {"raw_source_form", "page_number"} <= required
    assert "source_locator" not in required


def test_gap_schema_requires_motivating_evidence():
    item = result_schema(JobType.GAP_GENERATE)["properties"]["gaps"]["items"]
    assert "motivating_evidence_ids" in item["required"]
