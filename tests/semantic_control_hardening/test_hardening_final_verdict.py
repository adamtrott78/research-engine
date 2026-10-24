import json
from pathlib import Path
import warnings

ROOT=Path(__file__).resolve().parents[2]


def test_integrated_hardening_benchmark_and_emit_result():
    from prototypes.semantic_control_hardening.integrated_benchmark import (
        run_integrated_hardening_benchmark,
    )
    result=run_integrated_hardening_benchmark(ROOT)
    assert result["verdict"]=="HARDENING_PASS"
    assert all(result["gates"].values())
    assert result["authorization_escape_count"]==0
    assert result["audit_valid"] is True
    assert result["audit_tamper_detected"] is True
    assert result["opaque_extract_socket"]["unaccepted_registration"]=="REJECT_BINDING"
    assert result["opaque_extract_socket"]["accepted_registration"]=="REGISTER_DESCRIPTOR_ONLY"
    assert result["opaque_extract_socket"]["accepted_binding_state"]=="UNBOUND"
    assert result["opaque_extract_socket"]["payload_accessed"] is False
    assert result["opaque_extract_socket"]["payload_schema_defined"] is False
    assert result["hardening_authorizes_production"] is False
    assert result["hardening_binds_extract"] is False
    assert result["production_mutation"] is False
    assert result["extract_mutation"] is False
    assert len(result["result_sha256"])==64

    warnings.warn(
        "SEMANTIC_CONTROL_INTEGRATED_HARDENING_V1="
        + json.dumps(result,sort_keys=True,separators=(",",":")),
        UserWarning,
        stacklevel=1,
    )


def test_final_benchmark_source_has_no_external_runtime_or_mutation_paths():
    text=(
        ROOT
        /"prototypes"
        /"semantic_control_hardening"
        /"integrated_benchmark.py"
    ).read_text().lower()
    for token in [
        "requests","psycopg","subprocess","openai","boto3",
        "research_engine.postgres","workspace_control_adapter",
        "update_file","create_file",
    ]:
        assert token not in text
