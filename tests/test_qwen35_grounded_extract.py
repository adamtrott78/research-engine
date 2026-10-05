import json
import re
from pathlib import Path

from research_engine.qwen_grounded_extract import (
    exact_text_candidates,
    execution_identity,
    guard_and_canonicalize,
    operation_id,
    package_identity,
)


ROOT=Path(__file__).parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"qwen35-grounded-extract-contract.v1.json"


def test_exact_candidates_preserve_authoritative_raw_substring():
    text="First con-\ntrast sentence.\nSecond result is 90.28%.\n"
    rows=exact_text_candidates([{"page_number":3,"text":text}])
    assert rows
    assert any("contrast sentence" in row["display_text"] for row in rows)
    for row in rows:
        assert row["raw_source_form"] in text
        assert row["page_number"]==3
        assert row["source_layer"]=="AUTHORITATIVE_PDF_TEXT"


def test_qwen_selection_cannot_invent_quote_or_page():
    plan={"candidates":[{
        "candidate_id":"pdf:p2:0000",
        "source_layer":"AUTHORITATIVE_PDF_TEXT",
        "candidate_kind":"TEXT_SPAN",
        "page_number":2,
        "pdf_bbox":None,
        "raw_source_form":"exact source text",
        "display_text":"exact source text",
        "derivation":{"processor":"test"},
    }]}
    good=guard_and_canonicalize(plan,{"selections":[{
        "candidate_id":"pdf:p2:0000",
        "type":"result",
        "normalized_form":"source result",
        "confidence":0.9,
    }]})
    assert good["status"]=="PASS_GROUNDED"
    ev=good["canonical_result"]["evidence"][0]
    assert ev["raw_source_form"]=="exact source text"
    assert ev["page_number"]==2

    bad=guard_and_canonicalize(plan,{"selections":[{
        "candidate_id":"invented",
        "type":"result",
        "normalized_form":"fake",
        "confidence":0.9,
    }]})
    assert bad["status"]=="FALLTHROUGH_CODEX"
    assert any("UNKNOWN_CANDIDATE_IDS" in x for x in bad["reasons"])


def test_operation_identity_is_deterministic_and_input_bound():
    a=operation_id(git_sha="a"*40,source_job_id="job:x",source_pdf_sha256="b"*64)
    b=operation_id(git_sha="a"*40,source_job_id="job:x",source_pdf_sha256="b"*64)
    c=operation_id(git_sha="a"*40,source_job_id="job:y",source_pdf_sha256="b"*64)
    assert a==b
    assert a!=c
    assert a.startswith("sha256:")


def test_contract_has_one_scientific_c4_and_live_shared_c3():
    d=json.loads(CONTRACT.read_text())
    assert d["call_budget"]["c3_synthetic_canary"]==1
    assert d["call_budget"]["c4_grounded_extract_scientific"]==1
    assert d["call_budget"]["one_model_boot"] is True
    assert d["call_budget"]["c3_and_c4_share_live_engine"] is True
    assert d["qwen_output_contract"]["qwen_may_not_return"]==[
        "raw_source_form","page_number","pdf_bbox","source_sha256"
    ]
    assert d["boundaries"]["production_routing_changed"] is False
    assert d["boundaries"]["scientific_state_materialized"] is False


def test_helper_has_single_engine_boot_c3_before_c4():
    text=(ROOT/"benchmarks"/"document_intelligence"/"run_qwen35_grounded_extract.py").read_text()
    assert text.count("llm = LLM(")==1
    assert text.count("C3_CANARY_START")==1
    assert text.count("C4_DOMAIN_START")==1
    assert text.index("C3_CANARY_START") < text.index("C4_DOMAIN_START")
    assert 'report["scientific_inference_calls"] = 1' in text
    prompt=text.split("def domain_prompt",1)[1].split("def main",1)[0]
    assert "raw_source_form" in prompt  # forbidden field is explicitly named
    assert "Do not output" in prompt


def test_runner_reuses_sidecar_and_has_no_materialization_or_install():
    text=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    assert "ensure-sidecar" in text
    assert "PACKAGE_INSTALL_DOWNLOAD_WORK=FORBIDDEN_AND_SKIPPED" in text
    assert "SemanticResultMaterializer" not in text
    assert "pip install" not in text
    assert "C3_CALLS=1" in text
    assert "C4_SCIENTIFIC_CALLS=1" in text
    assert "COMPLETED_REUSE" in text
    assert "runtime-operation-receipt.json" in text


def test_sidecar_candidate_layers_are_explicitly_derived():
    d=json.loads(CONTRACT.read_text())
    assert d["authority"]=="ORIGINAL_CONTENT_ADDRESSED_PDF"
    assert d["sidecar_policy"]["authority_remains_pdf"] is True
    assert "CodeFormulaV2 formula-enrichment outputs when sidecar routing naturally produces them" in d["candidate_layers"]
    assert "TableFormer ACCURATE grids when sidecar routing naturally produces them" in d["candidate_layers"]


def test_shared_wave3_identities_are_sha256_contract_ids():
    pattern=r"^sha256:[0-9a-f]{64}$"
    assert re.fullmatch(pattern, execution_identity())
    assert re.fullmatch(pattern, package_identity())
    op=operation_id(
        git_sha="a"*40,
        source_job_id="job:x",
        source_pdf_sha256="b"*64,
    )
    assert re.fullmatch(pattern, op)


def test_runner_separates_strict_shared_receipts_from_project_detail():
    text=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    assert 'certification="$out/runtime-certification.json"' in text
    assert 'wave3="$out/wave3-adoption-evidence.json"' in text
    assert 'operation_receipt="$out/runtime-operation-receipt.json"' in text
    assert 'operation_detail="$out/runtime-operation-detail.json"' in text
    assert '"execution_identity":execid' in text
    assert '"package_identity":pkgid' in text
    assert '"result_decision":decision' in text
    assert '"result_decision":decision' not in text.split(
        'receipt={',1
    )[1].split('detail={',1)[0]
    assert '"artifacts":{' not in text.split(
        'receipt={',1
    )[1].split('detail={',1)[0]


def test_wave3_certification_freeze_requires_explicit_r2_hit_and_c3():
    text=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    assert 'shared_r2="VERIFIED_HIT" if local_r2=="VERIFIED_HIT"' in text
    assert 'freeze=runtime_certified and shared_r2=="VERIFIED_HIT"' in text
    assert 'runtime_certified=all(x["status"]=="PASS" for x in stages[:5])' in text
