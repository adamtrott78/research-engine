import json
import re
from pathlib import Path

import importlib.util

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
    assert "source quotes" in prompt
    assert "page numbers" in prompt
    assert "bboxes" in prompt
    assert "source SHA" in prompt


def test_runner_reuses_sidecar_and_has_no_materialization_or_install():
    text=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    assert "ensure-sidecar" in text
    assert "PACKAGE_INSTALL_DOWNLOAD_WORK=FORBIDDEN_AND_SKIPPED" in text
    assert "SemanticResultMaterializer" not in text
    assert "pip install" not in text
    assert "C3_CALLS=" in text
    assert "C3_REUSED=" in text
    assert "C4_SCIENTIFIC_CALLS_STARTED=" in text
    assert "C4_SCIENTIFIC_CALLS_COMPLETED=" in text
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


def _load_grounded_helper():
    path=ROOT/"benchmarks"/"document_intelligence"/"run_qwen35_grounded_extract.py"
    spec=importlib.util.spec_from_file_location("grounded_helper",path)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_domain_prompt_compaction_preserves_candidate_ids_and_hides_provenance():
    helper=_load_grounded_helper()
    plan={
        "job":{"payload":{"item":{"id":"paper","title":"title","rationale":"why"}}},
        "candidates":[
            {
                "candidate_id":"pdf:p2:0000",
                "source_layer":"AUTHORITATIVE_PDF_TEXT",
                "candidate_kind":"TEXT_SPAN",
                "page_number":2,
                "display_text":"A"*1000,
            },
            {
                "candidate_id":"formula:p3:x",
                "source_layer":"CODEFORMULAV2",
                "candidate_kind":"FORMULA_ENRICHMENT",
                "page_number":3,
                "display_text":"B"*1000,
            },
        ],
    }
    messages=helper.domain_prompt(plan,excerpt_cap=80)
    payload=json.loads(messages[1]["content"])
    assert [row[0] for row in payload["candidates"]]==[
        "pdf:p2:0000","formula:p3:x"
    ]
    assert all(len(row)==2 for row in payload["candidates"])
    serialized=messages[1]["content"]
    assert "page_number" not in serialized
    assert "source_layer" not in serialized
    assert "candidate_kind" not in serialized
    assert "raw_source_form" not in serialized


def test_token_fit_preserves_every_candidate_id():
    helper=_load_grounded_helper()

    class FakeTokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            # Simulate a tokenizer where each four serialized characters cost a token.
            chars=sum(len(str(message["content"])) for message in messages)
            return list(range((chars+3)//4))

    plan={
        "job":{"payload":{"item":{"id":"paper","title":"title","rationale":"why"}}},
        "candidates":[
            {
                "candidate_id":f"pdf:p2:{i:04d}",
                "display_text":"candidate text "+("x"*900),
            }
            for i in range(30)
        ],
    }
    messages,meta=helper.fit_domain_prompt(plan,FakeTokenizer())
    payload=json.loads(messages[1]["content"])
    assert [row[0] for row in payload["candidates"]]==[
        f"pdf:p2:{i:04d}" for i in range(30)
    ]
    assert meta["all_candidate_ids_preserved"] is True
    assert meta["input_tokens"] <= helper.DOMAIN_INPUT_TOKEN_LIMIT
    assert meta["excerpt_cap_chars"] is not None


def test_runner_persists_certification_before_failure_exit_and_uses_actual_counts():
    text=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    cert=text.index("=== WORKSPACE-CONTROL WAVE-3 CERTIFICATION + ADOPTION DETAIL ===")
    failure=text.index('if [ "$qwen_rc" -ne 0 ]; then')
    assert cert < failure
    assert 'c4_scientific_domain_calls_started' in text
    assert 'c4_scientific_domain_calls_completed' in text
    assert 'live_engine_reused_for_domain_calls=(c3_calls==1 and c3_pass and c4_completed==1)' in text
    assert 'stage=str(result.get("failure_stage")' in text


def test_retry_reuses_passing_c3_for_same_execution_identity():
    helper=(ROOT/"benchmarks"/"document_intelligence"/"run_qwen35_grounded_extract.py").read_text()
    runner=(ROOT/"deploy"/"run-qwen35-grounded-extract.sh").read_text()
    assert "--reuse-canary-from" in helper
    assert "C3_CANARY_REUSED" in helper
    assert 'report["c3_canary_reused"] = True' in helper
    assert "RECOVER REUSABLE C3 CERTIFICATION" in runner
    assert 'receipt.get("execution_identity") != execid' in runner
    assert 'qwen_args+=(--reuse-canary-from "$reuse_c3_from")' in runner
    assert "live_engine_reused_for_domain_calls=(c3_calls==1 and c3_pass and c4_completed==1)" in runner


def test_guard_allows_one_candidate_to_support_distinct_claims():
    plan={"candidates":[{
        "candidate_id":"pdf:p2:0006",
        "source_layer":"AUTHORITATIVE_PDF_TEXT",
        "candidate_kind":"TEXT_SPAN",
        "page_number":2,
        "pdf_bbox":None,
        "raw_source_form":"one source passage with multiple supported facts",
        "display_text":"one source passage with multiple supported facts",
        "derivation":{"processor":"test"},
    }]}
    result=guard_and_canonicalize(plan,{"selections":[
        {
            "candidate_id":"pdf:p2:0006",
            "type":"method",
            "normalized_form":"method fact",
            "confidence":0.9,
        },
        {
            "candidate_id":"pdf:p2:0006",
            "type":"result",
            "normalized_form":"result fact",
            "confidence":0.8,
        },
    ]})
    assert result["status"]=="PASS_GROUNDED"
    evidence=result["canonical_result"]["evidence"]
    assert len(evidence)==2
    assert len({row["id"] for row in evidence})==2
    assert {row["raw_source_form"] for row in evidence}=={
        "one source passage with multiple supported facts"
    }
    assert result["canonicalization"]=={
        "input_selection_count":2,
        "canonical_selection_count":2,
        "exact_semantic_duplicates_collapsed":0,
        "candidate_reuse_allowed_for_distinct_claims":True,
    }


def test_guard_collapses_exact_semantic_duplicates_conservatively():
    plan={"candidates":[{
        "candidate_id":"pdf:p2:0010",
        "source_layer":"AUTHORITATIVE_PDF_TEXT",
        "candidate_kind":"TEXT_SPAN",
        "page_number":2,
        "pdf_bbox":None,
        "raw_source_form":"exact source",
        "display_text":"exact source",
        "derivation":{"processor":"test"},
    }]}
    result=guard_and_canonicalize(plan,{"selections":[
        {
            "candidate_id":"pdf:p2:0010",
            "type":"result",
            "normalized_form":"same claim",
            "confidence":0.95,
        },
        {
            "candidate_id":"pdf:p2:0010",
            "type":"result",
            "normalized_form":"same claim",
            "confidence":0.70,
        },
    ]})
    assert result["status"]=="PASS_GROUNDED"
    evidence=result["canonical_result"]["evidence"]
    assert len(evidence)==1
    assert evidence[0]["confidence"]==0.70
    assert result["canonicalization"]["exact_semantic_duplicates_collapsed"]==1
