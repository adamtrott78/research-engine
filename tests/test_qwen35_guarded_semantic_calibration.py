import json
from pathlib import Path

from research_engine.orchestration import ExecutorTier, JobType, ResearchJob
from research_engine.qwen_guarded_semantic import evaluate_calibration, guard_qwen_semantic_result

ROOT=Path(__file__).parents[1]
CONTRACT=ROOT/"benchmarks"/"document_intelligence"/"qwen35-guarded-semantic-calibration-contract.v1.json"

def _job(job_type: JobType,pages,evidence=None):
    context={"pages":pages}
    if job_type==JobType.VERIFY:
        context["authoritative_evidence"]=evidence or []
        context["verification_scope"]={"cited_page_numbers":[pages[0]["page_number"]],"page_radius":0,"supplied_page_numbers":[pages[0]["page_number"]]}
    return ResearchJob(id=f"job-{job_type.lower()}",run_id="run",job_type=job_type,executor_tier=ExecutorTier.STANDARD,status="CALIBRATION",priority=0,idempotency_key="x",payload={"_research_context":context},result=None,depends_on=[],attempt_count=0,max_attempts=1,lease_owner=None,lease_expires_at=None)

def test_contract_is_two_calls_one_boot_and_nonmaterializing():
    d=json.loads(CONTRACT.read_text())
    assert d["source_selection"]["jobs_exactly"]==2
    assert d["source_selection"]["job_types"]==["EXTRACT","VERIFY"]
    assert d["qwen_execution"]["one_model_boot"] is True
    assert d["qwen_execution"]["inference_calls_exactly"]==2
    assert d["qwen_execution"]["package_install_or_download_during_gate"] is False
    assert d["wave3_opportunistic"]["extra_certification_calls"]==0
    assert d["promotion_boundary"]["production_routing_changed"] is False
    assert d["promotion_boundary"]["scientific_state_materialized"] is False
    assert d["promotion_boundary"]["semantic_materialization_invoked"] is False

def test_extract_guard_accepts_only_quote_grounded_page_bounded_result():
    job=_job(JobType.EXTRACT,[{"page_number":2,"text":"The measured gain was 17 percent in the target condition."}])
    good={"evidence":[{"id":"e1","type":"RESULT","normalized_form":"gain 17 percent","raw_source_form":"measured gain was 17 percent","page_number":2,"confidence":0.91}]}
    assert guard_qwen_semantic_result(job,good)["status"]=="PASS_GUARDED"
    bad=json.loads(json.dumps(good)); bad["evidence"][0]["raw_source_form"]="the gain was 99 percent"
    result=guard_qwen_semantic_result(job,bad)
    assert result["status"]=="FALLTHROUGH_CODEX"
    assert any("RAW_QUOTE_NOT_ON_CITED_PAGE" in x for x in result["reasons"])

def test_extract_guard_rejects_fifth_page():
    job=_job(JobType.EXTRACT,[{"page_number":i,"text":"quoted fact"} for i in range(1,6)])
    result=guard_qwen_semantic_result(job,{"evidence":[]})
    assert result["status"]=="FALLTHROUGH_CODEX"
    assert "EXTRACT_PAGE_BOUND_VIOLATION:5" in result["reasons"]

def test_verify_guard_requires_radius_zero_single_page_and_known_ids():
    evidence=[{"id":"ev-1","raw_source_form":"fact","page_number":3}]
    job=_job(JobType.VERIFY,[{"page_number":3,"text":"fact"}],evidence)
    good={"verified":True,"item_id":"item","accepted_evidence_ids":["ev-1"],"issues":[]}
    assert guard_qwen_semantic_result(job,good)["status"]=="PASS_GUARDED"
    bad={**good,"accepted_evidence_ids":["ev-404"]}
    result=guard_qwen_semantic_result(job,bad)
    assert result["status"]=="FALLTHROUGH_CODEX"
    assert any("UNKNOWN_ACCEPTED_EVIDENCE_IDS" in x for x in result["reasons"])

def test_evaluation_fails_closed_on_codex_disagreement():
    extract=_job(JobType.EXTRACT,[{"page_number":1,"text":"alpha evidence"}])
    verify=_job(JobType.VERIFY,[{"page_number":2,"text":"beta"}],[{"id":"ev","raw_source_form":"beta","page_number":2}])
    plan={"source_run_id":"run","jobs":[{"job":{"id":extract.id,"run_id":"run","job_type":"EXTRACT","payload":extract.payload},"context_page_numbers":[1],"context_chars":14},{"job":{"id":verify.id,"run_id":"run","job_type":"VERIFY","payload":verify.payload},"context_page_numbers":[2],"context_chars":4}]}
    qwen={"jobs":[{"job_id":extract.id,"parsed":{"evidence":[{"id":"e","type":"RESULT","normalized_form":"alpha","raw_source_form":"alpha evidence","page_number":1,"confidence":0.9}]}},{"job_id":verify.id,"parsed":{"verified":True,"item_id":"item","accepted_evidence_ids":["ev"],"issues":[]}}]}
    codex={"jobs":[{"job_id":extract.id,"result":{"evidence":[{"id":"c","type":"RESULT","normalized_form":"alpha","raw_source_form":"alpha evidence","page_number":1,"confidence":0.9}]}},{"job_id":verify.id,"result":{"verified":False,"item_id":"item","accepted_evidence_ids":[],"issues":["not supported"]}}]}
    result=evaluate_calibration(plan,qwen,codex)
    assert result["decision"]=="FAIL_CLOSED_RETAIN_CODEX"
    assert result["guarded_jobs_passed"]==2
    assert result["codex_agreement_jobs_passed"]==1

def test_runner_forbids_install_and_keeps_c3_out():
    runner=(ROOT/"deploy"/"run-qwen35-guarded-semantic-calibration.sh").read_text()
    assert "pip install" not in runner
    assert "PACKAGE_INSTALL_DOWNLOAD_WORK=FORBIDDEN_AND_SKIPPED" in runner
    assert "QWEN_CALLS=2" in runner
    assert "C3_CALLS=0" in runner
    assert "SemanticResultMaterializer" not in runner
    visual=(ROOT/"benchmarks"/"document_intelligence"/"run_qwen35_visual_benchmark.py").read_text()
    assert '("figure6_first", args.figure6, FIGURE6_PROMPT, False)' in visual
    assert '("figure3_warm", args.figure3, FIGURE3_PROMPT, True)' in visual
    assert '("figure6_warm_repeat", args.figure6, FIGURE6_PROMPT, True)' in visual


def test_postgres_is_boot_persistent_and_runner_self_heals_db_only():
    compose=(ROOT/"docker-compose.yml").read_text()
    assert "db:\n    restart: unless-stopped\n" in compose

    runner=(ROOT/"deploy"/"run-qwen35-guarded-semantic-calibration.sh").read_text()
    assert "docker compose -f docker-compose.yml up -d db" in runner
    assert "POSTGRESQL=running/ready" in runner
    assert "POSTGRESQL_NOT_READY" in runner
    assert "up -d daemon" not in runner
    assert "research-engine-daemon.service" in runner
