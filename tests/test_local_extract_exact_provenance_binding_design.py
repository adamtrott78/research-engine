from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-exact-provenance-binding-microbenchmark.v1.json"
)


def _design() -> dict:
    return json.loads(DESIGN_PATH.read_text())


def test_design_identity_and_review_boundary() -> None:
    d = _design()
    assert d["design_id"] == (
        "local-extract-exact-provenance-binding-microbenchmark-v1"
    )
    assert d["status"] == "FROZEN_DESIGN_ONLY"
    assert d["selected_discriminating_experiment"]["id"] == (
        "EXACT_PROVENANCE_BINDING_MICROBENCH_V1"
    )
    assert "Independent pre-implementation review" in d["review_requirement"]


def test_exact_prior_result_and_reviewer_identity_are_frozen() -> None:
    d = _design()
    prior = d["prior_result_basis"]
    assert prior["run_id"] == 40166893479
    assert prior["job_id"] == 121089689701
    assert prior["artifact_id"] == 12588764634
    assert prior["report_sha256"] == (
        "82a71963c19b47a2e05807997353bce9294103e570173e3371985a8968b480dc"
    )
    assert prior["reviewer_comment_id"] == 6395379857
    assert prior["reviewer_disposition"] == "NO_BLOCKING_FINDING"


def test_seven_case_membership_is_unchanged() -> None:
    x = _design()["selected_discriminating_experiment"]
    false_cases = x["case_set"]["known_false_positive_cases"]
    true_cases = x["case_set"]["known_true_support_cases"]
    assert [c["case_id"] for c in false_cases] == [
        "FP_TRAINING_PRETRAINING_NO_LABELS",
        "FP_TESTING_GENERIC_DOMAIN_SHIFT",
        "FP_ACQUISITION_GENERIC_RF_SAMPLING",
    ]
    assert [c["case_id"] for c in true_cases] == [
        "TP_TRAINING_SOURCE_LABELS",
        "TP_TESTING_TARGET_DOMAIN",
        "TP_ACQUISITION_B210_45MSPS",
        "TP_PERSISTENT_FAILURE_FOLLOWUP",
    ]
    assert [c["evidence_proposition_ids"] for c in false_cases] == [
        ["page:p2:s33"],
        ["page:p1:s10"],
        ["page:p1:s09"],
    ]
    assert [c["evidence_proposition_ids"] for c in true_cases] == [
        ["page:p4:s15"],
        ["page:p2:s35"],
        ["page:p2:s17"],
        ["page:p6:s09"],
    ]


def test_opaque_evidence_id_is_content_independent_and_runtime_labels_hidden() -> None:
    x = _design()["selected_discriminating_experiment"]
    case_set = x["case_set"]
    assert "literal opaque handle E0" in case_set[
        "opaque_evidence_id_contract"
    ]
    shared = x["shared_prompt_and_evidence_contract"]
    assert shared["evidence_json_serialization"]["exact_visible_object"] == {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "evidence_id": "E0",
        "text": "<exact authoritative proposition text>",
    }
    hidden = set(shared["model_hidden_fields"])
    for required in {
        "obligation id",
        "case id",
        "expected_posthoc",
        "teacher atom ids",
        "legacy anchors",
        "proposition_id",
        "page number",
        "retrieval rank",
        "TRUE/FALSE case role",
        "prior raw model outputs",
        "prior verifier outcomes",
    }:
        assert required in hidden


def test_no_regenerated_source_text_is_part_of_either_proof_contract() -> None:
    x = _design()["selected_discriminating_experiment"]
    shared = x["shared_prompt_and_evidence_contract"]
    forbidden = set(shared["forbidden_model_proof_fields"])
    assert {
        "supporting_quotes",
        "verbatim_span",
        "source_quote",
        "source_text",
        "page_number",
        "proposition_id",
        "character_offsets",
    } <= forbidden

    arm_a = x["arm_a_deterministic_attachment"]
    arm_b = x["arm_b_opaque_pointer_selection"]
    assert set(arm_a["model_output_contract"]) == {
        "candidate_status",
        "missing_aspects",
        "rationale",
    }
    assert set(arm_b["model_output_contract"]) == {
        "candidate_status",
        "supporting_evidence_ids",
        "missing_aspects",
        "rationale",
    }
    assert "supporting_quotes" not in arm_a["prompt_contract"]["user_template"]
    assert "supporting_quotes" not in arm_b["prompt_contract"]["user_template"]
    assert "verbatim_span" not in arm_a["prompt_contract"]["user_template"]
    assert "verbatim_span" not in arm_b["prompt_contract"]["user_template"]


def test_arm_a_is_semantic_status_plus_deterministic_provenance_attachment() -> None:
    x = _design()["selected_discriminating_experiment"]
    arm = x["arm_a_deterministic_attachment"]
    assert arm["id"] == (
        "TRI_STATE_DETERMINISTIC_PROVENANCE_ATTACHMENT_V1"
    )
    guard = "\n".join(arm["deterministic_guard"])
    assert "runtime attaches E0" in guard
    assert "hidden one-to-one mapping" in guard
    assert "No generated source substring" in guard


def test_arm_b_requires_exact_opaque_pointer_contract() -> None:
    x = _design()["selected_discriminating_experiment"]
    arm = x["arm_b_opaque_pointer_selection"]
    assert arm["id"] == "TRI_STATE_OPAQUE_EVIDENCE_POINTER_V1"
    guard = "\n".join(arm["deterministic_guard"])
    assert 'exactly ["E0"]' in guard
    assert "supporting_evidence_ids=[]" in guard
    assert "Unknown IDs" in guard


def test_model_and_source_identities_are_unchanged() -> None:
    x = _design()["selected_discriminating_experiment"]
    frozen = x["frozen_inputs"]
    assert frozen["teacher_paper"] == "arxiv:2403.04036"
    assert frozen["teacher_pdf_sha256"] == (
        "1097bdfabf6cd1b1e26ff9df58b6bb810e4bf74fbd3890ad20e86316d2ee0c91"
    )
    assert frozen["corpus_fixture_blob"] == (
        "c472329d560df3d7dccab2bf49bf94d6805faef2"
    )
    model = frozen["local_model"]
    assert model["model_id"] == "qwen3.5-4b"
    assert model["revision"] == (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert model["temperature"] == 0
    assert model["top_p"] == 1
    assert model["seed"] == 0


def test_stateless_execution_and_one_valid_run_are_frozen() -> None:
    x = _design()["selected_discriminating_experiment"]
    isolation = x["stateless_case_isolation_contract"]
    assert isolation["total_model_calls"] == 14
    for key in (
        "few_shot_examples",
        "prior_case_outputs_visible",
        "prior_case_prompts_visible",
        "prior_validity_run_outputs_visible",
        "extra_paper_context_visible",
        "extra_corpus_context_visible",
        "cross_case_conversation_state",
        "chat_history_reused",
        "kv_cache_reused_across_cases",
        "prompt_tuning_after_review",
        "retry_on_bad_model_output",
        "retry_on_scientific_failure",
    ):
        assert isolation[key] is False
    assert "Exactly one valid frozen microbenchmark execution" in x[
        "stopping_rule"
    ]


def test_completeness_requires_valid_exact_provenance() -> None:
    x = _design()["selected_discriminating_experiment"]
    sim = x["completeness_simulation"]
    assert sim["evidence_set_mutation"] is False
    assert "exact provenance attachment is valid" in sim["rule"]
    assert sim["baseline_false_complete_state"][
        "required_outcome_for_any_safe_arm"
    ] == {
        "local_complete": False,
        "fallback_required": True,
    }
    assert sim["true_support_state"][
        "required_outcome_for_any_passing_arm"
    ] == {
        "local_complete": True,
        "fallback_required": False,
    }


def test_preservation_and_sealed_boundaries_remain_closed() -> None:
    d = _design()
    x = d["selected_discriminating_experiment"]
    p = x["preservation_controls"]
    assert p["selection_mutation_allowed"] is False
    assert len(p["prior_supported_atom_ids"]) == 9
    assert len(p["frozen_control_atom_ids"]) == 7

    b = d["frozen_scientific_boundaries"]
    assert b["evidence_set_mutation_allowed"] is False
    assert b["preservation_aware_admission_implemented"] is False
    assert b["regenerated_quote_or_span_proof_allowed"] is False
    assert b["nli_certificate_reopened"] is False
    assert b["predicate_argument_certificate_reopened"] is False
    assert b["amr_graph_certificate_reopened"] is False
    assert b["sealed_acceptance_benchmark_accessed"] is False
    assert b["acceptance_candidate_frozen"] is False
    assert b["production_routing_changed"] is False
    assert b["wave7b_started"] is False


def test_outcome_partition_has_no_acceptance_authority() -> None:
    x = _design()["selected_discriminating_experiment"]
    assert x["outcome_precedence"] == [
        "INVALID_EXECUTION",
        "BOTH_EXACT_PROVENANCE_INTERFACES_CONFIRMED",
        "DETERMINISTIC_ATTACHMENT_ONLY_CONFIRMED",
        "OPAQUE_POINTER_ONLY_CONFIRMED",
        "EXACT_PROVENANCE_BINDING_PREMISE_REJECTED",
    ]
    auth = x["interpretation_authority"]
    assert auth["acceptance_freeze_authority"] == "NONE"
    assert "preservation-aware admission" in auth[
        "on_BOTH_EXACT_PROVENANCE_INTERFACES_CONFIRMED"
    ]
