from __future__ import annotations

import json
from pathlib import Path


DESIGN = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "document_intelligence"
    / "local-extract-obligation-validity-microbenchmark.v1.json"
)


def _design() -> dict:
    return json.loads(DESIGN.read_text())


def test_design_is_development_only_and_preimplementation() -> None:
    design = _design()
    assert design["status"] == "FROZEN_DESIGN_ONLY"
    assert design["selected_discriminating_experiment"]["id"] == (
        "OBLIGATION_VALIDITY_PRECISION_MICROBENCH_V1"
    )
    assert design["interpretation_authority"]["implementation_authority_before_review"] == "NONE"
    boundaries = design["frozen_scientific_boundaries"]
    assert boundaries["development_only"] is True
    assert boundaries["sealed_acceptance_benchmark_accessed"] is False
    assert boundaries["acceptance_candidate_frozen"] is False
    assert boundaries["evidence_set_mutation_allowed"] is False


def test_hybrid_lineage_is_preserved_and_retrieval_is_not_reopened() -> None:
    design = _design()
    lineage = design["architecture_lineage"]
    assert lineage["preserved_positive_parent"] == "LOCAL_FIRST_ALLOCATION_FALLBACK_HYBRID_V1"
    assert lineage["parent_result"] == "HYBRID_COMPLEMENTARITY_REALIZED"
    boundaries = design["frozen_scientific_boundaries"]
    assert boundaries["retrieval_changed"] is False
    assert boundaries["retrieval_depth_changed"] is False
    assert boundaries["query_generation_changed"] is False
    assert boundaries["reranker_changed"] is False
    assert boundaries["larger_model_introduced"] is False


def test_materially_distinct_validity_families_are_compared() -> None:
    design = _design()
    families = {row["id"]: row["role_in_diagnostic"] for row in design["mechanism_family_comparison"]}
    assert families["A_PROOF_CARRYING_EXPLICIT_SLOT_GROUNDING"] == "PRIMARY_ARM"
    assert families["B_HOLISTIC_TRI_STATE_SOURCE_GROUNDED_VERIFIER"] == "COMPARATOR_ARM"
    assert families["C_PRESERVATION_AWARE_EVIDENCE_SET_ADMISSION"] == (
        "DEFERRED_NEXT_PHASE_IF_VALIDITY_PASSES"
    )
    assert families["D_VALIDATED_OBLIGATION_COMPLETENESS_CONTROLLER"] == (
        "SIMULATED_WITH_NO_EVIDENCE_SET_MUTATION"
    )


def test_case_set_contains_exact_false_positive_and_true_support_pairs() -> None:
    exp = _design()["selected_discriminating_experiment"]
    cases = exp["case_set"]
    false_pairs = {
        (row["obligation_id"], tuple(row["evidence_proposition_ids"]))
        for row in cases["known_false_positive_cases"]
    }
    true_pairs = {
        (row["obligation_id"], tuple(row["evidence_proposition_ids"]))
        for row in cases["known_true_support_cases"]
    }
    assert false_pairs == {
        ("method_training_label_source", ("page:p2:s33",)),
        ("method_testing_domain", ("page:p1:s10",)),
        ("acquisition_receiver_sampling_configuration", ("page:p1:s09",)),
    }
    assert true_pairs == {
        ("method_training_label_source", ("page:p4:s15",)),
        ("method_testing_domain", ("page:p2:s35",)),
        ("acquisition_receiver_sampling_configuration", ("page:p2:s17",)),
        ("persistent_failure_followup", ("page:p6:s09",)),
    }


def test_shallow_exposure_is_frozen_and_no_retrieval_experiment_is_needed() -> None:
    exposure = _design()["selected_discriminating_experiment"]["case_set"][
        "shallow_exposure_evidence"
    ]
    assert exposure["method_training_label_source"] == [
        {"proposition_id": "page:p4:s15", "rank": 1},
        {"proposition_id": "page:p2:s34", "rank": 2},
    ]
    assert exposure["method_testing_domain"] == [
        {"proposition_id": "page:p2:s35", "rank": 1}
    ]
    assert exposure["acquisition_receiver_sampling_configuration"] == [
        {"proposition_id": "page:p2:s17", "rank": 1}
    ]
    assert exposure["persistent_failure_followup"] == [
        {"proposition_id": "page:p6:s09", "rank": 1}
    ]


def test_both_arms_are_discrete_source_grounded_not_score_threshold_certificates() -> None:
    exp = _design()["selected_discriminating_experiment"]
    a = exp["arm_a_proof_carrying_slot_grounding"]
    b = exp["arm_b_holistic_tri_state_verifier"]
    for arm in (a, b):
        text = json.dumps(arm).lower()
        assert "teacher atom" in text
        assert "threshold" in text
        assert "exact" in text
        assert "similarity score" in text
    assert set(a["model_output_contract"]["candidate_status"]) == {
        "SUPPORTED_EXPLICITLY",
        "NOT_SUPPORTED",
        "INSUFFICIENT_OR_AMBIGUOUS",
    }
    assert set(b["model_output_contract"]["candidate_status"]) == {
        "SUPPORTED_EXPLICITLY",
        "NOT_SUPPORTED",
        "INSUFFICIENT_OR_AMBIGUOUS",
    }


def test_fail_closed_completeness_states_are_frozen() -> None:
    sim = _design()["selected_discriminating_experiment"]["completeness_simulation"]
    assert sim["evidence_set_mutation"] is False
    assert sim["baseline_false_complete_state"]["required_outcome_for_any_safe_arm"] == {
        "local_complete": False,
        "fallback_required": True,
    }
    assert sim["true_support_state"]["required_outcome_for_any_passing_arm"] == {
        "local_complete": True,
        "fallback_required": False,
    }
    assert "NOT_SUPPORTED and INSUFFICIENT_OR_AMBIGUOUS both force fallback" in sim["rule"]


def test_preservation_controls_are_no_mutation_invariants() -> None:
    p = _design()["selected_discriminating_experiment"]["preservation_controls"]
    assert p["selection_mutation_allowed"] is False
    assert "prior supported atom retention remains 9/9 because admission is disabled" in p[
        "required_invariants"
    ]
    assert "frozen control retention remains 7/7 because admission is disabled" in p[
        "required_invariants"
    ]
    assert "page:p2:s16 remains present; no e6.c admission swap is attempted" in p[
        "required_invariants"
    ]


def test_precision_first_gate_has_no_partial_promotion() -> None:
    gate = _design()["selected_discriminating_experiment"]["scientific_gate"]
    assert gate["per_arm_precision_first_pass"] == [
        "all_7_cases_arm_case_contract_pass",
        "reject_or_abstain_all_3_known_false_positives",
        "accept_all_4_known_true_supports",
        "zero_non_exact_supporting_quotes_or_spans",
        "baseline_false_complete_state_returns_incomplete_and_fallback",
        "true_support_state_returns_complete_without_fallback",
        "evidence_set_identity_unchanged",
        "prior_supported_retention_9_of_9",
        "frozen_control_retention_7_of_7",
    ]
    assert gate["no_partial_promotion"].startswith(
        "Any malformed/guard-failed case"
    )


def test_model_identity_is_exact_and_no_new_download_is_implied() -> None:
    model = _design()["selected_discriminating_experiment"]["frozen_inputs"]["local_model"]
    assert model == {
        "model_id": "qwen3.5-4b",
        "repository": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "model_object_id": "6e28e2560c07e7dabab92c55dbfb6941d5a7a75c8068bdc7a98115d8dc62072c",
        "runtime_tree_sha256": "b0cbb4efbe8673bdb421bad8036d29864bffe0687a9a654bbd95f54544659e63",
        "runtime_manifest_sha256": "f67ff9a26690c55930232c915e59e826e3cca2dd89012de9e35a852032b38f74",
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
    }


def test_rejected_certificate_families_remain_closed() -> None:
    b = _design()["frozen_scientific_boundaries"]
    assert b["rejected_pairwise_score_certificate_family_reopened"] is False
    assert b["nli_certificate_reopened"] is False
    assert b["predicate_argument_certificate_reopened"] is False
    assert b["amr_graph_certificate_reopened"] is False


def test_outcomes_do_not_authorize_acceptance_freeze() -> None:
    design = _design()
    exp = design["selected_discriminating_experiment"]
    assert exp["outcome_precedence"] == [
        "INVALID_EXECUTION",
        "BOTH_VALIDITY_MECHANISMS_CONFIRMED",
        "SLOT_GROUNDING_ONLY_CONFIRMED",
        "HOLISTIC_VERIFIER_ONLY_CONFIRMED",
        "VALIDITY_PREMISE_REJECTED",
    ]
    assert design["interpretation_authority"]["acceptance_freeze_authority"] == "NONE"


def test_slot_stage_enum_and_required_relations_are_frozen() -> None:
    exp = _design()["selected_discriminating_experiment"]
    arm = exp["arm_a_proof_carrying_slot_grounding"]
    assert arm["model_output_contract"]["stage_role"] == [
        "PRETRAINING",
        "CLASSIFIER_TRAINING",
        "TESTING",
        "EVALUATION",
        "ACQUISITION",
        "FAILURE_ANALYSIS",
        "RESULT_INTERPRETATION",
        "OTHER",
        "AMBIGUOUS",
    ]
    assert arm["model_output_contract"]["relation_check"] == [
        "EXPLICITLY_LINKED",
        "NOT_LINKED",
        "AMBIGUOUS",
    ]
    assert all("required_relation" in row for row in exp["runtime_obligations"])


def test_exact_preservation_atom_and_baseline_identities_are_frozen() -> None:
    p = _design()["selected_discriminating_experiment"]["preservation_controls"]
    assert p["prior_supported_atom_ids"] == [
        "e1.a",
        "e2.a",
        "e2.b",
        "e3.a",
        "e3.b",
        "e4.a",
        "e4.b",
        "e6.a",
        "e6.b",
    ]
    assert p["frozen_control_atom_ids"] == [
        "e2.a",
        "e2.b",
        "e3.a",
        "e3.b",
        "e4.a",
        "e4.b",
        "e6.b",
    ]
    assert p["baseline_identity"] == {
        "run_id": 40143476764,
        "artifact_id": 12580076512,
        "artifact_zip_sha256": "696fa26d942767844ddb24119ae98a503febac02da4e8f3dcbf2eeb021304b89",
        "report_sha256": "e1a2f61f16cae4e957b10a9c24b475d5cb481afa9381084de65a1cd9c6568d23",
    }


def test_reviewer_correction_freezes_arm_specific_visible_fields() -> None:
    exp = _design()["selected_discriminating_experiment"]
    a = exp["arm_a_proof_carrying_slot_grounding"]
    b = exp["arm_b_holistic_tri_state_verifier"]
    assert a["model_visible_obligation_fields"] == [
        "generic_information_need",
        "mandatory_slots",
        "accepted_stage_roles",
        "required_relation",
    ]
    assert b["model_visible_obligation_fields"] == [
        "generic_information_need",
        "required_relation",
    ]
    assert "fail_closed_conditions" in a["model_hidden_obligation_fields"]
    assert "fail_closed_conditions" in b["model_hidden_obligation_fields"]
    assert "mandatory_slots" in b["model_hidden_obligation_fields"]
    for arm in (a, b):
        prompt_text = json.dumps(arm["prompt_contract"]).lower()
        assert "fail_closed_conditions" not in prompt_text
        assert "case_id" in json.dumps(
            arm["prompt_contract"]["evidence_json_serialization"]["hidden_from_model"]
        )


def test_exact_prompt_templates_and_stateless_isolation_are_frozen() -> None:
    exp = _design()["selected_discriminating_experiment"]
    a = exp["arm_a_proof_carrying_slot_grounding"]["prompt_contract"]
    b = exp["arm_b_holistic_tri_state_verifier"]["prompt_contract"]
    assert a["template_id"] == "PROOF_CARRYING_EXPLICIT_SLOT_GROUNDING_PROMPT_V1"
    assert b["template_id"] == "HOLISTIC_TRI_STATE_SOURCE_GROUNDED_VERIFIER_PROMPT_V1"
    assert a["user_template"].startswith("OBLIGATION_JSON\n{obligation_json}")
    assert b["user_template"].startswith("OBLIGATION_JSON\n{obligation_json}")
    assert a["evidence_json_serialization"]["exact_visible_object"] == {
        "source_authority": "AUTHORITATIVE_PDF_TEXT",
        "text": "<exact authoritative proposition text or exact concatenation of the frozen evidence bundle texts separated by a single newline>",
    }
    assert b["evidence_json_serialization"]["exact_visible_object"] == (
        a["evidence_json_serialization"]["exact_visible_object"]
    )
    isolation = exp["stateless_case_isolation_contract"]
    assert isolation["few_shot_examples"] is False
    assert isolation["prior_case_outputs_visible"] is False
    assert isolation["extra_paper_context_visible"] is False
    assert isolation["extra_corpus_context_visible"] is False
    assert isolation["cross_case_conversation_state"] is False
    assert isolation["chat_history_reused"] is False
    assert isolation["kv_cache_reused_across_cases"] is False
    assert isolation["prompt_tuning_after_review"] is False
    assert isolation["retry_on_bad_model_output"] is False


def test_bad_model_output_fails_closed_and_counts_against_arm() -> None:
    sem = _design()["selected_discriminating_experiment"][
        "model_output_and_guard_semantics"
    ]
    bad = sem["ordinary_model_output_failure_semantics"]
    for key in (
        "malformed_json",
        "schema_missing_or_unknown_fields",
        "invalid_enum_value",
        "non_exact_quote_or_span_attempt",
        "supported_with_missing_required_binding_or_missing_aspect",
    ):
        assert bad[key]["effective_candidate_status"] == (
            "INSUFFICIENT_OR_AMBIGUOUS"
        )
        assert bad[key]["arm_case_contract_pass"] is False
        assert bad[key]["invalid_execution"] is False
    assert bad["contradictory_or_ambiguous_relation_or_stage"][
        "arm_case_contract_pass"
    ] is True
    assert sem["parser_contract"]["repair"] == "NONE"
    assert sem["parser_contract"]["second_parse_attempt"] is False
    assert sem["parser_contract"]["model_retry"] is False


def test_invalid_execution_is_integrity_only_not_bad_output() -> None:
    exp = _design()["selected_discriminating_experiment"]
    sem = exp["model_output_and_guard_semantics"]
    invalid = " ".join(sem["invalid_execution_reserved_for"]).lower()
    ordinary = " ".join(sem["invalid_execution_not_caused_by"]).lower()
    assert "wrong frozen source/corpus/model/runtime identity" in invalid
    assert "leakage" in invalid
    assert "implementation failure" in invalid
    assert "evidence-set mutation" in invalid
    for term in (
        "malformed verifier json",
        "non-exact quote/span attempt",
        "missing slot binding",
        "explicit ambiguity",
        "scientifically wrong verifier verdict",
    ):
        assert term in ordinary
    assert "Ordinary bad verifier outputs never cause INVALID_EXECUTION" in (
        exp["outcome_partition"]["INVALID_EXECUTION"]
    )
    assert "No rerun authority arises from ordinary verifier output failure" in (
        exp["stopping_rule"]
    )


def test_review_correction_preserves_exact_science() -> None:
    design = _design()
    correction = design["review_correction"]
    assert correction["corrected_from_commit"] == (
        "b939ff0bbb52024f55b2f6e719aef331f6286d06"
    )
    assert correction["corrected_from_design_blob"] == (
        "16100c540729a3c73a03778490b63846e6865030"
    )
    assert correction["reviewer_disposition"] == "BLOCKING_FINDING"
    assert correction["implementation_started"] is False
    assert correction["scientific_run_executed"] is False
    assert correction["unchanged_science"] == [
        "3 false-positive cases",
        "4 true-support cases",
        "Qwen3.5-4B identity",
        "source/corpus identities",
        "both mechanism families",
        "no evidence-set mutation",
        "all-or-nothing precision-first gate",
        "completeness simulation",
        "one valid run only",
        "acceptance_candidate_frozen=false",
        "sealed holdout untouched",
        "LOCAL_FIRST_ALLOCATION_FALLBACK_HYBRID_V1 positive lineage",
    ]
