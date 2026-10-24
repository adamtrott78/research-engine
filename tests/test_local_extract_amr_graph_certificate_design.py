from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "document_intelligence" / "local-extract-amr-graph-certificate-design.v1.json"


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_amr_validation_is_external_and_teacher_blind() -> None:
    row = _load()
    selected = row["selected_validation"]
    assert selected["id"] == "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1"
    assert selected["graph_similarity"]["teacher_blind"] is True
    assert row["hard_boundaries"]["extract_teacher_accessed"] is False
    assert row["hard_boundaries"]["amr_certificate_used_in_extract"] is False


def test_parser_identity_and_bio_gate_are_frozen() -> None:
    selected = _load()["selected_validation"]
    model = selected["parser_model"]
    assert model["release_asset_id"] == 57066548
    assert model["release_asset_size_bytes"] == 1506463657
    assert model["published_md5"] == "5633e5e6f8a4f4398ce64abab71c473a"
    assert model["base_model"] == "facebook/bart-large"
    assert model["tokenizer"] == {
        "repository": "facebook/bart-large",
        "revision": "22fa33834dccc11df99c4fc5fcc96c67f806dfdb",
        "license": "Apache-2.0",
        "acquisition": "Download tokenizer files from this exact Hugging Face revision into an ephemeral local directory and pass that directory as tok_name_or_path to amrlib. Network resolution from moving main is forbidden.",
    }
    assert model["inference"] == {
        "device": "cuda:0",
        "batch_size": 8,
        "num_beams": 4,
        "add_metadata": False,
        "deterministic_seed": 1729,
    }
    bio = selected["external_validation_sets"][0]
    assert bio["expected_amr_count"] == 500
    assert bio["mirror_commit"] == "802923ae6c7376a9930e93ce52cd6b9faf43c411"
    assert bio["mirror_git_blob_sha"] == "7e816eb86104fa6c07d8eff9f9c6abc84bae2898"
    assert bio["leakage_audit"]["result"] == "PASS"
    assert bio["predeclared_gate"] == {
        "corpus_smatch_f1_min": 0.60,
        "valid_prediction_fraction_min": 0.98,
    }


def test_smatch_reseeding_and_spark_gate_are_frozen() -> None:
    selected = _load()["selected_validation"]
    smatch = selected["amrlib"]["smatch"]
    assert smatch["release_tag"] == "v1.0.4"
    assert smatch["release_commit"] == "1bcd49cc8535056aa1c941763580fc5a572f1204"
    assert smatch["standard_iteration_num"] == 5
    assert "Call the captured original random.seed(seed) exactly once" in smatch["deterministic_reseed_rule"]
    assert "replace smatch.random.seed with a no-op" in smatch["deterministic_reseed_rule"]
    assert selected["graph_similarity"]["threshold_grid"] == [
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ]
    spark = selected["external_validation_sets"][1]
    assert spark["heldout_gate"] == {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }


def test_next_phase_does_not_authorize_extract_or_semantic_control() -> None:
    row = _load()
    assert row["hard_boundaries"]["production_routing_changed"] is False
    assert row["hard_boundaries"]["wave7b"] is False
    assert row["hard_boundaries"]["parser_training_authorized"] is False
    assert row["independent_review_boundary"]["authority_over_extract"] is False
    assert row["coordination_boundary"]["authority_over_extract"] is False
    assert row["hard_boundaries"]["semantic_control_or_ontology_in_extract"] is False
    assert row["hard_boundaries"]["rdf_or_ontology_output"] is False
