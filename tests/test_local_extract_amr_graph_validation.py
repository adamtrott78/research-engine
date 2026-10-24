from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

from research_engine.local_extract_amr_graph_validation import (
    deterministic_smatch_counts,
    git_blob_sha1_bytes,
    smatch_seed,
    validation_verdict,
)


class _FakeRandom:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def seed(self, value=None) -> None:
        self.calls.append(value)


class _FakeSmatch:
    def __init__(self) -> None:
        self.random = _FakeRandom()
        self.match_triple_dict = {"stale": 1}
        self.seed_function_seen = None

    def get_amr_match(self, left: str, right: str):
        self.random.seed()
        self.random.seed()
        self.seed_function_seen = self.random.seed
        assert left == "(a / alpha)"
        assert right == "(b / beta)"
        return 3, 4, 5


def test_smatch_seed_uses_first_64_bits_of_sha256() -> None:
    identity = "topic\0alpha\0beta"
    expected = int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8],
        "big",
    )
    assert smatch_seed(identity) == expected


def test_deterministic_smatch_seeds_once_and_suppresses_internal_reseeds() -> None:
    fake = _FakeSmatch()
    original_seed = fake.random.seed

    counts = deterministic_smatch_counts(
        "(a / alpha)",
        "(b / beta)",
        pair_identity="pair-123",
        smatch_module=fake,
    )

    assert counts == (3, 4, 5)
    assert fake.random.calls == [smatch_seed("pair-123")]
    assert fake.random.seed == original_seed
    assert fake.match_triple_dict == {}


def test_git_blob_identity_matches_git_object_formula() -> None:
    payload = b"hello\n"
    expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    assert git_blob_sha1_bytes(payload) == expected


def test_no_safe_spark_threshold_is_rejected_even_when_bio_passes() -> None:
    result = validation_verdict(
        bio_metrics={"f1": 0.75, "valid_prediction_fraction": 1.0},
        spark_threshold=None,
        spark_heldout={
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
        },
        bio_gate={
            "corpus_smatch_f1_min": 0.60,
            "valid_prediction_fraction_min": 0.98,
        },
        spark_gate={
            "precision_min": 0.90,
            "recall_min": 0.05,
            "false_positive_rate_max": 0.02,
        },
    )
    assert result["amr_graph_certificate_decision"] == (
        "AMR_GRAPH_CERTIFICATE_REJECTED"
    )
    assert result["bio_amr_pass"] is True
    assert result["spark_pass"] is False


def test_gate_logic_validated_partial_and_rejected() -> None:
    bio_gate = {
        "corpus_smatch_f1_min": 0.60,
        "valid_prediction_fraction_min": 0.98,
    }
    spark_gate = {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }
    threshold = {"threshold": 0.65}
    good_bio = {"f1": 0.70, "valid_prediction_fraction": 0.99}
    bad_bio = {"f1": 0.59, "valid_prediction_fraction": 1.0}
    good_spark = {
        "precision": 0.91,
        "recall": 0.06,
        "false_positive_rate": 0.01,
    }
    bad_spark = {
        "precision": 0.89,
        "recall": 0.06,
        "false_positive_rate": 0.01,
    }

    assert validation_verdict(
        bio_metrics=good_bio,
        spark_threshold=threshold,
        spark_heldout=good_spark,
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == "AMR_GRAPH_CERTIFICATE_VALIDATED"

    assert validation_verdict(
        bio_metrics=bad_bio,
        spark_threshold=threshold,
        spark_heldout=good_spark,
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == "AMR_GRAPH_CERTIFICATE_PARTIAL"

    assert validation_verdict(
        bio_metrics=bad_bio,
        spark_threshold=threshold,
        spark_heldout=bad_spark,
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == "AMR_GRAPH_CERTIFICATE_REJECTED"
