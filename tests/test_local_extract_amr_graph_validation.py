from __future__ import annotations

import hashlib
import pathlib

from research_engine.local_extract_amr_graph_validation import (
    deterministic_smatch_counts,
    pair_seed,
    score_spark_pairs,
    validation_verdict,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "lambda-ci-local-extract-amr-graph-certificate-validation.yml"
)


class _FakeRandom:
    def __init__(self) -> None:
        self.seed_calls: list[int | None] = []
        self.state = 0

    def seed(self, value: int | None = None) -> None:
        self.seed_calls.append(value)
        self.state = 0 if value is None else int(value)

    def randint(self, lower: int, upper: int) -> int:
        self.state = (
            6364136223846793005 * self.state + 1442695040888963407
        ) & ((1 << 64) - 1)
        return lower + (self.state % (upper - lower + 1))


class _FakeSmatch:
    def __init__(self) -> None:
        self.random = _FakeRandom()
        self.iteration_num = 11
        self.match_triple_dict: dict[tuple[int, ...], int] = {}
        self.observed_iteration_nums: list[int] = []

    def get_amr_match(
        self,
        test_graph: str,
        gold_graph: str,
    ) -> tuple[int, int, int]:
        del test_graph, gold_graph
        self.observed_iteration_nums.append(self.iteration_num)
        self.random.seed()
        first = self.random.randint(1, 5)
        self.random.seed()
        second = self.random.randint(1, 5)
        return first + second, 10, 10

    @staticmethod
    def compute_f(
        match_num: int,
        test_num: int,
        gold_num: int,
    ) -> tuple[float, float, float]:
        precision = match_num / test_num if test_num else 0.0
        recall = match_num / gold_num if gold_num else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, recall, f1


def test_pair_seed_is_first_64_bits_of_sha256() -> None:
    identity = "topic-x\\0left\\0right"
    expected = int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8],
        "big",
    )
    assert pair_seed(identity) == expected


def test_deterministic_smatch_seeds_once_and_blocks_internal_reseed() -> None:
    fake = _FakeSmatch()
    identity = "pair-identity"
    result = deterministic_smatch_counts(
        "(a / alpha)",
        "(b / beta)",
        pair_identity=identity,
        smatch_module=fake,
    )

    assert result[1:] == (10, 10)
    assert fake.random.seed_calls == [pair_seed(identity)]
    assert fake.observed_iteration_nums == [5]
    assert fake.iteration_num == 11
    assert fake.random.seed.__self__ is fake.random


def test_spark_invalid_graph_scores_zero() -> None:
    fake = _FakeSmatch()
    rows = [
        {
            "pair_id": "p1",
            "topic": "topic",
            "left": "left",
            "right": "right",
            "label": 1,
        }
    ]
    labels, scores, diagnostics = score_spark_pairs(
        rows,
        graph_by_text={
            "left": "(a / alpha)",
            "right": None,
        },
        smatch_module=fake,
    )
    assert labels == [1]
    assert scores == [0.0]
    assert diagnostics["invalid_graph_text_count"] == 1
    assert diagnostics["valid_graph_text_fraction"] == 0.5


def test_spark_seed_uses_canonical_topic_and_text_identity() -> None:
    fake = _FakeSmatch()
    rows = [
        {
            "pair_id": "already-hashed-id",
            "topic": "topic",
            "left": "left",
            "right": "right",
            "label": 1,
        }
    ]
    score_spark_pairs(
        rows,
        graph_by_text={
            "left": "(a / alpha)",
            "right": "(b / beta)",
        },
        smatch_module=fake,
    )
    assert fake.random.seed_calls == [
        pair_seed("topic\\0left\\0right")
    ]


def test_validation_verdict_obeys_frozen_gate_logic() -> None:
    bio_gate = {
        "corpus_smatch_f1_min": 0.60,
        "valid_prediction_fraction_min": 0.98,
    }
    spark_gate = {
        "precision_min": 0.90,
        "recall_min": 0.05,
        "false_positive_rate_max": 0.02,
    }
    bio_pass = {
        "corpus_smatch_f1": 0.70,
        "valid_prediction_fraction": 1.0,
    }
    spark_pass = {
        "precision": 0.95,
        "recall": 0.10,
        "false_positive_rate": 0.01,
    }

    assert validation_verdict(
        bio_metrics=bio_pass,
        threshold_selected=True,
        spark_heldout=spark_pass,
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == (
        "AMR_GRAPH_CERTIFICATE_VALIDATED"
    )

    assert validation_verdict(
        bio_metrics=bio_pass,
        threshold_selected=True,
        spark_heldout={
            **spark_pass,
            "precision": 0.80,
        },
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == (
        "AMR_GRAPH_CERTIFICATE_PARTIAL"
    )

    assert validation_verdict(
        bio_metrics=bio_pass,
        threshold_selected=False,
        spark_heldout=spark_pass,
        bio_gate=bio_gate,
        spark_gate=spark_gate,
    )["amr_graph_certificate_decision"] == (
        "AMR_GRAPH_CERTIFICATE_REJECTED"
    )


def test_workflow_is_exact_sha_teacher_blind_and_identity_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: lambda-research-engine-gpu" in text
    assert "timeout-minutes: 90" in text
    assert "/lambda-ci-local-extract-amr-graph-certificate-validation " in text
    assert "38b29706ea982da5a5b081a04f00c93cc6cc1e9c" in text
    assert "1bcd49cc8535056aa1c941763580fc5a572f1204" in text
    assert "22fa33834dccc11df99c4fc5fcc96c67f806dfdb" in text
    assert "802923ae6c7376a9930e93ce52cd6b9faf43c411" in text
    assert "7e816eb86104fa6c07d8eff9f9c6abc84bae2898" in text
    assert "71465f36c83cfef589facadcb014e70c0bad1b3b" in text
    assert "eb0b82a6f53b0a7d2a6288f0dff4ed2e72abf4ad" in text
    assert "local-extract-teacher" not in text
    assert "SemanticObservationV1" not in text
