from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any

from research_engine.local_extract_nli_certificate_validation import (
    average_precision,
    binary_metrics,
    build_spark_pairs,
    roc_auc,
    select_threshold,
    validation_verdict,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_scinli(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"sentence1", "sentence2", "label"}
        if not required <= fieldnames:
            raise RuntimeError(
                f"unexpected SciNLI columns: {sorted(fieldnames)}"
            )
        for row in reader:
            left = " ".join(str(row["sentence1"]).split()).strip()
            right = " ".join(str(row["sentence2"]).split()).strip()
            label = str(row["label"]).strip().lower()
            if not left or not right:
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "label": 1 if label == "entailment" else 0,
                    "original_label": label,
                }
            )
    return rows


class EntailmentScorer:
    def __init__(
        self,
        *,
        model_dir: pathlib.Path,
        entailment_label_id: int,
        max_length: int,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._entailment_label_id = entailment_label_id
        self._max_length = max_length
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_safetensors=True,
        )
        self._model.eval()

    def directional(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start:start + self._batch_size]
            premises = [row[0] for row in batch]
            hypotheses = [row[1] for row in batch]
            encoded = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[
                    :, self._entailment_label_id
                ]
            scores.extend(float(value) for value in probs.tolist())
        return scores

    def mutual(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        forward = self.directional(pairs)
        backward = self.directional(
            [(right, left) for left, right in pairs]
        )
        return [
            float(min(left, right))
            for left, right in zip(forward, backward)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--spark-json", type=pathlib.Path, required=True)
    parser.add_argument("--scinli-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-length", type=int, default=1680)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    design = _load_json(args.design)
    selected = design["selected_validation"]
    model_contract = selected["model"]
    spark_contract = selected["external_validation_sets"][0]
    scinli_contract = selected["external_validation_sets"][1]

    model_sha = _sha256(args.model_dir / "model.safetensors")
    if model_sha != model_contract["expected_model_safetensors_sha256"]:
        raise RuntimeError(
            f"model identity mismatch: {model_sha}"
        )

    config = _load_json(args.model_dir / "config.json")
    expected_labels = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    actual_labels = {
        str(key): str(value).lower()
        for key, value in (config.get("id2label") or {}).items()
    }
    if actual_labels != expected_labels:
        raise RuntimeError(
            f"unexpected NLI label map: {actual_labels}"
        )

    spark = _load_json(args.spark_json)
    pairs = build_spark_pairs(
        spark,
        negative_per_positive=int(
            spark_contract["pair_construction"][
                "negative_sampling"
            ].split("at most ")[1].split(" negatives")[0]
        ),
    )

    scorer = EntailmentScorer(
        model_dir=args.model_dir,
        entailment_label_id=int(model_contract["entailment_label_id"]),
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    calibration_pairs = [
        (str(row["left"]), str(row["right"]))
        for row in pairs["calibration"]
    ]
    heldout_pairs = [
        (str(row["left"]), str(row["right"]))
        for row in pairs["heldout"]
    ]
    calibration_labels = [
        int(row["label"]) for row in pairs["calibration"]
    ]
    heldout_labels = [
        int(row["label"]) for row in pairs["heldout"]
    ]

    calibration_scores = scorer.mutual(calibration_pairs)
    heldout_scores = scorer.mutual(heldout_pairs)

    calibration_contract = selected["calibration"]
    threshold_result = select_threshold(
        calibration_labels,
        calibration_scores,
        [float(value) for value in calibration_contract["threshold_grid"]],
        precision_min=0.95,
        recall_min=0.05,
    )

    if threshold_result is None:
        spark_heldout = {
            "threshold": None,
            "tp": 0,
            "fp": 0,
            "tn": sum(1 for label in heldout_labels if label == 0),
            "fn": sum(1 for label in heldout_labels if label == 1),
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
        }
    else:
        spark_heldout = binary_metrics(
            heldout_labels,
            heldout_scores,
            float(threshold_result["threshold"]),
        )

    scinli_rows = _load_scinli(args.scinli_csv)
    expected_rows = int(scinli_contract["expected_rows"])
    if len(scinli_rows) != expected_rows:
        raise RuntimeError(
            f"SciNLI row mismatch: {len(scinli_rows)} != {expected_rows}"
        )
    scinli_pairs = [
        (str(row["left"]), str(row["right"]))
        for row in scinli_rows
    ]
    scinli_labels = [
        int(row["label"]) for row in scinli_rows
    ]
    scinli_scores = scorer.directional(scinli_pairs)
    scinli_metrics = {
        "row_count": len(scinli_rows),
        "positive_count": sum(scinli_labels),
        "negative_count": len(scinli_labels) - sum(scinli_labels),
        "roc_auc": roc_auc(scinli_labels, scinli_scores),
        "average_precision": average_precision(
            scinli_labels,
            scinli_scores,
        ),
    }

    decision = validation_verdict(
        threshold_selected=threshold_result is not None,
        spark_heldout=spark_heldout,
        scinli=scinli_metrics,
        gates=selected["gates"],
    )

    report = {
        "schema_version": 1,
        "operation": "SCIENTIFIC_NLI_REDUNDANCY_CERTIFICATE_VALIDATION",
        "design_id": design["design_id"],
        **decision,
        "model": {
            "repository": model_contract["repository"],
            "revision": model_contract["revision"],
            "model_safetensors_sha256": model_sha,
            "config_sha256": _sha256(args.model_dir / "config.json"),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        },
        "external_data": {
            "spark": {
                "repository": spark_contract["repository"],
                "repository_commit": spark_contract["repository_commit"],
                "git_blob_sha": spark_contract["git_blob_sha"],
                "file_sha256": _sha256(args.spark_json),
            },
            "scinli": {
                "repository": scinli_contract["repository"],
                "revision": scinli_contract["revision"],
                "file": scinli_contract["file"],
                "file_sha256": _sha256(args.scinli_csv),
            },
        },
        "spark": {
            "calibration_topic_count": len(
                pairs["calibration_topics"]
            ),
            "heldout_topic_count": len(pairs["heldout_topics"]),
            "calibration_topics": pairs["calibration_topics"],
            "heldout_topics": pairs["heldout_topics"],
            "calibration_positive_count": pairs[
                "calibration_positive_count"
            ],
            "calibration_negative_count": pairs[
                "calibration_negative_count"
            ],
            "heldout_positive_count": pairs[
                "heldout_positive_count"
            ],
            "heldout_negative_count": pairs[
                "heldout_negative_count"
            ],
            "selected_threshold": (
                float(threshold_result["threshold"])
                if threshold_result is not None
                else None
            ),
            "calibration_metrics": threshold_result,
            "heldout_metrics": spark_heldout,
        },
        "scinli": scinli_metrics,
        "hard_safety": {
            "extract_teacher_accessed": False,
            "extract_scientific_state_materialized": False,
            "production_routing_changed": False,
            "model_training_executed": False,
            "learned_certificate_used_in_extract": False,
            "semantic_control_or_ontology_in_extract": False,
            "rdf_or_ontology_output_emitted": False,
        },
        "extract_teacher_accessed": False,
        "extract_scientific_state_materialized": False,
        "production_routing_changed": False,
        "model_training_executed": False,
        "learned_certificate_used_in_extract": False,
        "semantic_control_or_ontology_in_extract": False,
        "rdf_or_ontology_output_emitted": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "nli_certificate_decision": report[
                "nli_certificate_decision"
            ],
            "selected_threshold": report["spark"][
                "selected_threshold"
            ],
            "spark_heldout": report["spark"][
                "heldout_metrics"
            ],
            "scinli": report["scinli"],
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
