from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
from typing import Any

from research_engine.local_extract_amr_graph_validation import (
    score_bio_predictions,
    score_spark_pairs,
    validation_verdict,
)
from research_engine.local_extract_nli_certificate_validation import (
    binary_metrics,
    build_spark_pairs,
    select_threshold,
)


TOKENIZER_MANIFEST_FILES = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: pathlib.Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _tokenizer_manifest(directory: pathlib.Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in TOKENIZER_MANIFEST_FILES:
        path = directory / name
        if not path.is_file():
            continue
        rows.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise RuntimeError("no pinned tokenizer files found")
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "files": rows,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _load_bio_amr(
    path: pathlib.Path,
) -> tuple[list[str], list[str], list[str]]:
    text = path.read_text(encoding="utf-8")
    example_ids: list[str] = []
    sentences: list[str] = []
    graphs: list[str] = []

    for block in re.split(r"\n\s*\n", text):
        lines = [line.rstrip() for line in block.splitlines()]
        sentence = None
        example_id = None
        graph_lines: list[str] = []
        for line in lines:
            if line.startswith("# ::id "):
                example_id = line.split()[2]
            elif line.startswith("# ::snt "):
                sentence = line[len("# ::snt "):].strip()
            elif line.strip() and not line.lstrip().startswith("#"):
                graph_lines.append(line.strip())
        if sentence is None or not graph_lines:
            continue
        example_ids.append(
            example_id or f"index-{len(example_ids):04d}"
        )
        sentences.append(sentence)
        graphs.append(" ".join(graph_lines))

    return example_ids, sentences, graphs


def _set_parser_seed(seed: int) -> None:
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--spark-json", type=pathlib.Path, required=True)
    parser.add_argument("--bio-amr", type=pathlib.Path, required=True)
    parser.add_argument("--parser-tar", type=pathlib.Path, required=True)
    parser.add_argument("--parser-model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    design = _load_json(args.design)
    selected = design["selected_validation"]
    if selected["id"] != "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1":
        raise RuntimeError("unexpected AMR validation design")

    parser_spec = selected["parser_model"]
    amrlib_spec = selected["amrlib"]
    bio_spec = selected["external_validation_sets"][0]
    spark_spec = selected["external_validation_sets"][1]
    inference = parser_spec["inference"]

    if args.parser_tar.stat().st_size != int(
        parser_spec["release_asset_size_bytes"]
    ):
        raise RuntimeError("parser release asset size mismatch")
    if _md5(args.parser_tar) != parser_spec["published_md5"]:
        raise RuntimeError("parser release asset MD5 mismatch")
    if _git_blob_sha1(args.bio_amr) != bio_spec["mirror_git_blob_sha"]:
        raise RuntimeError("Bio-AMR mirror blob mismatch")
    if _git_blob_sha1(args.spark_json) != spark_spec["git_blob_sha"]:
        raise RuntimeError("SPARK blob mismatch")

    tokenizer_manifest = _tokenizer_manifest(args.tokenizer_dir)

    import amrlib
    import smatch
    import torch

    if amrlib.__version__ != amrlib_spec["version"]:
        raise RuntimeError(
            f"amrlib version mismatch: {amrlib.__version__}"
        )
    if int(smatch.iteration_num) != int(
        amrlib_spec["smatch"]["standard_iteration_num"]
    ):
        raise RuntimeError("Smatch iteration_num mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen AMR validation")

    model_meta = _load_json(args.parser_model_dir / "amrlib_meta.json")
    if model_meta.get("model_type") != parser_spec["model_type"]:
        raise RuntimeError("parser model_type mismatch")
    if model_meta.get("version") != parser_spec["model_version"]:
        raise RuntimeError("parser model version mismatch")
    if model_meta.get("base_model") != parser_spec["base_model"]:
        raise RuntimeError("parser base_model mismatch")

    _set_parser_seed(int(inference["deterministic_seed"]))
    stog = amrlib.load_stog_model(
        model_dir=str(args.parser_model_dir),
        device=str(inference["device"]),
        batch_size=int(inference["batch_size"]),
        num_beams=int(inference["num_beams"]),
        tok_name_or_path=str(args.tokenizer_dir),
    )

    bio_ids, bio_sentences, bio_gold = _load_bio_amr(args.bio_amr)
    if len(bio_gold) != int(bio_spec["expected_amr_count"]):
        raise RuntimeError(
            f"expected {bio_spec['expected_amr_count']} Bio-AMRs, "
            f"got {len(bio_gold)}"
        )
    bio_predictions = stog.parse_sents(
        bio_sentences,
        add_metadata=bool(inference["add_metadata"]),
        disable_progress=True,
    )
    bio_metrics = score_bio_predictions(
        list(bio_predictions),
        bio_gold,
        example_ids=bio_ids,
        smatch_module=smatch,
    )

    spark = _load_json(args.spark_json)
    pairs = build_spark_pairs(spark, negative_per_positive=5)
    spark_rows = list(pairs["calibration"]) + list(pairs["heldout"])
    spark_texts = sorted(
        {
            str(row["left"])
            for row in spark_rows
        } | {
            str(row["right"])
            for row in spark_rows
        }
    )
    spark_graph_list = stog.parse_sents(
        spark_texts,
        add_metadata=bool(inference["add_metadata"]),
        disable_progress=True,
    )
    graph_by_text = dict(zip(spark_texts, spark_graph_list))

    cal_labels, cal_scores, cal_diag = score_spark_pairs(
        list(pairs["calibration"]),
        graph_by_text=graph_by_text,
        smatch_module=smatch,
    )
    held_labels, held_scores, held_diag = score_spark_pairs(
        list(pairs["heldout"]),
        graph_by_text=graph_by_text,
        smatch_module=smatch,
    )

    grid = [
        float(value)
        for value in selected["graph_similarity"]["threshold_grid"]
    ]
    selected_threshold = select_threshold(
        cal_labels,
        cal_scores,
        grid,
        precision_min=0.95,
        recall_min=0.05,
    )
    if selected_threshold is None:
        heldout_metrics = {
            "threshold": None,
            "tp": 0,
            "fp": 0,
            "tn": sum(1 for value in held_labels if value == 0),
            "fn": sum(1 for value in held_labels if value == 1),
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
        }
    else:
        heldout_metrics = binary_metrics(
            held_labels,
            held_scores,
            float(selected_threshold["threshold"]),
        )

    decision = validation_verdict(
        bio_metrics=bio_metrics,
        threshold_selected=selected_threshold is not None,
        spark_heldout=heldout_metrics,
        bio_gate=bio_spec["predeclared_gate"],
        spark_gate=spark_spec["heldout_gate"],
    )

    report = {
        "schema_version": 1,
        "operation": "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION",
        "design_id": design["design_id"],
        "candidate_sha": args.candidate_sha,
        **decision,
        "amrlib": {
            "repository": amrlib_spec["repository"],
            "source_commit": amrlib_spec["source_commit"],
            "version": amrlib.__version__,
            "parse_inference_blob": amrlib_spec["parse_inference_blob"],
            "smatch_evaluator_blob": amrlib_spec["smatch_evaluator_blob"],
        },
        "smatch": {
            "repository": amrlib_spec["smatch"]["repository"],
            "release_tag": amrlib_spec["smatch"]["release_tag"],
            "release_commit": amrlib_spec["smatch"]["release_commit"],
            "iteration_num": int(smatch.iteration_num),
            "deterministic_reseed_rule": amrlib_spec["smatch"][
                "deterministic_reseed_rule"
            ],
        },
        "parser_model": {
            "release_asset_id": parser_spec["release_asset_id"],
            "release_asset_name": parser_spec["release_asset_name"],
            "release_asset_size_bytes": args.parser_tar.stat().st_size,
            "published_md5": parser_spec["published_md5"],
            "measured_md5": _md5(args.parser_tar),
            "tarball_sha256": _sha256(args.parser_tar),
            "model_type": model_meta["model_type"],
            "model_version": model_meta["version"],
            "base_model": model_meta["base_model"],
            "device": str(inference["device"]),
            "batch_size": int(inference["batch_size"]),
            "num_beams": int(inference["num_beams"]),
            "deterministic_seed": int(inference["deterministic_seed"]),
        },
        "tokenizer": {
            "repository": parser_spec["tokenizer"]["repository"],
            "revision": parser_spec["tokenizer"]["revision"],
            **tokenizer_manifest,
        },
        "external_data": {
            "bio_amr": {
                "mirror_repository": bio_spec["mirror_repository"],
                "mirror_commit": bio_spec["mirror_commit"],
                "mirror_file": bio_spec["mirror_file"],
                "mirror_git_blob_sha": bio_spec["mirror_git_blob_sha"],
                "file_sha256": _sha256(args.bio_amr),
            },
            "spark": {
                "repository": spark_spec["repository"],
                "repository_commit": spark_spec["repository_commit"],
                "file": spark_spec["file"],
                "git_blob_sha": spark_spec["git_blob_sha"],
                "file_sha256": _sha256(args.spark_json),
            },
        },
        "bio_amr": bio_metrics,
        "spark": {
            "calibration_topic_count": len(pairs["calibration_topics"]),
            "heldout_topic_count": len(pairs["heldout_topics"]),
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
                float(selected_threshold["threshold"])
                if selected_threshold is not None
                else None
            ),
            "calibration_metrics": selected_threshold,
            "heldout_metrics": heldout_metrics,
            "calibration_graph_diagnostics": cal_diag,
            "heldout_graph_diagnostics": held_diag,
        },
        "hard_safety": {
            "extract_teacher_accessed": False,
            "extract_scientific_state_materialized": False,
            "amr_certificate_used_in_extract": False,
            "production_routing_changed": False,
            "model_or_parser_training_executed": False,
            "wave7b_started": False,
            "semantic_control_or_ontology_in_extract": False,
            "rdf_or_ontology_output_emitted": False,
        },
        "extract_teacher_accessed": False,
        "extract_scientific_state_materialized": False,
        "amr_certificate_used_in_extract": False,
        "production_routing_changed": False,
        "model_or_parser_training_executed": False,
        "wave7b_started": False,
        "semantic_control_or_ontology_in_extract": False,
        "rdf_or_ontology_output_emitted": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "amr_graph_certificate_decision": report[
                    "amr_graph_certificate_decision"
                ],
                "bio_amr": report["bio_amr"],
                "spark_selected_threshold": report["spark"][
                    "selected_threshold"
                ],
                "spark_heldout": report["spark"]["heldout_metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
