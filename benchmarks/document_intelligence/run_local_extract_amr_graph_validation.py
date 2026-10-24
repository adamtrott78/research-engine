from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
from typing import Any

from research_engine.local_extract_amr_graph_validation import (
    aggregate_bio_smatch,
    build_and_score_spark,
    git_blob_sha1_file,
    md5_file,
    sha256_file,
    tokenizer_manifest_sha256,
    validation_verdict,
)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bio_amr(path: pathlib.Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []
    for raw in re.split(r"\n\s*\n", text):
        lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        metadata: dict[str, str] = {}
        graph_lines: list[str] = []
        for line in lines:
            if line.startswith("# ::"):
                match = re.match(r"# ::(\S+)\s+(.*)$", line)
                if match:
                    metadata[match.group(1)] = match.group(2).strip()
                continue
            if not line.startswith("#"):
                graph_lines.append(line.strip())
        if not graph_lines:
            continue
        sentence = metadata.get("snt", "").strip()
        if not sentence:
            raise RuntimeError("Bio-AMR entry missing ::snt metadata")
        entries.append(
            {
                "id": metadata.get("id", str(len(entries))),
                "sentence": sentence,
                "graph": " ".join(graph_lines),
            }
        )
    return entries


def _all_spark_texts(pairs: dict[str, Any]) -> list[str]:
    values: set[str] = set()
    for split in ("calibration", "heldout"):
        for row in pairs[split]:
            values.add(str(row["left"]))
            values.add(str(row["right"]))
    return sorted(values)


def _set_parser_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_parser(
    *,
    model_dir: pathlib.Path,
    tokenizer_dir: pathlib.Path,
    inference: dict[str, Any],
) -> Any:
    import amrlib
    import torch

    if amrlib.__version__ != "0.8.1":
        raise RuntimeError(f"unexpected amrlib version: {amrlib.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen AMR validation")
    _set_parser_seed(int(inference["deterministic_seed"]))
    return amrlib.load_stog_model(
        model_dir=str(model_dir),
        device=str(inference["device"]),
        batch_size=int(inference["batch_size"]),
        num_beams=int(inference["num_beams"]),
        tok_name_or_path=str(tokenizer_dir),
    )


def _parse_sentences(model: Any, sentences: list[str]) -> list[str | None]:
    return model.parse_sents(
        sentences,
        add_metadata=False,
        disable_progress=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parser-tarball", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--amrlib-source", type=pathlib.Path, required=True)
    parser.add_argument("--smatch-source", type=pathlib.Path, required=True)
    parser.add_argument("--spark-json", type=pathlib.Path, required=True)
    parser.add_argument("--bio-amr", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    design = _load_json(args.design)
    selected = design["selected_validation"]
    if selected["id"] != "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1":
        raise RuntimeError(f"unexpected validation id: {selected['id']}")

    amrlib_contract = selected["amrlib"]
    model_contract = selected["parser_model"]
    inference = model_contract["inference"]
    data_contracts = {
        row["id"]: row for row in selected["external_validation_sets"]
    }
    bio_contract = data_contracts["BIO_AMR_V0_8_TEST"]
    spark_contract = data_contracts["SPARK_PROPOSITION_CLUSTERING_MANUAL"]

    if args.parser_tarball.stat().st_size != int(
        model_contract["release_asset_size_bytes"]
    ):
        raise RuntimeError("parser release asset size mismatch")
    parser_md5 = md5_file(args.parser_tarball)
    if parser_md5 != str(model_contract["published_md5"]):
        raise RuntimeError(f"parser MD5 mismatch: {parser_md5}")
    parser_sha256 = sha256_file(args.parser_tarball)

    amrlib_inference = args.amrlib_source / "amrlib/models/parse_xfm/inference.py"
    amrlib_smatch_eval = args.amrlib_source / "amrlib/evaluate/smatch_enhanced.py"
    if git_blob_sha1_file(amrlib_inference) != str(
        amrlib_contract["parse_inference_blob"]
    ):
        raise RuntimeError("amrlib parse inference blob mismatch")
    if git_blob_sha1_file(amrlib_smatch_eval) != str(
        amrlib_contract["smatch_evaluator_blob"]
    ):
        raise RuntimeError("amrlib Smatch evaluator blob mismatch")

    smatch_py = args.smatch_source / "smatch.py"
    if not smatch_py.is_file():
        raise RuntimeError("pinned Smatch source missing smatch.py")

    if git_blob_sha1_file(args.spark_json) != str(spark_contract["git_blob_sha"]):
        raise RuntimeError("SPARK git blob identity mismatch")
    if git_blob_sha1_file(args.bio_amr) != str(bio_contract["mirror_git_blob_sha"]):
        raise RuntimeError("Bio-AMR git blob identity mismatch")

    tokenizer_manifest, tokenizer_files = tokenizer_manifest_sha256(
        args.tokenizer_dir
    )

    spark = _load_json(args.spark_json)
    from research_engine.local_extract_nli_certificate_validation import (
        build_spark_pairs,
    )

    spark_pairs = build_spark_pairs(spark, negative_per_positive=5)
    spark_texts = _all_spark_texts(spark_pairs)

    bio_entries = _parse_bio_amr(args.bio_amr)
    expected_bio_count = int(bio_contract["expected_amr_count"])
    if len(bio_entries) != expected_bio_count:
        raise RuntimeError(
            f"Bio-AMR count mismatch: {len(bio_entries)} != {expected_bio_count}"
        )

    model = _load_parser(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        inference=inference,
    )

    spark_graphs = _parse_sentences(model, spark_texts)
    graph_by_text = dict(zip(spark_texts, spark_graphs))
    bio_predictions = _parse_sentences(
        model,
        [entry["sentence"] for entry in bio_entries],
    )

    import smatch

    if int(getattr(smatch, "iteration_num")) != int(
        amrlib_contract["smatch"]["standard_iteration_num"]
    ):
        raise RuntimeError(
            f"unexpected Smatch iteration_num: {getattr(smatch, 'iteration_num')}"
        )

    spark_result = build_and_score_spark(
        spark,
        graph_by_text,
        smatch_module=smatch,
        threshold_grid=[
            float(value) for value in selected["graph_similarity"]["threshold_grid"]
        ],
    )
    bio_metrics = aggregate_bio_smatch(
        bio_entries,
        bio_predictions,
        smatch_module=smatch,
    )

    decision = validation_verdict(
        bio_metrics=bio_metrics,
        spark_threshold=spark_result["threshold"],
        spark_heldout=spark_result["heldout_metrics"],
        bio_gate=bio_contract["predeclared_gate"],
        spark_gate=spark_contract["heldout_gate"],
    )

    report = {
        "schema_version": 1,
        "operation": "AMR_GRAPH_REDUNDANCY_CERTIFICATE_VALIDATION_V1",
        "design_id": design["design_id"],
        **decision,
        "parser": {
            "amrlib_repository": amrlib_contract["repository"],
            "amrlib_source_commit": amrlib_contract["source_commit"],
            "amrlib_version": amrlib_contract["version"],
            "parse_inference_blob": amrlib_contract["parse_inference_blob"],
            "smatch_evaluator_blob": amrlib_contract["smatch_evaluator_blob"],
            "parser_repository": model_contract["repository"],
            "release_tag": model_contract["release_tag"],
            "release_asset_id": model_contract["release_asset_id"],
            "release_asset_name": model_contract["release_asset_name"],
            "release_asset_size_bytes": args.parser_tarball.stat().st_size,
            "release_asset_md5": parser_md5,
            "release_asset_sha256": parser_sha256,
            "model_type": model_contract["model_type"],
            "model_version": model_contract["model_version"],
            "base_model": model_contract["base_model"],
            "device": inference["device"],
            "batch_size": inference["batch_size"],
            "num_beams": inference["num_beams"],
            "deterministic_seed": inference["deterministic_seed"],
        },
        "tokenizer": {
            "repository": model_contract["tokenizer"]["repository"],
            "revision": model_contract["tokenizer"]["revision"],
            "manifest_sha256": tokenizer_manifest,
            "files": tokenizer_files,
        },
        "smatch": {
            "repository": amrlib_contract["smatch"]["repository"],
            "release_tag": amrlib_contract["smatch"]["release_tag"],
            "release_commit": amrlib_contract["smatch"]["release_commit"],
            "iteration_num": amrlib_contract["smatch"]["standard_iteration_num"],
            "smatch_py_sha256": sha256_file(smatch_py),
            "deterministic_reseed_rule": amrlib_contract["smatch"][
                "deterministic_reseed_rule"
            ],
        },
        "external_data": {
            "bio_amr": {
                "mirror_repository": bio_contract["mirror_repository"],
                "mirror_commit": bio_contract["mirror_commit"],
                "mirror_file": bio_contract["mirror_file"],
                "git_blob_sha": bio_contract["mirror_git_blob_sha"],
                "file_sha256": sha256_file(args.bio_amr),
            },
            "spark": {
                "repository": spark_contract["repository"],
                "repository_commit": spark_contract["repository_commit"],
                "file": spark_contract["file"],
                "git_blob_sha": spark_contract["git_blob_sha"],
                "file_sha256": sha256_file(args.spark_json),
            },
        },
        "bio_amr": bio_metrics,
        "spark": {
            "calibration_topic_count": len(
                spark_result["pairs"]["calibration_topics"]
            ),
            "heldout_topic_count": len(
                spark_result["pairs"]["heldout_topics"]
            ),
            "calibration_positive_count": spark_result["pairs"][
                "calibration_positive_count"
            ],
            "calibration_negative_count": spark_result["pairs"][
                "calibration_negative_count"
            ],
            "heldout_positive_count": spark_result["pairs"][
                "heldout_positive_count"
            ],
            "heldout_negative_count": spark_result["pairs"][
                "heldout_negative_count"
            ],
            "unique_text_count": len(spark_texts),
            "invalid_graph_count": sum(
                1 for value in spark_graphs if not value
            ),
            "calibration_invalid_pair_count": spark_result[
                "calibration_invalid_pair_count"
            ],
            "heldout_invalid_pair_count": spark_result[
                "heldout_invalid_pair_count"
            ],
            "selected_threshold": (
                float(spark_result["threshold"]["threshold"])
                if spark_result["threshold"] is not None
                else None
            ),
            "calibration_metrics": spark_result["threshold"],
            "heldout_metrics": spark_result["heldout_metrics"],
        },
        "hard_safety": {
            "extract_teacher_accessed": False,
            "extract_scientific_state_materialized": False,
            "amr_used_in_extract": False,
            "parser_training_executed": False,
            "qwen_or_codex_changed": False,
            "accepted_verify_changed": False,
            "production_routing_changed": False,
            "wave7b_started": False,
            "semantic_control_or_ontology_in_extract": False,
            "rdf_or_ontology_output_emitted": False,
        },
        "extract_teacher_accessed": False,
        "extract_scientific_state_materialized": False,
        "amr_used_in_extract": False,
        "parser_training_executed": False,
        "qwen_or_codex_changed": False,
        "accepted_verify_changed": False,
        "production_routing_changed": False,
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
