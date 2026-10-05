from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .document_docling import _table_grid
from .document_package import BlockKind, PaperPackage
from .document_sidecar import document_sidecar_compiler_from_env
from .local_inference import local_inference_service_from_env
from .orchestration import ExecutorTier, JobType
from .paper_package_ab import PaperPackageABPlanner, extract_agreement
from .postgres import PostgresStore
from .qwen_guarded_semantic import (
    GuardedQwenCalibrationPlanner,
    job_from_record,
    job_to_record,
)

TASK_ID = "research-engine-local-document-pipeline-v1"
OPERATION_TYPE = "QWEN35_GROUNDED_CANDIDATE_EXTRACT_CALIBRATION"
EXECUTION_CONTRACT = "qwen35-vllm-cu129-image-native-sampler-v3"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
VLLM_WHEEL_SHA256 = "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
MAX_SELECTIONS = 12


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def execution_identity() -> str:
    return stable_sha256(
        {
            "runtime_contract": EXECUTION_CONTRACT,
            "model_revision": MODEL_REVISION,
            "sampling_backend": "vllm_native",
            "vllm_distribution": "0.30.0+cu129",
            "torch": "2.13.0+cu129",
            "torchvision": "0.28.0+cu129",
            "torchcodec": "0.16.0+cu129",
        }
    )


def package_identity() -> str:
    return stable_sha256(
        {
            "vllm_distribution": "0.30.0+cu129",
            "vllm_release_wheel_sha256": VLLM_WHEEL_SHA256,
            "torch": "2.13.0+cu129",
            "torchvision": "0.28.0+cu129",
            "torchcodec": "0.16.0+cu129",
            "cuda_runtime": "12.9",
        }
    )


def operation_id(*, git_sha: str, source_job_id: str, source_pdf_sha256: str) -> str:
    return stable_sha256(
        {
            "task_id": TASK_ID,
            "git_sha": git_sha,
            "operation_type": OPERATION_TYPE,
            "execution_identity": execution_identity(),
            "material_inputs": {
                "source_job_id": source_job_id,
                "source_pdf_sha256": source_pdf_sha256,
                "candidate_contract_version": 1,
                "model_revision": MODEL_REVISION,
            },
        }
    )


def _display(raw: str) -> str:
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", str(raw))
    return re.sub(r"\s+", " ", value).strip()


def exact_text_candidates(pages: list[dict[str, Any]], max_chars: int = 650):
    rows: list[dict[str, Any]] = []
    for page in pages:
        page_no = int(page["page_number"])
        text = str(page.get("text") or "")
        start: int | None = None
        end = 0
        for line in re.finditer(r".*(?:\n|$)", text):
            if not line.group(0):
                continue
            if start is None:
                start = line.start()
            end = line.end()
            current = text[start:end]
            terminal = bool(re.search(r'[.!?]["\')\]]?$', current.rstrip()))
            if len(current) < max_chars and not terminal:
                continue
            raw = text[start:end]
            if len(_display(raw)) >= 12:
                rows.append(
                    {
                        "candidate_id": f"pdf:p{page_no}:{len(rows):04d}",
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "candidate_kind": "TEXT_SPAN",
                        "page_number": page_no,
                        "pdf_bbox": None,
                        "raw_source_form": raw,
                        "display_text": _display(raw),
                        "derivation": {
                            "processor": "pypdf-selected-page-text",
                            "mode": "NATIVE_TEXT",
                            "start_char": start,
                            "end_char": end,
                        },
                    }
                )
            start = None
        if start is not None and end > start:
            raw = text[start:end]
            if len(_display(raw)) >= 12:
                rows.append(
                    {
                        "candidate_id": f"pdf:p{page_no}:{len(rows):04d}",
                        "source_layer": "AUTHORITATIVE_PDF_TEXT",
                        "candidate_kind": "TEXT_SPAN",
                        "page_number": page_no,
                        "pdf_bbox": None,
                        "raw_source_form": raw,
                        "display_text": _display(raw),
                        "derivation": {
                            "processor": "pypdf-selected-page-text",
                            "mode": "NATIVE_TEXT",
                            "start_char": start,
                            "end_char": end,
                        },
                    }
                )
    return rows


def _candidate_id(layer: str, page: int, kind: str, content: str, bbox):
    digest = hashlib.sha256(
        _canonical(
            {
                "layer": layer,
                "page": page,
                "kind": kind,
                "content": content,
                "bbox": bbox,
            }
        )
    ).hexdigest()[:16]
    return f"{layer.lower()}:p{page}:{digest}"


def _package_candidates(
    package: PaperPackage,
    package_root: Path,
    layer: str,
    selected_pages: set[int],
):
    rows: list[dict[str, Any]] = []
    keep = {
        BlockKind.HEADING,
        BlockKind.CAPTION,
        BlockKind.EQUATION,
        BlockKind.TABLE,
        BlockKind.FIGURE,
    }
    for block in package.blocks:
        page = int(block.source_region.page_number)
        content = str(block.markdown or "").strip()
        if page not in selected_pages or block.kind not in keep or not content:
            continue
        bbox = list(block.source_region.pdf_bbox) if block.source_region.pdf_bbox else None
        rows.append(
            {
                "candidate_id": _candidate_id(layer, page, str(block.kind), content, bbox),
                "source_layer": layer,
                "candidate_kind": str(block.kind),
                "page_number": page,
                "pdf_bbox": bbox,
                "raw_source_form": content,
                "display_text": _display(content),
                "derivation": {
                    "block_id": block.id,
                    "asset_ids": list(block.asset_ids),
                    "section_path": list(block.source_region.section_path),
                },
            }
        )
    for asset in package.equations:
        page = int(asset.source_region.page_number)
        content = asset.latex.strip()
        if page not in selected_pages or not content:
            continue
        bbox = list(asset.source_region.pdf_bbox) if asset.source_region.pdf_bbox else None
        rows.append(
            {
                "candidate_id": _candidate_id(
                    layer + "_EQUATION", page, "EQUATION_ASSET", content, bbox
                ),
                "source_layer": layer + "_EQUATION",
                "candidate_kind": "EQUATION_ASSET",
                "page_number": page,
                "pdf_bbox": bbox,
                "raw_source_form": content,
                "display_text": content,
                "derivation": {"asset_id": asset.id},
            }
        )
    for asset in package.tables:
        page = int(asset.source_region.page_number)
        if page not in selected_pages or not asset.markdown_path:
            continue
        path = package_root / asset.markdown_path
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        bbox = list(asset.source_region.pdf_bbox) if asset.source_region.pdf_bbox else None
        rows.append(
            {
                "candidate_id": _candidate_id(
                    layer + "_TABLE", page, "TABLE_ASSET", content, bbox
                ),
                "source_layer": layer + "_TABLE",
                "candidate_kind": "TABLE_ASSET",
                "page_number": page,
                "pdf_bbox": bbox,
                "raw_source_form": content,
                "display_text": _display(content),
                "derivation": {"asset_id": asset.id},
            }
        )
    return rows


def _specialized_candidates(sidecar: Path, pages: set[int]):
    rows: list[dict[str, Any]] = []
    for page in sorted(pages):
        formula = sidecar / "specialized" / "formula" / f"page-{page:04d}"
        if formula.is_dir():
            for path in sorted(formula.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                for idx, chunk in enumerate(
                    x.strip() for x in re.split(r"\n{2,}", text) if x.strip()
                ):
                    if len(chunk) > 1200:
                        continue
                    rows.append(
                        {
                            "candidate_id": _candidate_id(
                                "CODEFORMULAV2", page, "FORMULA_ENRICHMENT", chunk, None
                            ),
                            "source_layer": "CODEFORMULAV2",
                            "candidate_kind": "FORMULA_ENRICHMENT",
                            "page_number": page,
                            "pdf_bbox": None,
                            "raw_source_form": chunk,
                            "display_text": _display(chunk),
                            "derivation": {
                                "path": str(path.relative_to(sidecar)),
                                "chunk_index": idx,
                            },
                        }
                    )
        accurate = sidecar / "specialized" / "table-accurate" / f"page-{page:04d}"
        if accurate.is_dir():
            for path in sorted(accurate.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for idx, item in enumerate(payload.get("tables") or []):
                    if not isinstance(item, dict):
                        continue
                    grid = _table_grid(item)
                    if not grid:
                        continue
                    content = "\n".join(
                        " | ".join(str(cell) for cell in row) for row in grid
                    )
                    rows.append(
                        {
                            "candidate_id": _candidate_id(
                                "TABLEFORMER_ACCURATE",
                                page,
                                "TABLE_GRID",
                                content,
                                None,
                            ),
                            "source_layer": "TABLEFORMER_ACCURATE",
                            "candidate_kind": "TABLE_GRID",
                            "page_number": page,
                            "pdf_bbox": None,
                            "raw_source_form": content,
                            "display_text": _display(content),
                            "derivation": {
                                "path": str(path.relative_to(sidecar)),
                                "table_index": idx,
                            },
                        }
                    )
    return rows


def load_sidecar_candidates(
    source_sha: str,
    sidecar_root: str | Path,
    selected_pages: set[int],
):
    root = Path(sidecar_root).expanduser().resolve()
    sidecar = root / "sha256" / source_sha[:2] / source_sha[2:4] / source_sha
    receipt_path = sidecar / "receipt.json"
    if not receipt_path.is_file():
        return [], {
            "available": False,
            "cache_path": str(sidecar),
            "reason": "SIDECAR_RECEIPT_MISSING",
        }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for layer, key in (
        ("DOCLING_NATIVE", "native_package_relative_path"),
        ("DOCLING_STANDARD", "standard_package_relative_path"),
    ):
        rel = receipt.get(key)
        if not rel:
            continue
        path = sidecar / str(rel)
        if not path.is_file():
            continue
        package = PaperPackage.model_validate_json(path.read_text(encoding="utf-8"))
        added = _package_candidates(package, path.parent, layer, selected_pages)
        rows.extend(added)
        counts[layer] = len(added)
    specialized = _specialized_candidates(sidecar, selected_pages)
    rows.extend(specialized)
    for row in specialized:
        counts[row["source_layer"]] = counts.get(row["source_layer"], 0) + 1
    return rows, {
        "available": True,
        "cache_path": str(sidecar),
        "receipt": receipt,
        "layer_counts": counts,
    }


def _dedupe(rows):
    out = []
    seen = set()
    for row in rows:
        key = (
            int(row["page_number"]),
            str(row["candidate_kind"]),
            _display(str(row["raw_source_form"])).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


class GroundedExtractPlanner:
    def __init__(self, store, artifact_root, sidecar_root):
        self.store = store
        self.base = GuardedQwenCalibrationPlanner(
            store=store,
            artifact_root=artifact_root,
            max_context_chars=20_000,
        )
        self.paper = PaperPackageABPlanner(store=store, artifact_root=artifact_root)
        self.sidecar_root = Path(sidecar_root)

    def selection(self, run_id: str):
        controlled = self.base._controlled_job_ids(run_id, JobType.EXTRACT)
        completed = {
            job.id: job
            for job in self.paper.completed_jobs(run_id, JobType.EXTRACT)
        }
        rejected = []
        for job_id in controlled:
            source = completed.get(job_id)
            if source is None:
                rejected.append({"job_id": job_id, "reason": "NOT_COMPLETED"})
                continue
            bounded, reason = self.base._bounded_extract(source)
            if bounded is None:
                rejected.append({"job_id": job_id, "reason": reason})
                continue
            context = dict(bounded.payload.get("_research_context") or {})
            artifact = dict(context.get("source_artifact") or {})
            return {
                "schema_version": 1,
                "operation": OPERATION_TYPE,
                "source_run_id": run_id,
                "source_job_id": source.id,
                "source_pdf_sha256": str(
                    artifact.get("sha256")
                    or bounded.payload["acquisition"]["sha256"]
                ),
                "source_artifact_id": str(
                    artifact.get("id")
                    or bounded.payload["acquisition"]["artifact_id"]
                ),
                "context_page_numbers": [
                    int(page["page_number"])
                    for page in context.get("pages") or []
                ],
                "context_chars": sum(
                    len(str(page.get("text") or ""))
                    for page in context.get("pages") or []
                ),
                "job": job_to_record(bounded),
                "production_routing_changed": False,
                "scientific_state_materialized": False,
            }
        raise ValueError(
            "no bounded EXTRACT source; "
            f"rejections={json.dumps(rejected, sort_keys=True)}"
        )

    def build(self, selection):
        job = job_from_record(selection["job"])
        context = dict(job.payload.get("_research_context") or {})
        pages = [dict(x) for x in context.get("pages") or []]
        selected_pages = {int(x["page_number"]) for x in pages}
        exact = exact_text_candidates(pages)
        derived, sidecar = load_sidecar_candidates(
            str(selection["source_pdf_sha256"]),
            self.sidecar_root,
            selected_pages,
        )
        candidates = _dedupe(exact + derived)
        exact_candidates = [
            row for row in candidates
            if row["source_layer"] == "AUTHORITATIVE_PDF_TEXT"
        ]
        extras = [
            row for row in candidates
            if row["source_layer"] != "AUTHORITATIVE_PDF_TEXT"
        ]
        kept = list(exact_candidates)
        chars = sum(len(row["display_text"]) for row in kept)
        for row in extras:
            size = len(row["display_text"])
            if chars + size > 18_000:
                continue
            kept.append(row)
            chars += size
        return {
            **selection,
            "candidate_contract_version": 1,
            "scientific_qwen_call_budget": 1,
            "c3_call_budget": 1,
            "candidates": kept,
            "candidate_summary": {
                "total": len(kept),
                "prompt_text_chars": chars,
                "by_layer": {
                    layer: sum(1 for x in kept if x["source_layer"] == layer)
                    for layer in sorted({x["source_layer"] for x in kept})
                },
                "sidecar": sidecar,
            },
        }


def ensure_sidecar(selection):
    store = PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"])
    with local_inference_service_from_env(require_enabled=True) as local:
        compiler = document_sidecar_compiler_from_env(
            store=store,
            artifact_root=os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"],
            local_inference=local,
            require_enabled=True,
        )
        assert compiler is not None
        return compiler.compile_artifact(str(selection["source_artifact_id"]))


def guard_and_canonicalize(plan, payload):
    candidates = {
        str(row["candidate_id"]): dict(row)
        for row in plan.get("candidates") or []
    }
    selections = list(payload.get("selections") or [])
    reasons = []
    if not selections:
        reasons.append("NO_SELECTIONS")
    if len(selections) > MAX_SELECTIONS:
        reasons.append(f"TOO_MANY_SELECTIONS:{len(selections)}")
    ids = [str(row.get("candidate_id") or "") for row in selections]
    if len(ids) != len(set(ids)):
        reasons.append("DUPLICATE_CANDIDATE_ID")
    unknown = sorted(x for x in ids if x not in candidates)
    if unknown:
        reasons.append("UNKNOWN_CANDIDATE_IDS:" + ",".join(unknown))

    evidence = []
    provenance = []
    for idx, row in enumerate(selections):
        candidate = candidates.get(str(row.get("candidate_id") or ""))
        if candidate is None:
            continue
        kind = str(row.get("type") or "").strip()
        normalized = str(row.get("normalized_form") or "").strip()
        confidence = row.get("confidence")
        if not kind:
            reasons.append(f"SELECTION_{idx}:EMPTY_TYPE")
        if not normalized:
            reasons.append(f"SELECTION_{idx}:EMPTY_NORMALIZED_FORM")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            reasons.append(f"SELECTION_{idx}:INVALID_CONFIDENCE")
            confidence = 0.0
        evidence.append(
            {
                "id": f"grounded:{candidate['candidate_id']}",
                "type": kind or "unknown",
                "normalized_form": normalized,
                "raw_source_form": candidate["raw_source_form"],
                "page_number": candidate["page_number"],
                "confidence": float(confidence),
            }
        )
        provenance.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_layer": candidate["source_layer"],
                "candidate_kind": candidate["candidate_kind"],
                "page_number": candidate["page_number"],
                "pdf_bbox": candidate.get("pdf_bbox"),
                "derivation": copy.deepcopy(candidate.get("derivation") or {}),
            }
        )
    return {
        "status": "PASS_GROUNDED" if not reasons else "FALLTHROUGH_CODEX",
        "reasons": reasons,
        "canonical_result": {"evidence": evidence} if not reasons else None,
        "selected_provenance": provenance,
    }


def evaluate(plan, qwen, codex):
    guard = guard_and_canonicalize(
        plan,
        dict((qwen.get("domain") or {}).get("parsed") or {}),
    )
    control = dict(codex.get("result") or {})
    agreement = (
        extract_agreement(control, guard["canonical_result"])
        if guard["status"] == "PASS_GROUNDED"
        else None
    )
    pass_thresholds = False
    if agreement:
        pass_thresholds = (
            agreement["historical_coverage_at_f1_0_5"] >= 0.95
            and agreement["replay_precision_at_f1_0_5"] >= 0.95
            and agreement["page_agreement_on_passing_matches"] >= 0.95
            and agreement["type_agreement_on_passing_matches"] >= 0.95
        )
    canary_pass = bool((qwen.get("canary") or {}).get("pass"))
    return {
        "schema_version": 1,
        "operation": OPERATION_TYPE,
        "decision": (
            "PROMOTION_CANDIDATE_GROUNDED_EXTRACT"
            if canary_pass
            and guard["status"] == "PASS_GROUNDED"
            and pass_thresholds
            else "FAIL_CLOSED_RETAIN_CODEX"
        ),
        "c3_canary_pass": canary_pass,
        "guard": guard,
        "codex_agreement": agreement,
        "promotion_thresholds_pass": pass_thresholds,
        "production_routing_changed": False,
        "scientific_state_materialized": False,
    }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("select")
    a.add_argument("run_id")
    a.add_argument("--output", required=True)
    a = sub.add_parser("ensure-sidecar")
    a.add_argument("--selection", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("build")
    a.add_argument("--selection", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("control")
    a.add_argument("--plan", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("evaluate")
    a.add_argument("--plan", required=True)
    a.add_argument("--qwen", required=True)
    a.add_argument("--codex", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("operation-id")
    a.add_argument("--selection", required=True)
    a.add_argument("--git-sha", required=True)
    sub.add_parser("execution-id")
    sub.add_parser("package-id")
    args = p.parse_args()

    store = PostgresStore(os.environ["RESEARCH_ENGINE_DATABASE_URL"])
    artifact_root = os.environ["RESEARCH_ENGINE_ARTIFACT_ROOT"]
    sidecar_root = os.environ.get(
        "RESEARCH_ENGINE_DOCUMENT_SIDECAR_ROOT",
        str(Path.home() / "adamArchives/Adam/.research-engine/document-sidecars"),
    )

    if args.command == "select":
        result = GroundedExtractPlanner(
            store, artifact_root, sidecar_root
        ).selection(args.run_id)
        _write(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "ensure-sidecar":
        result = ensure_sidecar(_read(args.selection))
        _write(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "build":
        result = GroundedExtractPlanner(
            store, artifact_root, sidecar_root
        ).build(_read(args.selection))
        _write(args.output, result)
        print(json.dumps(result["candidate_summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "control":
        from .codex_executor import CodexExecExecutor
        plan = _read(args.plan)
        job = job_from_record(plan["job"])
        result = CodexExecExecutor(tier=ExecutorTier.STANDARD).execute(job)
        payload = {
            "schema_version": 1,
            "control": "CONTEMPORANEOUS_STANDARD_CODEX_AUTHORITATIVE_PDF",
            "result": result,
        }
        _write(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "evaluate":
        result = evaluate(
            _read(args.plan),
            _read(args.qwen),
            _read(args.codex),
        )
        _write(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "execution-id":
        print(execution_identity())
        return 0
    if args.command == "package-id":
        print(package_identity())
        return 0

    selection = _read(args.selection)
    print(
        operation_id(
            git_sha=args.git_sha,
            source_job_id=selection["source_job_id"],
            source_pdf_sha256=selection["source_pdf_sha256"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
