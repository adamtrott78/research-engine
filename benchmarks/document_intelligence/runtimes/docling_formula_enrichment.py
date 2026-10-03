from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_REPO = "docling-project/CodeFormulaV2"
MODEL_REVISION = "0284bc97b73c88776957d00d6695dd63a5a8784e"


def _device_name():
    from docling.datamodel.accelerator_options import AcceleratorDevice

    raw = os.getenv("RESEARCH_ENGINE_DOCLING_DEVICE", "auto").strip().lower()
    values = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    if raw not in values:
        raise ValueError(
            "RESEARCH_ENGINE_DOCLING_DEVICE must be one of auto,cpu,cuda,mps"
        )
    return values[raw]


def prefetch_formula_models(output_dir: Path) -> None:
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
    from docling.models.utils.hf_model_download import download_hf_model

    output_dir.mkdir(parents=True, exist_ok=True)
    layout_spec = LayoutObjectDetectionOptions().model_spec
    if layout_spec.repo_id is None or layout_spec.revision is None:
        raise RuntimeError("Docling layout model spec lacks repo/revision")

    layout_repos = {layout_spec.repo_id: layout_spec.revision}
    for override in layout_spec.engine_overrides.values():
        merged = override.merge_with(
            layout_spec.repo_id,
            layout_spec.revision,
        )
        if merged.repo_id is None or merged.revision is None:
            raise RuntimeError("Docling layout engine override is incomplete")
        layout_repos[merged.repo_id] = merged.revision

    for repo_id, revision in sorted(layout_repos.items()):
        print(
            json.dumps(
                {
                    "prefetch": "layout",
                    "repo_id": repo_id,
                    "revision": revision,
                },
                sort_keys=True,
            )
        )
        download_hf_model(
            repo_id=repo_id,
            revision=revision,
            local_dir=output_dir / repo_id.replace("/", "--"),
            force=False,
            progress=True,
        )

    print(
        json.dumps(
            {
                "prefetch": "formula",
                "repo_id": MODEL_REPO,
                "revision": MODEL_REVISION,
            },
            sort_keys=True,
        )
    )
    download_hf_model(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=output_dir / MODEL_REPO.replace("/", "--"),
        force=False,
        progress=True,
    )


def build_converter():
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        PdfPipelineOptions,
    )
    from docling.datamodel.vlm_engine_options import (
        TransformersVlmEngineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    artifacts = os.getenv("RESEARCH_ENGINE_DOCLING_ARTIFACTS")
    if not artifacts:
        raise RuntimeError(
            "RESEARCH_ENGINE_DOCLING_ARTIFACTS must point to the explicit local cache"
        )
    artifacts_path = Path(artifacts).resolve()
    if not artifacts_path.is_dir():
        raise FileNotFoundError(artifacts_path)

    threads = int(
        os.getenv(
            "RESEARCH_ENGINE_DOCLING_THREADS",
            str(max(1, os.cpu_count() or 1)),
        )
    )
    accelerator = AcceleratorOptions(
        num_threads=threads,
        device=_device_name(),
    )
    formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2",
        engine_options=TransformersVlmEngineOptions(
            compile_model=False,
        ),
    )
    formula_options.model_spec.revision = MODEL_REVISION
    formula_options.extract_code = False
    formula_options.extract_formulas = True

    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=False,
        do_table_structure=False,
        do_code_enrichment=False,
        do_formula_enrichment=True,
        code_formula_options=formula_options,
        generate_page_images=False,
        generate_picture_images=False,
        enable_remote_services=False,
        accelerator_options=accelerator,
    )
    if hasattr(pipeline_options, "do_picture_classification"):
        pipeline_options.do_picture_classification = False
    if hasattr(pipeline_options, "do_picture_description"):
        pipeline_options.do_picture_description = False
    if hasattr(pipeline_options, "do_chart_extraction"):
        pipeline_options.do_chart_extraction = False

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        },
    )


def run_formula_page(
    converter,
    *,
    source_pdf: str | Path,
    output_dir: str | Path,
    page_number: int,
) -> dict:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    source = Path(source_pdf).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    result = converter.convert(
        source,
        page_range=(page_number, page_number),
    )
    document = result.document
    json_path = output / f"{source.stem}.json"
    markdown_path = output / f"{source.stem}.md"
    json_path.write_text(
        json.dumps(
            document.export_to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        document.export_to_markdown().rstrip() + "\n",
        encoding="utf-8",
    )
    return {
        "adapter_id": "docling",
        "mode": "formula_enrichment_targeted",
        "model": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "source": source.name,
        "page": page_number,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "do_ocr": False,
        "do_table_structure": False,
        "do_formula_enrichment": True,
        "general_document_vlm_pipeline": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument("--page", type=int)
    parser.add_argument(
        "--prefetch-models",
        type=Path,
        help="Download only layout + CodeFormulaV2 artifacts.",
    )
    args = parser.parse_args()

    if args.prefetch_models is not None:
        prefetch_formula_models(args.prefetch_models.resolve())
        print("DOCLING_FORMULA_ENRICHMENT_PREFETCH=PASS")
        return 0

    if not args.source_pdf or not args.output_dir or args.page is None:
        parser.error(
            "source_pdf, output_dir and --page are required for formula execution"
        )
    if args.page < 1:
        parser.error("--page must be >= 1")

    converter = build_converter()
    result = run_formula_page(
        converter,
        source_pdf=args.source_pdf,
        output_dir=args.output_dir,
        page_number=args.page,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
