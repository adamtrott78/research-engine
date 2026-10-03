from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


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


def build_converter():
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
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

    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=False,
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        enable_remote_services=False,
        accelerator_options=AcceleratorOptions(
            num_threads=int(
                os.getenv(
                    "RESEARCH_ENGINE_DOCLING_THREADS",
                    str(max(1, os.cpu_count() or 1)),
                )
            ),
            device=_device_name(),
        ),
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf")
    parser.add_argument("output_dir")
    parser.add_argument("--page", type=int, required=True)
    args = parser.parse_args()

    if args.page < 1:
        parser.error("--page must be >= 1")

    source = Path(args.source_pdf).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    result = build_converter().convert(
        source,
        page_range=(args.page, args.page),
    )
    document = result.document
    (output / f"{source.stem}.json").write_text(
        json.dumps(
            document.export_to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / f"{source.stem}.md").write_text(
        document.export_to_markdown().rstrip() + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": "tableformer_accurate_targeted",
                "source": source.name,
                "page": args.page,
                "do_ocr": False,
                "do_table_structure": True,
                "tableformer_mode": "accurate",
                "do_formula_enrichment": False,
                "general_document_vlm_pipeline": False,
                "remote_services": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
