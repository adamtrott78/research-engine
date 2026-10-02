from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import NativePdfPipelineOptions
    from docling.document_converter import DocumentConverter, NativePdfFormatOption

    pipeline_options = NativePdfPipelineOptions(
        generate_page_images=False,
        generate_picture_images=False,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=pipeline_options,
            )
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    converter = build_converter()

    if args.self_test:
        print("DOCLING_NATIVE_SDK_SELF_TEST=PASS")
        return 0

    if not args.source_pdf or not args.output_dir:
        parser.error("source_pdf and output_dir are required unless --self-test is used")

    source = Path(args.source_pdf).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    result = converter.convert(source)
    document = result.document

    markdown = document.export_to_markdown()
    structured = document.export_to_dict()

    (output / f"{source.stem}.md").write_text(
        markdown.rstrip() + "\n",
        encoding="utf-8",
    )
    (output / f"{source.stem}.json").write_text(
        json.dumps(structured, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "adapter_id": "docling",
                "mode": "native",
                "source": source.name,
                "markdown": f"{source.stem}.md",
                "structured_json": f"{source.stem}.json",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
