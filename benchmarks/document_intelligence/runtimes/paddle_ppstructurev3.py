#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paddleocr import PPStructureV3


def json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--text-recognition-model-name",
        default="en_PP-OCRv4_mobile_rec",
    )
    args = parser.parse_args()

    source = Path(args.source_pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    pipeline = PPStructureV3(
        text_recognition_model_name=args.text_recognition_model_name,
    )
    results = list(pipeline.predict(input=str(source)))

    markdown_pages = []
    pages_json = []
    for page_number, result in enumerate(results, start=1):
        md_info = result.markdown
        markdown_pages.append(md_info)

        page_json = result.json
        if isinstance(page_json, dict):
            page_json = dict(page_json)
            page_json.setdefault("source_page_number", page_number)
        pages_json.append(page_json)

        for relative, image in (md_info.get("markdown_images", {}) or {}).items():
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)

    document_markdown = pipeline.concatenate_markdown_pages(markdown_pages)
    (output_dir / "document.md").write_text(
        document_markdown,
        encoding="utf-8",
    )
    (output_dir / "pages.json").write_text(
        json.dumps(
            pages_json,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
