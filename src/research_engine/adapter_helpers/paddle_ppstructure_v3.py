from __future__ import annotations

import json
import sys
from pathlib import Path


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: paddle_ppstructure_v3.py SOURCE_PDF OUTPUT_DIRECTORY",
            file=sys.stderr,
        )
        return 2

    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3()
    results = list(pipeline.predict(input=str(source)))

    markdown_pages = []
    image_groups = []
    page_records = []
    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for index, result in enumerate(results, start=1):
        markdown = result.markdown
        markdown_pages.append(markdown)
        image_groups.append(markdown.get("markdown_images", {}))

        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            payload = {"page_index": index - 1}

        (pages_dir / f"page-{index:04d}.json").write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        page_records.append(
            {
                "page_number": index,
                "structured_json": f"pages/page-{index:04d}.json",
            }
        )

    document_markdown = pipeline.concatenate_markdown_pages(markdown_pages)
    (output / "document.md").write_text(
        document_markdown,
        encoding="utf-8",
    )

    for group in image_groups:
        for relative, image in group.items():
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)

    (output / "adapter-output.json").write_text(
        json.dumps(
            {
                "adapter_id": "paddle-pp-structure-v3",
                "source": source.name,
                "pages": page_records,
                "page_count": len(page_records),
                "document_markdown": "document.md",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
