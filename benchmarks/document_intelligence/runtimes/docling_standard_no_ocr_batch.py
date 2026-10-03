from __future__ import annotations

import argparse
import json
from pathlib import Path

from docling_standard_no_ocr import build_converter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    converter = build_converter()
    rows = []
    for item in plan.get("source_artifacts", []):
        sha = str(item["sha256"])
        source = Path(item["compile_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_root / sha
        target.mkdir(parents=True, exist_ok=True)
        result = converter.convert(source)
        document = result.document
        json_path = target / "document.json"
        md_path = target / "document.md"
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
        md_path.write_text(
            document.export_to_markdown().rstrip() + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "sha256": sha,
                "source_artifact_id": str(item["id"]),
                "source_pdf_path": str(Path(item["source_path"]).resolve()),
                "compile_path": str(source),
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            }
        )
        print(
            json.dumps(
                {"mode": "standard_no_ocr", "sha256": sha},
                sort_keys=True,
            )
        )

    summary = {
        "schema_version": 1,
        "mode": "docling_standard_no_ocr_batch",
        "converter_load_count": 1,
        "documents": len(rows),
        "rows": rows,
    }
    path = output_root / "batch-summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"DOCLING_STANDARD_BATCH_SUMMARY={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
