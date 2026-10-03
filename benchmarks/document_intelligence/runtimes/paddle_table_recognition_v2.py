from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _render_crop(
    *,
    source_pdf: Path,
    page_number: int,
    bbox: list[float],
    output_path: Path,
    dpi: int,
    padding_pt: float,
) -> dict[str, Any]:
    import pypdfium2 as pdfium

    if page_number < 1:
        raise ValueError("page_number must be >= 1")
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")

    pdf = pdfium.PdfDocument(str(source_pdf))
    try:
        page = pdf[page_number - 1]
        scale = dpi / 72.0
        image = page.render(scale=scale).to_pil().convert("RGB")
    finally:
        try:
            page.close()
        except Exception:
            pass
        pdf.close()

    left, top, right, bottom = [float(x) for x in bbox]
    if right <= left or bottom <= top:
        raise ValueError(f"invalid bbox: {bbox!r}")

    left -= padding_pt
    top -= padding_pt
    right += padding_pt
    bottom += padding_pt

    crop_box = (
        max(0, math.floor(left * scale)),
        max(0, math.floor(top * scale)),
        min(image.width, math.ceil(right * scale)),
        min(image.height, math.ceil(bottom * scale)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"empty rendered crop for bbox {bbox!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(output_path)
    return {
        "dpi": dpi,
        "padding_pt": padding_pt,
        "source_bbox_top_left_points": bbox,
        "crop_box_pixels": list(crop_box),
        "crop_size_pixels": [
            crop_box[2] - crop_box[0],
            crop_box[3] - crop_box[1],
        ],
    }


def _pipeline(*, device: str):
    # Cache/source environment must be set before importing PaddleOCR.
    from paddleocr import TableRecognitionPipelineV2

    return TableRecognitionPipelineV2(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        wired_table_structure_recognition_model_name="SLANeXt_wired",
        wireless_table_structure_recognition_model_name="SLANeXt_wireless",
        table_classification_model_name="PP-LCNet_x1_0_table_cls",
        wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
        wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
        enable_hpi=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_plan")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--padding-pt", type=float, default=6.0)
    parser.add_argument("--prefetch-only", action="store_true")
    args = parser.parse_args()

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(args.model_cache).resolve())
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

    plan = json.loads(Path(args.target_plan).read_text(encoding="utf-8"))
    bundle_root = Path(args.bundle_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = _pipeline(device=args.device)
    if args.prefetch_only:
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "mode": "paddle_table_recognition_v2_crop",
                    "device": args.device,
                    "model_cache": os.environ["PADDLE_PDX_CACHE_HOME"],
                    "target_tables": len(plan.get("targets", [])),
                },
                sort_keys=True,
            )
        )
        print("PADDLE_TABLE_RECOGNITION_V2_PREFETCH=PASS")
        return 0

    outputs: list[dict[str, Any]] = []
    for target in plan.get("targets", []):
        case_id = str(target["case_id"])
        table_asset_id = str(target["table_asset_id"])
        page_number = int(target["page_number"])
        bbox = target.get("pdf_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"{case_id}/{table_asset_id}: missing four-value pdf_bbox"
            )

        source_pdf = bundle_root / str(target["source_location"])
        if not source_pdf.is_file():
            raise FileNotFoundError(source_pdf)

        target_dir = output_root / case_id / table_asset_id
        crop_path = target_dir / "table-crop.png"
        crop_meta = _render_crop(
            source_pdf=source_pdf,
            page_number=page_number,
            bbox=bbox,
            output_path=crop_path,
            dpi=args.dpi,
            padding_pt=args.padding_pt,
        )

        results = list(
            pipeline.predict(
                input=str(crop_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_ocr_model=True,
            )
        )
        if len(results) != 1:
            raise RuntimeError(
                f"{case_id}/{table_asset_id}: expected one result, got {len(results)}"
            )

        payload = results[0].json
        if callable(payload):
            payload = payload()
        result_path = target_dir / "result.json"
        result_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

        outputs.append(
            {
                "case_id": case_id,
                "table_asset_id": table_asset_id,
                "page_number": page_number,
                "reference_index": int(target["reference_index"]),
                "crop_path": str(crop_path.relative_to(output_root)),
                "result_json": str(result_path.relative_to(output_root)),
                "crop": crop_meta,
            }
        )
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "table_asset_id": table_asset_id,
                    "page_number": page_number,
                    "result_json": str(result_path),
                },
                sort_keys=True,
            )
        )

    summary = {
        "schema_version": 1,
        "mode": "paddle_table_recognition_v2_crop",
        "paddleocr_version": "3.7.0",
        "paddlepaddle_version": "3.3.0",
        "device": args.device,
        "layout_detection": False,
        "doc_orientation_classify": False,
        "doc_unwarping": False,
        "ocr": True,
        "target_tables": len(outputs),
        "outputs": outputs,
    }
    summary_path = output_root / "paddle-table-output-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PADDLE_TABLE_OUTPUT_SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
