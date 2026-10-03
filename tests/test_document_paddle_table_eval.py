import json
from pathlib import Path

from research_engine.document_paddle_table_eval import (
    evaluate_paddle_table_fallback,
    html_table_to_grid,
)


SHA = "a" * 64


def _manifest(path: Path):
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "source_pdf_sha256": SHA,
                        "source_pdf_byte_size": 100,
                        "page_count": 2,
                        "difficulty_tags": ["fixture"],
                        "source_location": "papers/case/case.pdf",
                        "mathpix_oracle_location": "papers/case/case.mmd",
                        "mathpix_oracle_sha256": "b" * 64,
                        "mathpix_image_mode": "NO_IMAGES",
                        "expected_features": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_html_table_to_grid_expands_spans_like_paper_package_grid():
    html = """
    <html><body><table>
      <tr><th rowspan="2">A</th><th colspan="2">B</th></tr>
      <tr><td>C</td><td>D</td></tr>
    </table></body></html>
    """
    assert html_table_to_grid(html) == [
        ["A", "B", ""],
        ["", "C", "D"],
    ]


def test_paddle_table_evaluation_accepts_matching_html(tmp_path: Path):
    manifest = tmp_path / "cases.json"
    _manifest(manifest)

    bundle = tmp_path / "bundle"
    oracle = bundle / "papers" / "case" / "case.mmd"
    oracle.parent.mkdir(parents=True)
    oracle.write_text(
        "\\begin{table}\\begin{tabular}{cc}"
        "A & B \\\\ 1 & 2"
        "\\end{tabular}\\end{table}\n",
        encoding="utf-8",
    )

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "case_id": "case",
                        "table_asset_id": "table-0001",
                        "page_number": 1,
                        "reference_index": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paddle_root = tmp_path / "paddle"
    result_dir = paddle_root / "case" / "table-0001"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "res": {
                    "table_res_list": [
                        {
                            "pred_html": (
                                "<html><body><table>"
                                "<tr><td>A</td><td>B</td></tr>"
                                "<tr><td>1</td><td>2</td></tr>"
                                "</table></body></html>"
                            )
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (paddle_root / "paddle-table-output-summary.json").write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "case_id": "case",
                        "table_asset_id": "table-0001",
                        "result_json": "case/table-0001/result.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_paddle_table_fallback(
        case_manifest_path=manifest,
        bundle_root=bundle,
        target_plan_path=plan,
        paddle_output_root=paddle_root,
    )

    assert report["summary"]["target_tables"] == 1
    assert report["summary"]["matched_tables"] == 1
    assert report["summary"]["weighted_content_f1_on_matched"] == 1.0
    assert report["summary"]["weighted_aligned_cell_f1_on_matched"] == 1.0
    assert report["summary"]["decision"] == "PROMOTION_CANDIDATE"


def test_paddle_runtime_is_crop_only_and_keeps_one_pipeline_instance():
    root = Path(__file__).parents[1]
    text = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "paddle_table_recognition_v2.py"
    ).read_text(encoding="utf-8")

    assert "TableRecognitionPipelineV2" in text
    assert "use_layout_detection=False" in text
    assert "use_doc_orientation_classify=False" in text
    assert "use_doc_unwarping=False" in text
    assert '"SLANeXt_wired"' in text
    assert '"SLANeXt_wireless"' in text
    assert '"RT-DETR-L_wired_table_cell_det"' in text
    assert '"RT-DETR-L_wireless_table_cell_det"' in text
    assert '"PP-OCRv5_server_det"' in text
    assert '"PP-OCRv5_server_rec"' in text
    assert text.count("pipeline = _pipeline(") == 1
    assert "PPStructureV3" not in text


def test_paddle_runner_is_stopped_daemon_targeted_and_visible():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-paddle-table-fallback-gate.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert 'visible_tmp_root="${WORKSPACE_CONTROL_TMP_ROOT:-$adam/workspace-control-tmp}"' in text
    assert "EXPECTED_EXACTLY_THREE_PERSISTENT_TABLE_TARGETS" in text
    assert "paddlepaddle-gpu==3.3.0" in text
    assert '"paddleocr[doc-parser]==3.7.0"' in text
    assert "--dpi 300" in text
    assert "--padding-pt 6" in text
    assert "document-benchmark-external" not in text
    assert "marker" not in text.lower()
    assert "mineru" not in text.lower()
    assert "PADDLE_TABLE_FALLBACK_GATE=PASS" in text
