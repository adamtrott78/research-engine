from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from paddle_table_recognition_v2 import _pipeline, run_table_crop


PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"


def _respond(payload: dict) -> None:
    sys.stdout.write(PREFIX + json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    device = os.getenv("RESEARCH_ENGINE_PADDLE_DEVICE", "gpu:0")
    pipeline = _pipeline(device=device)
    model_load_count = 1
    request_seq = 0

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = -1
        try:
            request = json.loads(raw)
            request_id = int(request.get("request_id", -1))
            op = str(request.get("op") or "")
            if op == "shutdown":
                _respond(
                    {
                        "ok": True,
                        "request_id": request_id,
                        "family": "table",
                        "server_pid": os.getpid(),
                        "model_load_count": model_load_count,
                        "shutdown": True,
                    }
                )
                return 0
            if op == "health":
                _respond(
                    {
                        "ok": True,
                        "request_id": request_id,
                        "family": "table",
                        "server_pid": os.getpid(),
                        "model_load_count": model_load_count,
                        "request_seq": request_seq,
                    }
                )
                continue
            if op != "infer":
                raise ValueError(f"unsupported op: {op}")

            request_seq += 1
            result = run_table_crop(
                pipeline,
                source_pdf=Path(request["source_pdf"]),
                page_number=int(request["page_number"]),
                pdf_bbox=[float(x) for x in request["pdf_bbox"]],
                output_dir=Path(request["output_dir"]),
                dpi=int(request.get("dpi", 300)),
                padding_pt=float(request.get("padding_pt", 6.0)),
            )
            _respond(
                {
                    "ok": True,
                    "request_id": request_id,
                    "family": "table",
                    "server_pid": os.getpid(),
                    "model_load_count": model_load_count,
                    "request_seq": request_seq,
                    "result": result,
                }
            )
        except Exception as exc:
            _respond(
                {
                    "ok": False,
                    "request_id": request_id,
                    "family": "table",
                    "server_pid": os.getpid(),
                    "model_load_count": model_load_count,
                    "request_seq": request_seq,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
