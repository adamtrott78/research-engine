from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from docling_formula_enrichment import build_converter, run_formula_page


PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"


def _respond(payload: dict) -> None:
    sys.stdout.write(PREFIX + json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    converter = build_converter()
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
                        "family": "formula",
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
                        "family": "formula",
                        "server_pid": os.getpid(),
                        "model_load_count": model_load_count,
                        "request_seq": request_seq,
                    }
                )
                continue
            if op != "infer":
                raise ValueError(f"unsupported op: {op}")

            request_seq += 1
            result = run_formula_page(
                converter,
                source_pdf=Path(request["source_pdf"]),
                output_dir=Path(request["output_dir"]),
                page_number=int(request["page_number"]),
            )
            _respond(
                {
                    "ok": True,
                    "request_id": request_id,
                    "family": "formula",
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
                    "family": "formula",
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
