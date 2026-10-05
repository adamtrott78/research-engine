#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNTIME_CONTRACT = "qwen35-vllm-cu129-image-native-sampler-v3"


def _respond(payload: dict[str, Any]) -> None:
    print(PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=False,
        disable_log_stats=True,
        mm_processor_cache_gb=0.25,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_tokens=2048,
    )
    seq = 0

    for raw in os.fdopen(os.dup(0), "r", encoding="utf-8"):
        try:
            request = json.loads(raw)
        except Exception as exc:
            _respond(
                {
                    "ok": False,
                    "request_id": -1,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "error": f"invalid request JSON: {type(exc).__name__}: {exc}",
                }
            )
            continue

        request_id = int(request.get("request_id", -1))
        op = request.get("op")
        if op == "shutdown":
            _respond(
                {
                    "ok": True,
                    "request_id": request_id,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "shutdown": True,
                    "runtime_contract": RUNTIME_CONTRACT,
                    "model_revision": MODEL_REVISION,
                }
            )
            break
        if op != "verify":
            _respond(
                {
                    "ok": False,
                    "request_id": request_id,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "error": f"unsupported operation: {op}",
                }
            )
            continue

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            _respond(
                {
                    "ok": False,
                    "request_id": request_id,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "error": "messages must be a non-empty list",
                }
            )
            continue

        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(prompt_ids) + int(sampling.max_tokens) > 8192:
                raise ValueError(
                    "bounded VERIFY prompt plus output reserve exceeds max_model_len"
                )

            started = time.perf_counter()
            outputs = llm.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            wall_time_ms = max(
                0,
                int(round((time.perf_counter() - started) * 1000)),
            )
            request_output = outputs[0]
            generated = request_output.outputs[0]
            parsed = _parse_json_object(generated.text)
            if parsed is None:
                raise ValueError("Qwen VERIFY output was not one JSON object")
            seq += 1
            _respond(
                {
                    "ok": True,
                    "request_id": request_id,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "request_seq": seq,
                    "result": parsed,
                    "usage": {
                        "input_tokens": len(prompt_ids),
                        "output_tokens": len(generated.token_ids),
                    },
                    "wall_time_ms": wall_time_ms,
                    "runtime_contract": RUNTIME_CONTRACT,
                    "model_revision": MODEL_REVISION,
                    "vllm_version": getattr(vllm, "__version__", "unknown"),
                }
            )
        except Exception as exc:
            _respond(
                {
                    "ok": False,
                    "request_id": request_id,
                    "family": "qwen_verify",
                    "server_pid": os.getpid(),
                    "model_load_count": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_contract": RUNTIME_CONTRACT,
                    "model_revision": MODEL_REVISION,
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
