import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research_engine.local_inference import (
    LocalInferenceBackpressureError,
    LocalInferenceError,
    LocalInferenceService,
    LocalModelOwner,
    LocalModelProcessSpec,
    local_inference_service_from_env,
)


FAKE_SERVER = r"""
import json
import os
import sys
import time

PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"
family = os.environ.get("FAMILY", "fake")
seq = 0

def respond(payload):
    sys.stdout.write(PREFIX + json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()

for raw in sys.stdin:
    request = json.loads(raw)
    request_id = int(request.get("request_id", -1))
    op = request.get("op")
    if op == "shutdown":
        respond({
            "ok": True,
            "request_id": request_id,
            "family": family,
            "server_pid": os.getpid(),
            "model_load_count": 1,
            "shutdown": True,
        })
        break
    if op != "infer":
        respond({
            "ok": False,
            "request_id": request_id,
            "family": family,
            "server_pid": os.getpid(),
            "model_load_count": 1,
            "error": "unsupported",
        })
        continue
    seq += 1
    sys.stdout.write("model-startup-or-runtime-noise\n")
    sys.stdout.flush()
    time.sleep(float(request.get("delay", 0)))
    respond({
        "ok": True,
        "request_id": request_id,
        "family": family,
        "server_pid": os.getpid(),
        "model_load_count": 1,
        "request_seq": seq,
        "result": {"echo": request.get("value")},
    })
"""


def _owner(tmp_path: Path, *, family: str, max_pending: int = 8) -> LocalModelOwner:
    script = tmp_path / f"{family}-server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return LocalModelOwner(
        LocalModelProcessSpec(
            family=family,
            command=(sys.executable, str(script)),
            environment={"FAMILY": family},
            max_pending=max_pending,
            request_timeout_seconds=5,
            log_path=str(tmp_path / f"{family}.log"),
        )
    )


def test_owner_reuses_one_process_across_concurrent_callers(tmp_path: Path):
    owner = _owner(tmp_path, family="formula")
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(
                    owner.request,
                    {"op": "infer", "value": i, "delay": 0.02},
                )
                for i in range(5)
            ]
            results = [future.result() for future in futures]

        assert len({row["server_pid"] for row in results}) == 1
        assert {row["model_load_count"] for row in results} == {1}
        assert sorted(row["request_seq"] for row in results) == [1, 2, 3, 4, 5]
        health = owner.health()
        assert health["spawn_count"] == 1
        assert health["requests_accepted"] == 5
        assert health["requests_completed"] == 5
        assert health["requests_failed"] == 0
        assert health["queue_depth"] == 0
    finally:
        owner.close()


def test_owner_applies_bounded_backpressure(tmp_path: Path):
    owner = _owner(tmp_path, family="table", max_pending=1)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                owner.request,
                {"op": "infer", "value": "first", "delay": 0.6},
            )
            deadline = time.monotonic() + 2
            while owner.health()["spawn_count"] < 1:
                if time.monotonic() >= deadline:
                    raise AssertionError("first request did not start")
                time.sleep(0.01)

            second = pool.submit(
                owner.request,
                {"op": "infer", "value": "second", "delay": 0.1},
            )
            deadline = time.monotonic() + 2
            while owner.health()["queue_depth"] < 1:
                if time.monotonic() >= deadline:
                    raise AssertionError("second request did not enter queue")
                time.sleep(0.01)

            with pytest.raises(
                LocalInferenceBackpressureError,
                match="queue is full",
            ):
                owner.request(
                    {"op": "infer", "value": "overflow"},
                    timeout_seconds=1,
                )

            assert first.result()["result"]["echo"] == "first"
            assert second.result()["result"]["echo"] == "second"
    finally:
        owner.close()


def test_service_owns_independent_formula_and_table_processes(tmp_path: Path):
    formula = _owner(tmp_path, family="formula")
    table = _owner(tmp_path, family="table")
    service = LocalInferenceService(formula=formula, table=table)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(
                formula.request,
                {"op": "infer", "value": "f", "delay": 0.05},
            )
            b = pool.submit(
                table.request,
                {"op": "infer", "value": "t", "delay": 0.05},
            )
            fr = a.result()
            tr = b.result()

        assert fr["server_pid"] != tr["server_pid"]
        health = service.health()
        assert health["formula"]["spawn_count"] == 1
        assert health["table"]["spawn_count"] == 1
    finally:
        service.close()


def test_env_factory_is_opt_in(monkeypatch):
    monkeypatch.delenv("RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED", raising=False)
    assert local_inference_service_from_env() is None

    monkeypatch.setenv("RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED", "0")
    assert local_inference_service_from_env() is None


def test_env_factory_can_require_explicit_enablement(monkeypatch):
    monkeypatch.setenv("RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED", "0")
    with pytest.raises(LocalInferenceError, match="disabled"):
        local_inference_service_from_env(require_enabled=True)


def test_deterministic_executor_keeps_shared_broker_reference(tmp_path: Path):
    from research_engine.deterministic_executor import DeterministicResearchExecutor

    marker = object()
    executor = DeterministicResearchExecutor(
        store=object(),
        artifact_root=tmp_path / "artifacts",
        arxiv_provider=object(),
        local_inference=marker,
    )
    assert executor.local_inference is marker


def test_runtime_servers_build_models_once_and_use_protocol():
    root = Path(__file__).parents[1]
    formula = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_formula_server.py"
    ).read_text(encoding="utf-8")
    table = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "paddle_table_server.py"
    ).read_text(encoding="utf-8")

    assert formula.count("converter = build_converter()") == 1
    assert formula.count("run_formula_page(") == 1
    assert 'PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"' in formula
    assert "for raw in sys.stdin:" in formula

    assert table.count("pipeline = _pipeline(device=device)") == 1
    assert table.count("run_table_crop(") == 1
    assert 'PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"' in table
    assert "for raw in sys.stdin:" in table


def test_concurrency_runner_is_stopped_daemon_reuses_existing_runtimes():
    root = Path(__file__).parents[1]
    text = (
        root / "deploy" / "run-local-inference-concurrency-gate.sh"
    ).read_text(encoding="utf-8")

    assert "RESEARCH_ENGINE_DAEMON_MUST_REMAIN_STOPPED" in text
    assert "CONCURRENCY_GATE_REQUIRES_TWO_GPUS" in text
    assert "MODEL_FAMILIES_MUST_USE_DISTINCT_GPUS" in text
    assert "RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING=8" in text
    assert "local_inference_benchmark" in text
    assert "pip install -e" in text
    assert "python3 -m venv" not in text
    assert "pip install paddle" not in text
    assert "prefetch" not in text.lower()
    assert "LOCAL_INFERENCE_CONCURRENCY_GATE=PASS" in text
    assert "workspace-control-tmp" in text


def test_daemon_constructs_one_shared_broker_for_deterministic_workers():
    root = Path(__file__).parents[1]
    daemon_text = (root / "src" / "research_engine" / "daemon.py").read_text(
        encoding="utf-8"
    )
    worker_text = (
        root / "src" / "research_engine" / "worker_runtime.py"
    ).read_text(encoding="utf-8")

    assert daemon_text.count("local_inference_service_from_env()") == 1
    assert "local_inference=local_inference" in daemon_text
    assert "self.local_inference = local_inference" in daemon_text
    assert "self.local_inference.close()" in daemon_text
    assert "local_inference=local_inference" in worker_text


def test_shared_gpu_residency_evicts_table_and_qwen_alternately(tmp_path: Path):
    formula = _owner(tmp_path, family="formula")
    table = _owner(tmp_path, family="table")
    qwen = _owner(tmp_path, family="qwen_verify")
    service = LocalInferenceService(
        formula=formula,
        table=table,
        qwen_verify=qwen,
        gpu_bindings={"formula": 0, "table": 1, "qwen_verify": 1},
    )
    try:
        first = service._request_family(
            "table",
            {"op": "infer", "value": "table-first"},
        )
        assert first["result"]["echo"] == "table-first"
        assert table.health()["process_alive"] is True
        assert qwen.health()["process_alive"] is False

        second = service._request_family(
            "qwen_verify",
            {"op": "infer", "value": "qwen"},
        )
        assert second["result"]["echo"] == "qwen"
        assert table.health()["process_alive"] is False
        assert table.health()["eviction_count"] == 1
        assert qwen.health()["process_alive"] is True

        third = service._request_family(
            "table",
            {"op": "infer", "value": "table-second"},
        )
        assert third["result"]["echo"] == "table-second"
        assert qwen.health()["process_alive"] is False
        assert qwen.health()["eviction_count"] == 1
        assert table.health()["process_alive"] is True
        assert table.health()["spawn_count"] == 2
    finally:
        service.close()


def test_qwen_verify_protocol_server_is_one_boot_bounded_service():
    root = Path(__file__).parents[1]
    server = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "qwen35_verify_server.py"
    ).read_text(encoding="utf-8")

    assert server.count("llm = LLM(") == 1
    assert 'RUNTIME_CONTRACT = "qwen35-vllm-cu129-image-native-sampler-v3"' in server
    assert 'MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"' in server
    assert 'if op != "verify":' in server
    assert 'PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"' in server
    assert "max_tokens=2048" in server
    assert "enable_thinking" in server


def test_shared_gpu_same_family_requests_preserve_owner_queue_semantics(tmp_path: Path):
    formula = _owner(tmp_path, family="formula")
    table = _owner(tmp_path, family="table", max_pending=2)
    qwen = _owner(tmp_path, family="qwen_verify")
    service = LocalInferenceService(
        formula=formula,
        table=table,
        qwen_verify=qwen,
        gpu_bindings={"formula": 0, "table": 1, "qwen_verify": 1},
    )
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(
                service._request_family,
                "table",
                {"op": "infer", "value": "first", "delay": 0.35},
            )
            deadline = time.monotonic() + 2
            while table.health()["spawn_count"] < 1:
                if time.monotonic() >= deadline:
                    raise AssertionError("first shared-GPU table request did not start")
                time.sleep(0.01)

            second = pool.submit(
                service._request_family,
                "table",
                {"op": "infer", "value": "second", "delay": 0.05},
            )
            deadline = time.monotonic() + 2
            while table.health()["queue_depth"] < 1:
                if time.monotonic() >= deadline:
                    raise AssertionError("second same-family request did not enter owner queue")
                time.sleep(0.01)

            assert service.health()["gpu_resident_family"][1] == "table"
            assert service.health()["gpu_active_requests"][1] == 2
            assert first.result()["result"]["echo"] == "first"
            assert second.result()["result"]["echo"] == "second"
            assert table.health()["spawn_count"] == 1
            assert table.health()["requests_completed"] == 2
    finally:
        service.close()
