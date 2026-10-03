from __future__ import annotations

import json
import os
import queue
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


RESPONSE_PREFIX = "@@RESEARCH_ENGINE_RESPONSE@@"


class LocalInferenceError(RuntimeError):
    pass


class LocalInferenceBackpressureError(LocalInferenceError):
    pass


@dataclass(frozen=True)
class LocalModelProcessSpec:
    family: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    max_pending: int = 8
    request_timeout_seconds: float = 600.0
    cwd: str | None = None
    log_path: str | None = None

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("family is required")
        if not self.command:
            raise ValueError("command is required")
        if self.max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")


@dataclass
class _PendingRequest:
    request_id: int
    payload: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None


class LocalModelOwner:
    """Own exactly one long-lived subprocess for one heavyweight model family."""

    def __init__(self, spec: LocalModelProcessSpec):
        self.spec = spec
        self._queue: queue.Queue[_PendingRequest | None] = queue.Queue(
            maxsize=spec.max_pending
        )
        self._lock = threading.Lock()
        self._next_request_id = 1
        self._process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self._closed = False
        self._spawn_count = 0
        self._requests_accepted = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._last_response_metadata: dict[str, Any] = {}
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"local-inference:{spec.family}",
            daemon=True,
        )
        self._worker.start()

    def request(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise LocalInferenceError(
                    f"local inference owner {self.spec.family} is closed"
                )
            request_id = self._next_request_id
            self._next_request_id += 1

        pending = _PendingRequest(
            request_id=request_id,
            payload=dict(payload),
        )
        try:
            self._queue.put_nowait(pending)
        except queue.Full as exc:
            raise LocalInferenceBackpressureError(
                f"{self.spec.family} inference queue is full "
                f"(max_pending={self.spec.max_pending})"
            ) from exc

        with self._lock:
            self._requests_accepted += 1

        timeout = (
            self.spec.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not pending.event.wait(timeout):
            raise LocalInferenceError(
                f"{self.spec.family} request {request_id} timed out after "
                f"{timeout:.1f}s"
            )
        if pending.error is not None:
            raise LocalInferenceError(
                f"{self.spec.family} request {request_id} failed: "
                f"{pending.error}"
            ) from pending.error
        assert pending.response is not None
        return pending.response

    def _open_log(self):
        if self.spec.log_path is None:
            return subprocess.DEVNULL
        path = Path(self.spec.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = path.open("a", encoding="utf-8")
        return self._log_handle

    def _ensure_process(self) -> subprocess.Popen[str]:
        proc = self._process
        if proc is not None and proc.poll() is None:
            return proc
        if proc is not None:
            self._process = None

        executable = Path(self.spec.command[0])
        if not executable.is_file():
            raise FileNotFoundError(executable)

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.spec.environment.items()})
        proc = subprocess.Popen(
            list(self.spec.command),
            cwd=self.spec.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._open_log(),
            text=True,
            bufsize=1,
        )
        self._process = proc
        with self._lock:
            self._spawn_count += 1
        return proc

    def _exchange(
        self,
        proc: subprocess.Popen[str],
        pending: _PendingRequest,
    ) -> dict[str, Any]:
        if proc.stdin is None or proc.stdout is None:
            raise LocalInferenceError("model process has no stdio protocol")

        request = dict(pending.payload)
        request["request_id"] = pending.request_id
        proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + self.spec.request_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalInferenceError(
                    f"protocol timeout waiting for {self.spec.family} "
                    f"request {pending.request_id}"
                )
            if proc.poll() is not None:
                raise LocalInferenceError(
                    f"{self.spec.family} server exited with code "
                    f"{proc.returncode}"
                )
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                continue
            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    raise LocalInferenceError(
                        f"{self.spec.family} server closed stdout "
                        f"(exit={proc.returncode})"
                    )
                continue
            line = line.rstrip("\n")
            if not line.startswith(RESPONSE_PREFIX):
                continue
            payload = json.loads(line[len(RESPONSE_PREFIX):])
            if int(payload.get("request_id", -1)) != pending.request_id:
                raise LocalInferenceError(
                    f"{self.spec.family} protocol response id mismatch"
                )
            if not payload.get("ok"):
                raise LocalInferenceError(
                    str(payload.get("error") or "model server error")
                )
            return payload

    def _worker_loop(self) -> None:
        while True:
            pending = self._queue.get()
            if pending is None:
                self._queue.task_done()
                break
            try:
                proc = self._ensure_process()
                response = self._exchange(proc, pending)
                pending.response = response
                with self._lock:
                    self._requests_completed += 1
                    self._last_response_metadata = {
                        key: response.get(key)
                        for key in (
                            "server_pid",
                            "model_load_count",
                            "family",
                            "request_seq",
                        )
                        if key in response
                    }
            except BaseException as exc:
                pending.error = exc
                with self._lock:
                    self._requests_failed += 1
                self._stop_process()
            finally:
                pending.event.set()
                self._queue.task_done()
        self._stop_process()

    def _stop_process(self) -> None:
        proc = self._process
        self._process = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(
                        json.dumps(
                            {
                                "op": "shutdown",
                                "request_id": 0,
                            }
                        )
                        + "\n"
                    )
                    proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=3)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def health(self) -> dict[str, Any]:
        proc = self._process
        with self._lock:
            return {
                "family": self.spec.family,
                "process_pid": (
                    proc.pid if proc is not None and proc.poll() is None else None
                ),
                "process_alive": bool(
                    proc is not None and proc.poll() is None
                ),
                "spawn_count": self._spawn_count,
                "requests_accepted": self._requests_accepted,
                "requests_completed": self._requests_completed,
                "requests_failed": self._requests_failed,
                "queue_depth": self._queue.qsize(),
                "max_pending": self.spec.max_pending,
                "last_response_metadata": dict(self._last_response_metadata),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._queue.put(None)
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            self._stop_process()


class LocalInferenceService:
    """Shared process-local broker for specialized document model owners."""

    def __init__(
        self,
        *,
        formula: LocalModelOwner,
        table: LocalModelOwner,
    ):
        self.formula = formula
        self.table = table
        self._closed = False

    def formula_page(
        self,
        *,
        source_pdf: str | Path,
        page_number: int,
        output_dir: str | Path,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if page_number < 1:
            raise ValueError("page_number must be >= 1")
        return self.formula.request(
            {
                "op": "infer",
                "source_pdf": str(Path(source_pdf).resolve()),
                "page_number": page_number,
                "output_dir": str(Path(output_dir).resolve()),
            },
            timeout_seconds=timeout_seconds,
        )

    def table_crop(
        self,
        *,
        source_pdf: str | Path,
        page_number: int,
        pdf_bbox: list[float] | tuple[float, float, float, float],
        output_dir: str | Path,
        dpi: int = 300,
        padding_pt: float = 6.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if page_number < 1:
            raise ValueError("page_number must be >= 1")
        if len(pdf_bbox) != 4:
            raise ValueError("pdf_bbox must contain four values")
        return self.table.request(
            {
                "op": "infer",
                "source_pdf": str(Path(source_pdf).resolve()),
                "page_number": page_number,
                "pdf_bbox": [float(x) for x in pdf_bbox],
                "output_dir": str(Path(output_dir).resolve()),
                "dpi": int(dpi),
                "padding_pt": float(padding_pt),
            },
            timeout_seconds=timeout_seconds,
        )

    def health(self) -> dict[str, Any]:
        return {
            "enabled": not self._closed,
            "formula": self.formula.health(),
            "table": self.table.health(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.formula.close()
        self.table.close()

    def __enter__(self) -> "LocalInferenceService":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def local_inference_service_from_env(
    *,
    repo_root: str | Path | None = None,
    log_root: str | Path | None = None,
    require_enabled: bool = False,
) -> LocalInferenceService | None:
    enabled = os.getenv("RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED", "0").strip()
    if enabled not in {"0", "1"}:
        raise ValueError("RESEARCH_ENGINE_LOCAL_INFERENCE_ENABLED must be 0 or 1")
    if enabled == "0":
        if require_enabled:
            raise LocalInferenceError("local inference service is disabled")
        return None

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    adam = Path.home() / "adamArchives" / "Adam"
    state = adam / ".research-engine"

    formula_runtime = Path(
        os.getenv(
            "RESEARCH_ENGINE_FORMULA_RUNTIME",
            str(state / "document-runtimes" / "docling-formula-enrichment"),
        )
    )
    table_runtime = Path(
        os.getenv(
            "RESEARCH_ENGINE_TABLE_RUNTIME",
            str(state / "document-runtimes" / "paddle-table-recognition-v2"),
        )
    )
    formula_cache = Path(
        os.getenv(
            "RESEARCH_ENGINE_FORMULA_MODEL_CACHE",
            str(
                state
                / "document-models"
                / "docling-formula-enrichment-2.131.0"
            ),
        )
    )
    table_cache = Path(
        os.getenv(
            "RESEARCH_ENGINE_TABLE_MODEL_CACHE",
            str(
                state
                / "document-models"
                / "paddle-table-recognition-v2-3.7.0"
            ),
        )
    )

    formula_python = formula_runtime / "bin" / "python"
    table_python = table_runtime / "bin" / "python"
    formula_server = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "docling_formula_server.py"
    )
    table_server = (
        root
        / "benchmarks"
        / "document_intelligence"
        / "runtimes"
        / "paddle_table_server.py"
    )
    for path in (
        formula_python,
        table_python,
        formula_server,
        table_server,
        formula_cache,
        table_cache,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    max_pending = _env_int("RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING", 8)
    if max_pending <= 0:
        raise ValueError(
            "RESEARCH_ENGINE_LOCAL_INFERENCE_MAX_PENDING must be positive"
        )
    request_timeout = float(
        os.getenv("RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS", "600")
    )
    if request_timeout <= 0:
        raise ValueError(
            "RESEARCH_ENGINE_LOCAL_INFERENCE_TIMEOUT_SECONDS must be positive"
        )

    formula_gpu = _env_int("RESEARCH_ENGINE_FORMULA_GPU_INDEX", 0)
    table_gpu = _env_int("RESEARCH_ENGINE_TABLE_GPU_INDEX", 1)
    logs = Path(log_root).resolve() if log_root is not None else (
        state / "logs" / "local-inference"
    )

    formula_env = {
        "CUDA_VISIBLE_DEVICES": str(formula_gpu),
        "RESEARCH_ENGINE_DOCLING_DEVICE": "cuda",
        "RESEARCH_ENGINE_DOCLING_ARTIFACTS": str(formula_cache.resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    table_env = {
        "CUDA_VISIBLE_DEVICES": str(table_gpu),
        "PADDLE_PDX_CACHE_HOME": str(table_cache.resolve()),
        "PADDLE_PDX_MODEL_SOURCE": "BOS",
        "RESEARCH_ENGINE_PADDLE_DEVICE": "gpu:0",
    }

    return LocalInferenceService(
        formula=LocalModelOwner(
            LocalModelProcessSpec(
                family="formula",
                command=(str(formula_python), str(formula_server)),
                environment=formula_env,
                max_pending=max_pending,
                request_timeout_seconds=request_timeout,
                cwd=str(root),
                log_path=str(logs / "formula.log"),
            )
        ),
        table=LocalModelOwner(
            LocalModelProcessSpec(
                family="table",
                command=(str(table_python), str(table_server)),
                environment=table_env,
                max_pending=max_pending,
                request_timeout_seconds=request_timeout,
                cwd=str(root),
                log_path=str(logs / "table.log"),
            )
        ),
    )
