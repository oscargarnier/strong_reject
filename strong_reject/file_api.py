"""Persistent file-backed transport for API calls from isolated jobs."""

import json
import os
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_job_queue: Path | None = None
_timeout = 3600.0
_poll_interval = 2.0


def configure(
    queue_root: str,
    job_name: str | None = None,
    job_id: str | None = None,
    timeout: float = 3600.0,
    poll_interval: float = 2.0,
) -> Path:
    """Configure a persistent queue for one Slurm job."""
    global _job_queue, _timeout, _poll_interval
    job_name = job_name or os.getenv("SLURM_JOB_NAME", "interactive")
    job_id = job_id or os.getenv("SLURM_JOB_ID", str(os.getpid()))
    job_directory = Path(queue_root).expanduser().resolve() / f"{job_name}-{job_id}"
    (job_directory / "requests_in").mkdir(parents=True, exist_ok=True)
    (job_directory / "requests_out").mkdir(parents=True, exist_ok=True)
    _job_queue = job_directory
    _timeout = timeout
    _poll_interval = poll_interval
    os.environ["STRONG_REJECT_FILE_API_QUEUE"] = str(job_directory)
    os.environ["STRONG_REJECT_FILE_API_TIMEOUT"] = str(timeout)
    os.environ["STRONG_REJECT_FILE_API_POLL_INTERVAL"] = str(poll_interval)
    return job_directory


def enabled() -> bool:
    _configure_from_environment()
    return _job_queue is not None


def request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Submit one request and wait without deleting queue history."""
    _configure_from_environment()
    if _job_queue is None:
        raise RuntimeError("File API transport has not been configured.")

    request_id = uuid.uuid4().hex
    request_path = _job_queue / "requests_in" / f"{request_id}.json"
    response_path = _job_queue / "requests_out" / f"{request_id}.json"
    document = {
        "id": request_id,
        "operation": operation,
        "payload": payload,
        "created_at": _timestamp(),
        "client": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        },
    }
    _write_json_atomically(request_path, document)

    deadline = time.monotonic() + _timeout
    while time.monotonic() < deadline:
        if response_path.exists():
            response = _read_json(response_path)
            if not response.get("ok", False):
                raise RuntimeError(_format_failure(response, request_path))
            return response["result"]
        time.sleep(_poll_interval)

    failure = {
        "ok": False,
        "error": f"Timed out after {_timeout:g}s waiting for API response.",
        "request_id": request_id,
        "request_path": str(request_path),
        "created_at": document["created_at"],
        "failed_at": _timestamp(),
        "client": document["client"],
    }
    failure_path = _job_queue / "requests_out" / f"{request_id}.timeout.json"
    _write_json_atomically(failure_path, failure)
    raise TimeoutError(_format_failure(failure, request_path))


def _format_failure(response: dict[str, Any], request_path: Path) -> str:
    return (
        f"Internet-free API request failed at {response.get('failed_at', _timestamp())}. "
        f"Request: {request_path}. Error: {response.get('error', 'unknown error')}. "
        f"Metadata: {json.dumps(response.get('worker', response.get('client', {})))}"
    )


def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(document, temporary, ensure_ascii=False, indent=2)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_from_environment() -> None:
    global _job_queue, _timeout, _poll_interval
    if _job_queue is None and os.getenv("STRONG_REJECT_FILE_API_QUEUE"):
        queue = Path(os.environ["STRONG_REJECT_FILE_API_QUEUE"])
        (queue / "requests_in").mkdir(parents=True, exist_ok=True)
        (queue / "requests_out").mkdir(parents=True, exist_ok=True)
        _job_queue = queue
        _timeout = float(os.getenv("STRONG_REJECT_FILE_API_TIMEOUT", _timeout))
        _poll_interval = float(os.getenv("STRONG_REJECT_FILE_API_POLL_INTERVAL", _poll_interval))
