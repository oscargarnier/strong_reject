#!/usr/bin/env python3
"""Serve one persistent job queue from an internet-connected login node.

Run one worker per job queue, for example:

    tmux new -s sr-12345
    python scripts/file_api_worker.py \
        --queue-root /shared/strong_reject/api_queue \
        --job-name pair_attack --job-id 12345 --env-file ~/.strong_reject.env

Requests and responses remain visible in the job directory.
"""

import argparse
import json
import os
import socket
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
import openai


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomically(path: Path, document: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(document, temporary, ensure_ascii=False, indent=2)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def process_request(request_path: Path, output_dir: Path) -> None:
    with request_path.open(encoding="utf-8") as file:
        request = json.load(file)
    response_path = output_dir / request_path.name
    if response_path.exists():
        return

    started_at = _timestamp()
    try:
        operation = request["operation"]
        payload = request["payload"]
        if operation == "completion":
            response = completion(
                payload["model"], messages=payload["messages"], **payload.get("kwargs", {})
            )
            result = response.choices[0].message.content
        elif operation == "moderation":
            with openai.OpenAI() as client:
                response = client.moderations.create(
                    input=payload["input"], **payload.get("kwargs", {})
                )
            result = {"flagged": response.results[0].flagged}
        else:
            raise ValueError(f"Unsupported operation: {operation!r}")

        document = {"ok": True, "result": result, "completed_at": _timestamp()}
    except Exception as error:
        document = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_at": _timestamp(),
        }
    document.update(
        {
            "request_id": request.get("id"),
            "started_at": started_at,
            "worker": {"hostname": socket.gethostname(), "pid": os.getpid()},
        }
    )
    _write_json_atomically(response_path, document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--env-file", required=True, help="Dotenv file containing API credentials")
    args = parser.parse_args()
    load_dotenv(args.env_file)

    job_dir = Path(args.queue_root).expanduser().resolve() / f"{args.job_name}-{args.job_id}"
    input_dir = job_dir / "requests_in"
    output_dir = job_dir / "requests_out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Watching {input_dir}; responses remain in {output_dir}", flush=True)

    while True:
        for request_path in sorted(input_dir.glob("*.json")):
            process_request(request_path, output_dir)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
