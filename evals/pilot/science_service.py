"""Authenticated shell service running inside one persistent Science container.

This file uses only the standard library so it can be mounted into any task
image. The host launcher supplies the filesystem and process namespace.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class TaskService:
    def __init__(self, task_id: str, token: str, workdir: Path, trace: Path) -> None:
        self.task_id = task_id
        self.token = token
        self.workdir = workdir
        self.trace = trace
        self.lock = threading.Lock()

    def execute(self, body: dict[str, Any]) -> dict[str, Any]:
        command = body.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 65_536:
            raise ValueError("command must be a nonempty string of at most 65536 characters")
        seconds = float(body.get("timeout", 120))
        if not 0 < seconds <= 300:
            raise ValueError("timeout must be in (0, 300]")
        cwd = Path(body.get("cwd") or self.workdir)
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("cwd must name an existing absolute directory inside the task container")
        started = time.monotonic()
        with self.lock:
            process = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-c", command], cwd=cwd,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
            timed_out = False
            try:
                output, _ = process.communicate(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
            result = {"exit_code": process.returncode, "timeout": timed_out,
                      "elapsed_seconds": time.monotonic() - started,
                      "output": output[:65_536], "output_truncated": len(output) > 65_536}
            with self.trace.open("a") as log:
                log.write(json.dumps({"timestamp": time.time(), "command": command,
                                      "cwd": str(cwd), **result}) + "\n")
            return result


def handler(service: TaskService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self) -> bool:
            if hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + service.token):
                return True
            self.respond(401, {"error": "authorization required"})
            return False

        def do_GET(self) -> None:
            if not self.authorized():
                return
            if self.path == "/health":
                self.respond(200, {"ok": True, "task_id": service.task_id,
                                   "workdir": str(service.workdir)})
            else:
                self.respond(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            if not self.authorized():
                return
            if self.path != "/exec":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131_072:
                    raise ValueError("invalid request size")
                self.respond(200, service.execute(json.loads(self.rfile.read(length))))
            except (ValueError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    service = TaskService(args.task_id, args.token_file.read_text().strip(), args.workdir, args.trace)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(service))
    server.serve_forever()


if __name__ == "__main__":
    main()
