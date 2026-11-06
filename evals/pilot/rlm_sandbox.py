"""Run the official RLM LocalREPL behind the benchmark worker's OS sandbox.

Only JSON crosses this boundary. The original LocalREPL and its persistent
Python namespace live in the child; model calls still use the official local
LMHandler socket, whose client is Relay's quota-guarded task model.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


def clean_environment(env: dict[str, str], work: Path) -> dict[str, str]:
    """Pass runtime paths and locale, never auth tokens or provider variables."""
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "CODEX_HOME"}
    clean = {key: value for key, value in env.items() if key in allowed}
    clean["PATH"] = os.pathsep.join(
        part for part in clean.get("PATH", os.defpath).split(os.pathsep)
        if "/tmp/arg0" not in part
    )
    tmp = work / "rlm_tmp"
    tmp.mkdir(exist_ok=True)
    clean.update(TMPDIR=str(tmp), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return clean


class SandboxREPL:
    """JSON proxy for the LocalREPL API used by rlms 0.1.3."""

    def __init__(
        self, command: list[str], env: dict[str, str], work: Path,
        timeout: float = 1800, **kwargs: Any,
    ) -> None:
        if kwargs.get("subcall_fn") is not None:
            raise ValueError("The pilot sandbox supports the registered depth-1 RLM only")
        self.timeout = timeout
        self.buffer = b""
        self.lock = threading.Lock()
        self.closed = False
        self.stderr_path = work / f"rlm-sandbox-{time.time_ns()}.stderr"
        with self.stderr_path.open("wb") as stderr:
            self.process = subprocess.Popen(
                command, env=env, cwd=work, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, start_new_session=True,
            )
        try:
            self._call("initialize", kwargs)
        except BaseException:
            self.cleanup()
            raise

    def _call(self, method: str, *args: Any) -> Any:
        with self.lock:
            if self.closed or self.process.poll() is not None:
                raise RuntimeError("RLM sandbox process is not running")
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps({"method": method, "args": args}).encode() + b"\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    result = json.loads(line)
                    if "error" in result:
                        raise RuntimeError(f"RLM sandbox: {result['error']}")
                    return result["result"]
                if select.select([self.process.stdout], [], [], min(0.2, max(0, deadline - time.monotonic())))[0]:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(f"RLM sandbox exited without a response; see {self.stderr_path}")
                    self.buffer += chunk
            self.cleanup()
            raise TimeoutError("RLM sandbox execution deadline exceeded")

    def execute_code(self, code: str) -> Any:
        from rlm.core.types import REPLResult, RLMChatCompletion

        value = self._call("execute_code", code)
        value["rlm_calls"] = [RLMChatCompletion.from_dict(row) for row in value["rlm_calls"]]
        return REPLResult(**value)

    def add_context(self, payload: Any, index: int | None = None) -> int:
        return self._call("add_context", payload, index)

    def load_context(self, payload: Any) -> None:
        self._call("load_context", payload)

    def update_handler_address(self, address: tuple[str, int]) -> None:
        self._call("update_handler_address", address)

    def get_context_count(self) -> int:
        return self._call("get_context_count")

    def add_history(self, history: Any, index: int | None = None) -> int:
        return self._call("add_history", history, index)

    def get_history_count(self) -> int:
        return self._call("get_history_count")

    def append_compaction_entry(self, entry: Any) -> None:
        self._call("append_compaction_entry", entry)

    def cleanup(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)


def install_rlm_sandbox(
    *, python: Path, work: Path, env: dict[str, str], timeout: float = 1800,
) -> Callable[[], None]:
    """Install the pilot-only environment proxy and return mandatory cleanup.

    Use a dependency-only Python environment outside the denied reference tree,
    or stage it inside the worker directory. The scratch root, references,
    credentials, and Relay process remain denied. Invoke the returned callback
    in the runner's finally.
    """
    import rlm.core.rlm as core

    worker = work / "rlm_sandbox_worker.py"
    shutil.copyfile(__file__, worker)
    clean = clean_environment(env, work)
    codex = shutil.which("codex", path=clean["PATH"])
    if codex is None:
        raise FileNotFoundError("codex is not available in the sanitized PATH")
    command = [codex, "sandbox", "-P", "pilot", "-C", str(work), "--",
               str(python.absolute()), str(worker), "--worker"]
    original = core.get_environment
    children: list[SandboxREPL] = []

    def factory(name: str, kwargs: dict[str, Any]) -> Any:
        if name != "local":
            raise ValueError("Pilot RLM requires the isolated official LocalREPL")
        child = SandboxREPL(command, clean, work, timeout, **kwargs)
        children.append(child)
        return child

    core.get_environment = factory

    def cleanup() -> None:
        core.get_environment = original
        for child in children:
            child.cleanup()

    return cleanup


def worker_main() -> None:
    from rlm.environments.local_repl import LocalREPL

    # The outer codex process needs this path to load the sandbox configuration;
    # executed code does not need to know where the authentication file resides.
    os.environ.pop("CODEX_HOME", None)
    repl: LocalREPL | None = None
    methods = {"execute_code", "add_context", "load_context", "update_handler_address",
               "get_context_count", "add_history", "get_history_count", "append_compaction_entry"}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                method, args = request["method"], request["args"]
                if method == "initialize" and repl is None:
                    kwargs = args[0]
                    if kwargs.get("lm_handler_address") is not None:
                        kwargs["lm_handler_address"] = tuple(kwargs["lm_handler_address"])
                    repl = LocalREPL(**kwargs)
                    result = None
                elif method in methods and repl is not None:
                    if method == "update_handler_address":
                        args = [tuple(args[0])]
                    result = getattr(repl, method)(*args)
                    if method == "execute_code":
                        result = result.to_dict()
                else:
                    raise ValueError("Unknown or out-of-order RLM sandbox operation")
                response = {"result": result}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response), flush=True)
    finally:
        if repl is not None:
            repl.cleanup()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module is an internal benchmark sandbox worker")
    worker_main()
