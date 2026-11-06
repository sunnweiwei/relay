"""JSON-only function proxy for the traffic-flux-inversion candidate API.

The official verifier imports ``/app/traffic_solver.py`` in its unprivileged
harness. This loader exposes the same two functions in the harness process,
then forwards typed numeric/JSON requests to one persistent candidate worker
inside a private bubblewrap namespace. No Python object serialization is used.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import secrets
import importlib.util
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


TASK_SLUG = "traffic-flux-inversion"
TASK_ID = "terminal-bench-science/traffic-flux-inversion"
MODULE_NAME = "candidate_traffic_solver"
CANDIDATE_PATH = "/app/traffic_solver.py"
BUNDLE_PATH = "/opt/relay-isolation"
BWRAP_PATH = "/opt/relay-tools/bwrap"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
WORKER_READY_TIMEOUT_SECONDS = 15.0
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share", "/usr/include",
)


class TrafficProxyError(RuntimeError):
    """A bounded JSON request failed in the isolated candidate worker."""


class _TrafficModelView(dict[str, Any]):
    """Official ``form``/``v`` view with the model retained in the worker."""

    __slots__ = ("_worker_id", "_model_id")

    def __init__(self, worker_id: str, model_id: str, form: Any, value: Any) -> None:
        super().__init__(form=form, v=value)
        self._worker_id = worker_id
        self._model_id = model_id


def _task_matches(task_id: str) -> bool:
    return task_id in {TASK_ID, TASK_SLUG}


def _valid_episode_path(path: Any) -> bool:
    return isinstance(path, str) and re.fullmatch(r"/work/system_\d+_episodes\.npz", path) is not None


def _encode(value: Any) -> Any:
    """Encode JSON primitives and numeric arrays with explicit type metadata."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not supported by the traffic JSON protocol")
        return {"$": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape),
                "data": _encode(value.tolist())}
    if isinstance(value, np.generic):
        return {"$": "npscalar", "dtype": value.dtype.str, "value": _encode(value.item())}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$": "float", "value": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("model dictionaries must use string keys for JSON IPC")
        return {"$": "dict", "items": [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, list):
        return {"$": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, tuple):
        return {"$": "tuple", "items": [_encode(item) for item in value]}
    raise TypeError(f"unsupported traffic JSON value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    """Decode the tagged JSON numeric protocol without object deserialization."""
    import numpy as np

    if not isinstance(value, dict) or "$" not in value:
        return value
    kind = value["$"]
    if kind == "float":
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[value["value"]]
    if kind == "dict":
        return {key: _decode(item) for key, item in value["items"]}
    if kind == "list":
        return [_decode(item) for item in value["items"]]
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "ndarray":
        array = np.asarray(_decode(value["data"]), dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(int(dim) for dim in value["shape"]))
    if kind == "npscalar":
        decoded = _decode(value["value"])
        return np.asarray(decoded, dtype=np.dtype(value["dtype"])).reshape(())[()]
    raise ValueError("unknown tagged traffic JSON value")


def _json_line(value: Any) -> bytes:
    line = (json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()
    if len(line) > MAX_MESSAGE_BYTES:
        raise ValueError("traffic JSON message exceeds the audited size limit")
    return line


def _clean_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result = {
        key: value for key, value in environment.items()
        if not key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED"))
        and key not in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH",
                        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    }
    result.update(PATH="/usr/local/bin:/usr/bin:/bin", HOME="/tmp", TMPDIR="/tmp", TEMP="/tmp", TMP="/tmp")
    return result


def _copy_public_work(work_root: Path, destination: Path) -> None:
    if not work_root.is_dir() or work_root.is_symlink():
        raise FileNotFoundError("traffic public work directory is missing")
    destination.mkdir(mode=0o700)
    for item in work_root.iterdir():
        if item.is_symlink() or item.is_dir():
            raise ValueError("symlinked or nested traffic work inputs are unsupported")
        if item.name != "inputs.npz" and re.fullmatch(r"system_\d+_episodes\.npz", item.name) is None:
            continue
        if not item.is_file():
            raise ValueError("traffic public inputs must be regular files")
        shutil.copy2(item, destination / item.name)


def build_worker_command(
    *, app_root: Path, work_root: Path, work_copy: Path, worker_script: Path,
    python: str, bwrap: str = BWRAP_PATH, uid: int = 60001, gid: int = 60001,
    source_root: Path = Path("/"),
) -> list[str]:
    """Build the exact child namespace with candidate code and public inputs only."""
    app_root = Path(app_root)
    source_root = Path(source_root)
    if app_root != Path("/app") or Path(work_root) != Path("/work"):
        raise ValueError("traffic interface paths must remain /app and /work")
    app_source = source_root / app_root.as_posix().lstrip("/")
    if app_source.is_symlink() or not app_source.is_dir():
        raise FileNotFoundError("traffic app directory is missing")
    candidate = app_source / "traffic_solver.py"
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError("traffic solver artifact is missing or symlinked")
    if Path(worker_script).name != "science_traffic_isolation.py":
        raise ValueError("unexpected traffic worker script")
    if not Path(python).is_absolute() or not Path(python).exists():
        raise ValueError("traffic worker needs an absolute Python executable")
    resolved_python = Path(python).resolve()
    if not any(resolved_python.is_relative_to(Path(prefix)) for prefix in ("/usr/bin", "/usr/local/bin")):
        raise ValueError("traffic worker Python must be part of the verifier image")
    if not Path(worker_script).is_file():
        raise FileNotFoundError("traffic worker script is missing")
    worker_source = worker_script
    work_source = work_copy
    if not app_source.is_dir() or not worker_source.is_file() or not work_source.is_dir():
        raise FileNotFoundError("traffic namespace source paths are incomplete")

    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
        "--cap-drop", "ALL", "--die-with-parent", "--new-session",
    ]
    for path in SYSTEM_PATHS:
        if Path(path).is_symlink():
            command += ["--symlink", os.readlink(path), path]
        elif Path(path).exists():
            command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in ("/app", "/work", "/tests", "/logs", "/results", "/home", "/var/tmp", "/run/lock"):
        command += ["--dir", path]
    command += ["--ro-bind", str(app_source), "/app"]
    command += ["--bind", str(work_source), "/work"]
    command += ["--dir", "/opt", "--dir", BUNDLE_PATH]
    command += ["--ro-bind", str(worker_source), BUNDLE_PATH + "/science_traffic_isolation.py"]
    command += ["--remount-ro", "/", "--remount-ro", "/dev", "--chdir", "/"]
    command += ["--", python, BUNDLE_PATH + "/science_traffic_isolation.py", "--worker",
                CANDIDATE_PATH]
    return command


class TrafficFunctionProxy:
    """Expose the task's two function calls over one persistent JSON worker."""

    def __init__(
        self, *, task_id: str = TASK_SLUG, app_root: Path = Path("/app"),
        work_root: Path = Path("/work"), worker_script: Path | None = None,
        python: str | None = None, bwrap: str = BWRAP_PATH,
        source_root: Path = Path("/"),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not _task_matches(task_id):
            raise ValueError("traffic proxy used for a different Science task")
        self.app_root = Path(app_root)
        self.work_root = Path(work_root)
        self.worker_script = Path(worker_script or __file__)
        self.python = python or sys.executable
        self.bwrap = bwrap
        self.source_root = Path(source_root)
        self.environment = _clean_environment(environment or os.environ)
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._worker_id = secrets.token_hex(16)
        self._active_model_id: str | None = None
        atexit.register(self.close)

    def _start(self) -> None:
        if self._process is not None:
            return
        self._temp = tempfile.TemporaryDirectory(prefix="relay-traffic-worker-")
        work_copy = Path(self._temp.name) / "work"
        live_work = self.source_root / self.work_root.as_posix().lstrip("/")
        _copy_public_work(live_work, work_copy)
        command = build_worker_command(
            app_root=self.app_root, work_root=self.work_root, work_copy=work_copy,
            worker_script=self.worker_script, python=self.python, bwrap=self.bwrap,
            source_root=self.source_root,
        )
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=self.environment, cwd=None, close_fds=True, start_new_session=True,
        )
        ready = self._readline(WORKER_READY_TIMEOUT_SECONDS)
        try:
            message = json.loads(ready)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self.close()
            raise TrafficProxyError("traffic worker did not complete its private startup") from error
        if message != {"ready": True}:
            self.close()
            raise TrafficProxyError("traffic worker startup failed")

    def _readline(self, timeout: float | None = None) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise TrafficProxyError("traffic worker is not running")
        if timeout is not None:
            ready, _, _ = select.select([process.stdout], [], [], timeout)
            if not ready:
                raise TrafficProxyError("traffic worker startup timed out")
        line = process.stdout.readline(MAX_MESSAGE_BYTES + 1)
        if len(line) > MAX_MESSAGE_BYTES:
            self.close()
            raise TrafficProxyError("traffic worker response exceeds the audited size limit")
        if not line:
            raise TrafficProxyError("traffic worker exited before replying")
        return line

    def _request(self, request: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise TrafficProxyError("traffic worker is closed")
            self._start()
            assert self._process is not None and self._process.stdin is not None
            try:
                self._process.stdin.write(_json_line(request))
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self.close()
                raise TrafficProxyError("traffic worker request failed") from error
            try:
                response = json.loads(self._readline())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                self.close()
                raise TrafficProxyError("traffic worker returned invalid JSON") from error
            if not isinstance(response, dict) or response.get("ok") is not True:
                error_type = response.get("error_type", "CandidateError") if isinstance(response, dict) else "CandidateError"
                raise TrafficProxyError(f"candidate call failed ({error_type})")
            return _decode(response.get("value"))

    def fit_model(self, episode_file: str) -> Any:
        if not _valid_episode_path(episode_file):
            raise ValueError("fit_model path is outside the official public episode interface")
        path = self.source_root / Path(episode_file).as_posix().lstrip("/")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("fit_model public episode input is unavailable")
        result = self._request({"op": "fit", "episode_file": episode_file})
        if not isinstance(result, dict) or not isinstance(result.get("model_id"), str):
            raise TrafficProxyError("traffic worker returned an invalid model handle")
        self._active_model_id = result["model_id"]
        return _TrafficModelView(self._worker_id, result["model_id"], result["form"], result["v"])

    def predict_scenario(
        self, model: Any, init: Any, left: Any, right: Any,
        steps: int, dt: float, dx: float,
    ) -> Any:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("predict_scenario steps must be an integer")
        if (not isinstance(model, _TrafficModelView) or model._worker_id != self._worker_id
                or model._model_id != self._active_model_id):
            raise TypeError("predict_scenario requires the active model from this traffic worker")
        return self._request({
            "op": "predict", "model_id": model._model_id, "init": _encode(init),
            "left": _encode(left), "right": _encode(right),
            "steps": _encode(steps), "dt": _encode(dt), "dx": _encode(dx),
        })

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.stdin is not None and process.poll() is None:
                    process.stdin.write(_json_line({"op": "shutdown"}))
                    process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None


class _ProxyLoader:
    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        proxy = TrafficFunctionProxy()
        module.fit_model = proxy.fit_model
        module.predict_scenario = proxy.predict_scenario
        module.close = proxy.close
        module.__traffic_proxy__ = proxy


_ORIGINAL_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_INSTALLED = False


def install(task_id: str) -> None:
    """Replace only the verifier's exact candidate module load with the proxy."""
    global _INSTALLED
    if not _task_matches(task_id):
        raise ValueError("traffic isolation hook used for a different Science task")
    if _INSTALLED:
        return

    def spec_from_file_location(name: str, location: Any, *args: Any, **kwargs: Any):
        spec = _ORIGINAL_SPEC_FROM_FILE_LOCATION(name, location, *args, **kwargs)
        try:
            exact_path = os.path.normpath(os.fspath(location)) == CANDIDATE_PATH
        except TypeError:
            exact_path = False
        if spec is not None and name == MODULE_NAME and exact_path:
            spec.loader = _ProxyLoader()
        return spec

    importlib.util.spec_from_file_location = spec_from_file_location
    _INSTALLED = True


def _worker_main(candidate_path: str) -> int:
    import importlib.util as util

    if candidate_path != CANDIDATE_PATH:
        return 125
    candidate = Path(candidate_path)
    if candidate.is_symlink() or not candidate.is_file():
        return 125
    sys.path.insert(0, "/app")
    spec = util.spec_from_file_location(MODULE_NAME, candidate_path)
    if spec is None or spec.loader is None:
        return 125
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                spec.loader.exec_module(module)
    except BaseException:
        return 125

    protocol_in = sys.stdin.buffer
    protocol_out = os.fdopen(os.dup(1), "wb", buffering=0)
    protocol_out.write(b'{"ready":true}\n')
    models: dict[str, Any] = {}
    for raw in protocol_in:
        if len(raw) > MAX_MESSAGE_BYTES:
            protocol_out.write(b'{"ok":false,"error_type":"RequestTooLarge"}\n')
            continue
        try:
            request = json.loads(raw)
            operation = request.get("op")
            if operation == "shutdown":
                return 0
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    if operation == "fit":
                        episode_file = request.get("episode_file")
                        if not _valid_episode_path(episode_file):
                            raise ValueError("invalid episode path")
                        # The official harness fits one system, predicts all
                        # its scenarios, then replaces the model for the next
                        # system. Match that lifetime instead of retaining
                        # every fitted model for the whole verifier process.
                        models.clear()
                        gc.collect()
                        model_id = secrets.token_hex(16)
                        fitted = module.fit_model(episode_file)
                        models[model_id] = fitted
                        if not isinstance(fitted, dict) or not {"form", "v"}.issubset(fitted):
                            raise TypeError("fit_model must return the official form/v dictionary")
                        form, model_value = fitted["form"], fitted["v"]
                        value = {"model_id": model_id, "form": form, "v": model_value}
                    elif operation == "predict":
                        model_id = request.get("model_id")
                        if not isinstance(model_id, str) or model_id not in models:
                            raise ValueError("unknown model handle")
                        value = module.predict_scenario(
                            models[model_id], _decode(request["init"]),
                            _decode(request["left"]), _decode(request["right"]),
                            _decode(request["steps"]), _decode(request["dt"]), _decode(request["dx"]),
                        )
                    else:
                        raise ValueError("unknown operation")
            response = {"ok": True, "value": _encode(value)}
            line = _json_line(response)
        except BaseException as error:
            line = _json_line({"ok": False, "error_type": type(error).__name__})
        protocol_out.write(line)
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return _worker_main(sys.argv[2])
    return 125


if __name__ == "__main__":
    raise SystemExit(main())

