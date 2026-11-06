"""Run task-approved ONNX graphs in a minimal, no-network child process.

The trusted verifier retains model-schema checks and scoring. This module only
replaces the two task-specific execution APIs with a numeric, framed IPC proxy.
It deliberately does not use pickle or expose verifier/test/reward directories.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    # Keep the child able to report a framed startup error instead of exiting
    # before the parent receives any protocol data.
    np = None  # type: ignore[assignment]

BWRAP_PATH = "/opt/relay-tools/bwrap"
BUNDLE_PATH = "/opt/relay-isolation"
MODEL_MOUNT = "/opt/relay-onnx/model.onnx"
MAX_MODEL_BYTES = 50 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_FRAME_JSON_BYTES = 1024 * 1024
MAX_ARRAY_RANK = 8
MICROARCH_INPUT_DIR = re.compile(r"/tmp/microarch-input\.[A-Za-z0-9_-]+/model\.onnx\Z")
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
HIDDEN_PATHS = ("/tests", "/logs", "/results", "/root", "/home", "/app", "/work",
                "/mnt", "/media", "/workspace")
_ALLOWED_DTYPE_KINDS = frozenset("buif")
_ALLOWED_TASKS = {"microarch-modeling", "cmb-cross-inference"}
_ORIGINAL_POPEN = subprocess.Popen


class OnnxIsolationError(RuntimeError):
    """The isolated ONNX execution boundary failed closed."""


def _task_slug(task_id: str) -> str:
    if task_id in _ALLOWED_TASKS:
        return task_id
    slug = task_id.rsplit("/", 1)[-1]
    if task_id.count("/") != 1 or not task_id.startswith("terminal-bench-science/") or slug not in _ALLOWED_TASKS:
        raise OnnxIsolationError(f"No ONNX child adapter for Science task {task_id!r}")
    return slug


def allowed_model_path(task_id: str, path: str | os.PathLike[str]) -> Path:
    """Return the exact task-declared ONNX path, rejecting all other locations."""
    slug = _task_slug(task_id)
    raw = os.fspath(path)
    if "\x00" in raw or not Path(raw).is_absolute():
        raise OnnxIsolationError("ONNX model path must be an absolute declared artifact path")
    if slug == "microarch-modeling":
        if not MICROARCH_INPUT_DIR.fullmatch(raw):
            raise OnnxIsolationError("Microarch ONNX path is outside the test.sh model input slot")
    else:
        result_dir = Path(os.environ.get("CMB_RESULTS", "/root/results"))
        if result_dir != Path("/root/results") or raw != "/root/results/extension_model.onnx":
            raise OnnxIsolationError("CMB ONNX path is outside the declared extension model artifact")
    model = Path(raw)
    # Preserve the container-visible mount path for bwrap. Path.resolve() can
    # canonicalize an Apptainer bind to its host source path, which is outside
    # the verifier namespace and must never be passed to the child.
    if any(parent.is_symlink() for parent in model.parents) or model.is_symlink():
        raise OnnxIsolationError("ONNX model path must not contain symbolic links")
    try:
        model.stat()
    except OSError as error:
        raise OnnxIsolationError("ONNX model is missing or inaccessible") from error
    if not model.is_file() or model.stat().st_size <= 0 or model.stat().st_size > MAX_MODEL_BYTES:
        raise OnnxIsolationError("ONNX model is missing, empty, or exceeds its task size limit")
    return model


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _array_bytes(value: Any, *, maximum: int) -> tuple[np.ndarray, bytes]:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise OnnxIsolationError("ONNX IPC accepts plain numeric and boolean arrays only")
    if array.ndim > MAX_ARRAY_RANK:
        raise OnnxIsolationError("ONNX IPC array rank exceeds the adapter limit")
    contiguous = np.ascontiguousarray(array)
    data = contiguous.tobytes(order="C")
    if len(data) > maximum:
        raise OnnxIsolationError("ONNX IPC array exceeds the adapter byte limit")
    return contiguous, data


def _decode_array(descriptor: Mapping[str, Any], data: bytes, *, maximum: int) -> np.ndarray:
    try:
        dtype = np.dtype(descriptor["dtype"])
        shape = descriptor["shape"]
        size = descriptor["nbytes"]
    except (KeyError, TypeError, ValueError) as error:
        raise OnnxIsolationError("Invalid ONNX IPC array descriptor") from error
    if (dtype.hasobject or dtype.fields is not None or dtype.kind not in _ALLOWED_DTYPE_KINDS
            or not isinstance(shape, list) or len(shape) > MAX_ARRAY_RANK
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or type(size) is not int or size < 0 or size > maximum or size != len(data)):
        raise OnnxIsolationError("Invalid ONNX IPC numeric array metadata")
    count = math.prod(shape)
    if count * dtype.itemsize != size:
        raise OnnxIsolationError("ONNX IPC array shape and byte count disagree")
    return np.frombuffer(data, dtype=dtype, count=count).reshape(tuple(shape)).copy()


def _write_all(fd: int, value: bytes, deadline: float) -> None:
    view = memoryview(value)
    while view:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ONNX child input write exceeded its time bound")
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            if not selector.select(remaining):
                raise TimeoutError("ONNX child input write exceeded its time bound")
        try:
            written = os.write(fd, view)
        except BrokenPipeError as error:
            raise OnnxIsolationError("ONNX child exited while reading numeric input") from error
        if written <= 0:
            raise OnnxIsolationError("ONNX child closed its numeric input channel")
        view = view[written:]


class _FrameReader:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.buffer = bytearray()

    def _fill(self, count: int, deadline: float) -> None:
        while len(self.buffer) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ONNX child response exceeded its time bound")
            with selectors.DefaultSelector() as selector:
                selector.register(self.fd, selectors.EVENT_READ)
                if not selector.select(remaining):
                    raise TimeoutError("ONNX child response exceeded its time bound")
            chunk = os.read(self.fd, max(4096, count - len(self.buffer)))
            if not chunk:
                raise OnnxIsolationError("ONNX child closed its response channel")
            self.buffer.extend(chunk)

    def read_frame(self, deadline: float, *, max_payload: int) -> tuple[dict[str, Any], bytes]:
        self._fill(4, deadline)
        header_size = struct.unpack("!I", self.buffer[:4])[0]
        del self.buffer[:4]
        if header_size <= 0 or header_size > MAX_FRAME_JSON_BYTES:
            raise OnnxIsolationError("ONNX child sent an invalid frame header")
        self._fill(header_size, deadline)
        try:
            header = json.loads(bytes(self.buffer[:header_size]))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OnnxIsolationError("ONNX child sent invalid frame metadata") from error
        del self.buffer[:header_size]
        if not isinstance(header, dict):
            raise OnnxIsolationError("ONNX child frame metadata is not an object")
        payload_size = header.get("payload_bytes", 0)
        if type(payload_size) is not int or payload_size < 0 or payload_size > max_payload:
            raise OnnxIsolationError("ONNX child frame exceeds its payload limit")
        self._fill(payload_size, deadline)
        payload = bytes(self.buffer[:payload_size])
        del self.buffer[:payload_size]
        return header, payload


def _send_frame(fd: int, header: dict[str, Any], payload: bytes, deadline: float) -> None:
    header = {**header, "payload_bytes": len(payload)}
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_FRAME_JSON_BYTES:
        raise OnnxIsolationError("ONNX IPC metadata exceeds its size limit")
    _write_all(fd, struct.pack("!I", len(encoded)) + encoded + payload, deadline)


def _encode_feeds(feeds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    if not isinstance(feeds, Mapping) or not feeds or any(not isinstance(key, str) or not key for key in feeds):
        raise OnnxIsolationError("ONNX IPC requires a nonempty name-to-array feed map")
    descriptors: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    total = 0
    for name, value in feeds.items():
        array, raw = _array_bytes(value, maximum=MAX_INPUT_BYTES)
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise OnnxIsolationError("Combined ONNX input arrays exceed the adapter byte limit")
        descriptors.append({"name": name, "dtype": array.dtype.str,
                            "shape": list(array.shape), "nbytes": len(raw)})
        chunks.append(raw)
    return descriptors, b"".join(chunks)


def _decode_outputs(header: dict[str, Any], payload: bytes) -> list[np.ndarray]:
    descriptors = header.get("arrays")
    if not isinstance(descriptors, list):
        raise OnnxIsolationError("ONNX child omitted output array metadata")
    result: list[np.ndarray] = []
    offset = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise OnnxIsolationError("ONNX child output descriptor is invalid")
        size = descriptor.get("nbytes")
        if type(size) is not int or size < 0 or offset + size > len(payload):
            raise OnnxIsolationError("ONNX child output byte range is invalid")
        result.append(_decode_array(descriptor, payload[offset:offset + size], maximum=MAX_OUTPUT_BYTES))
        offset += size
    if offset != len(payload):
        raise OnnxIsolationError("ONNX child returned trailing numeric IPC bytes")
    return result


def _python_prefix_mounts() -> list[str]:
    prefix = Path(sys.prefix).resolve()
    roots = [Path(item).resolve() for item in SYSTEM_PATHS if Path(item).exists()]
    if any(prefix == root or root in prefix.parents for root in roots):
        return []
    if prefix == Path("/") or not prefix.is_dir():
        raise OnnxIsolationError("Cannot identify the verifier Python environment for the ONNX child")
    commands: list[str] = []
    for parent in reversed(prefix.parents):
        if parent == Path("/") or any(parent == root or root in parent.parents for root in roots):
            continue
        commands += ["--dir", str(parent)]
    commands += ["--ro-bind", str(prefix), str(prefix)]
    return commands


def _python_package_paths() -> list[Path]:
    """Find only installed numeric/ONNX package roots, not verifier sys.path."""
    roots: set[Path] = set()
    for name in ("numpy", "onnxruntime", "onnx"):
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None)
        if location:
            path = Path(location).resolve()
            package_root = next((parent for parent in path.parents
                                 if parent.name in {"site-packages", "dist-packages"}), None)
            if package_root is not None and package_root.is_dir():
                roots.add(package_root)
    if not roots:
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        if purelib.is_dir():
            roots.add(purelib)
    if not roots:
        raise OnnxIsolationError("Cannot identify the ONNX verifier package directory")
    return sorted(roots, key=str)


def _python_package_mounts() -> list[str]:
    system_roots = [Path(item).resolve() for item in SYSTEM_PATHS if Path(item).exists()]
    prefix = Path(sys.prefix).resolve()
    commands: list[str] = []
    for root in _python_package_paths():
        if any(root == base or base in root.parents for base in system_roots):
            continue
        if root == prefix or prefix in root.parents:
            continue  # The full isolated Python prefix is already mounted.
        for parent in reversed(root.parents):
            if parent == Path("/") or any(parent == base or base in parent.parents
                                           for base in system_roots):
                continue
            commands += ["--dir", str(parent)]
        commands += ["--ro-bind", str(root), str(root)]
    return commands


def _bubblewrap_command(model: Path, model_sha256: str, kind: str) -> list[str]:
    helper = Path(__file__)
    if not helper.is_file():
        raise OnnxIsolationError("Trusted ONNX child helper is missing from the verifier bundle")
    command = [BWRAP_PATH, "--unshare-user", "--uid", "60001", "--gid", "60001",
               "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
               "--cap-drop", "ALL", "--die-with-parent", "--new-session"]
    for path in SYSTEM_PATHS:
        if Path(path).is_symlink():
            command += ["--symlink", os.readlink(path), path]
        elif Path(path).exists():
            command += ["--ro-bind", path, path]
    # Preserve the logical bind path for the helper before /opt is replaced
    # with a private tmpfs. Resolving __file__ can turn an Apptainer bind into
    # an inaccessible host-side source path.
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", "/tmp/relay-helper", "--ro-bind", str(helper),
                "/tmp/relay-helper/science_onnx_isolation.py",
                "--tmpfs", "/opt", "--tmpfs", "/run", "--tmpfs", "/var"]
    # CMB's declared model is below /root/results. Mount it before masking
    # /root so bwrap can still resolve the source path during setup.
    command += ["--dir", "/opt/relay-onnx", "--ro-bind", str(model), MODEL_MOUNT]
    for path in HIDDEN_PATHS:
        command += ["--dir", path]
    command += ["--dir", "/run/lock", "--dir", "/tmp/onnx-home"]
    command += _python_prefix_mounts()
    command += _python_package_mounts()
    command += ["--remount-ro", "/", "--remount-ro", "/dev", "--chdir", "/tmp",
                "--clearenv", "--setenv", "HOME", "/tmp/onnx-home",
                "--setenv", "TMPDIR", "/tmp", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
                "--setenv", "PYTHONPATH", os.pathsep.join(map(str, _python_package_paths())),
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--setenv", "PYTHONNOUSERSITE", "1",
                "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "--setenv", "RELAY_ONNX_EXPECTED_SHA256", model_sha256,
                "--", sys.executable, "/tmp/relay-helper/science_onnx_isolation.py",
                "--child", "--kind", kind,
                "--model", MODEL_MOUNT]
    return command


class _ChildSession:
    def __init__(self, kind: str, model: Path, *, model_sha256: str,
                 options: dict[str, Any] | None = None) -> None:
        self.kind = kind
        self.model = model
        self.model_sha256 = model_sha256
        self.options = options or {}
        self.owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: _FrameReader | None = None
        self._request_id = 0
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        if _digest(self.model) != self.model_sha256:
            raise OnnxIsolationError("Declared ONNX model changed before isolated execution")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        command = _bubblewrap_command(self.model, self.model_sha256, self.kind)
        process = _ORIGINAL_POPEN(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, close_fds=True, env=env)
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise OnnxIsolationError("Could not create numeric IPC channels to ONNX child")
        self._process = process
        self._reader = _FrameReader(process.stdout.fileno())
        deadline = time.monotonic() + 60
        try:
            header, payload = self._reader.read_frame(deadline, max_payload=0)
        except BaseException as error:
            exit_code = process.poll()
            if exit_code is None:
                try:
                    exit_code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            self._kill()
            if exit_code is not None:
                raise OnnxIsolationError(
                    f"Isolated ONNX child exited before initialization (status {exit_code})"
                ) from error
            raise
        if payload or header.get("status") != "ready" or header.get("model_sha256") != self.model_sha256:
            detail = header.get("error")
            self._kill()
            raise OnnxIsolationError("Isolated ONNX child failed model initialization" +
                                     (f" ({detail})" if isinstance(detail, str) else ""))

    def _reset_after_fork(self) -> None:
        if os.getpid() == self.owner_pid:
            return
        # Do not signal the parent's ONNX child after fork. Closing these
        # duplicated pipe descriptors only drops this process's references.
        if self._process is not None:
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._process = None
            self._reader = None
        self.owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._request_id = 0
        self._start()

    def run(self, output_names: Sequence[str] | None, feeds: Mapping[str, Any]) -> list[np.ndarray]:
        self._reset_after_fork()
        if output_names is not None and (not isinstance(output_names, (list, tuple))
                                         or any(not isinstance(name, str) for name in output_names)):
            raise OnnxIsolationError("ONNX output names must be a list of strings or None")
        descriptors, payload = _encode_feeds(feeds)
        self._request_id += 1
        request_id = self._request_id
        deadline = time.monotonic() + (260 if self.kind == "ort" else 1700)
        header = {"op": "run", "request_id": request_id,
                  "output_names": list(output_names) if output_names is not None else None,
                  "feeds": descriptors}
        with self._lock:
            process = self._process
            reader = self._reader
            if process is None or process.stdin is None or reader is None:
                raise OnnxIsolationError("Isolated ONNX child is not running")
            _send_frame(process.stdin.fileno(), header, payload, deadline)
            response, result_payload = reader.read_frame(deadline, max_payload=MAX_OUTPUT_BYTES)
        if response.get("request_id") != request_id:
            self._kill()
            raise OnnxIsolationError("ONNX child response does not match the active request")
        if response.get("status") != "ok":
            raise OnnxIsolationError("Isolated ONNX inference failed")
        return _decode_outputs(response, result_payload)

    def _kill(self) -> None:
        process, self._process = self._process, None
        self._reader = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        if os.getpid() != self.owner_pid:
            return
        self._kill()


def _session_options(options: Any) -> dict[str, int]:
    if options is None:
        return {"intra_op_num_threads": 1, "inter_op_num_threads": 1}
    intra = getattr(options, "intra_op_num_threads", 1)
    inter = getattr(options, "inter_op_num_threads", 1)
    if type(intra) is not int or type(inter) is not int or intra != 1 or inter != 1:
        raise OnnxIsolationError("Microarch requires the task-approved single-thread ONNX session")
    return {"intra_op_num_threads": intra, "inter_op_num_threads": inter}


class IsolatedInferenceSession:
    """Drop-in for the Microarch test's narrow InferenceSession API."""

    _relay_onnx_isolated = True

    def __init__(self, path_or_bytes: Any, *args: Any, **kwargs: Any) -> None:
        task_id = os.environ.get("RELAY_SCIENCE_TASK_ID", "")
        model = allowed_model_path(task_id, path_or_bytes) if isinstance(path_or_bytes, (str, os.PathLike)) else None
        if model is None or args:
            raise OnnxIsolationError("Microarch ONNX adapter accepts one declared model path")
        options = kwargs.pop("sess_options", None)
        providers = kwargs.pop("providers", None)
        if kwargs or providers != ["CPUExecutionProvider"]:
            raise OnnxIsolationError("Microarch ONNX adapter only permits the CPU provider")
        self._session = _ChildSession("ort", model, model_sha256=_digest(model),
                                      options=_session_options(options))

    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any],
            run_options: Any = None) -> list[np.ndarray]:
        if run_options is not None:
            raise OnnxIsolationError("Microarch ONNX adapter does not accept run options")
        return self._session.run(output_names, input_feed)

    def close(self) -> None:
        self._session.close()


class IsolatedReferenceEvaluator:
    """Drop-in for CMB's ReferenceEvaluator construction and run interface."""

    _relay_onnx_isolated = True

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        task_id = os.environ.get("RELAY_SCIENCE_TASK_ID", "")
        model_path = allowed_model_path(task_id, "/root/results/extension_model.onnx")
        if args or kwargs:
            raise OnnxIsolationError("CMB ONNX adapter accepts the validated model only")
        # Confirm the model object that passed the grader's static checks is the
        # exact graph at the task-declared path before the child reads it.
        import onnx
        on_disk = onnx.load(model_path, load_external_data=False)
        if (model.SerializeToString(deterministic=True)
                != on_disk.SerializeToString(deterministic=True)):
            raise OnnxIsolationError("Validated CMB ONNX graph differs from the declared model file")
        self._session = _ChildSession("reference", model_path, model_sha256=_digest(model_path))

    def run(self, output_names: Sequence[str] | None, feed_inputs: Mapping[str, Any],
            attributes: Any = None, verbose: int = 0, intermediate: bool = False) -> list[np.ndarray]:
        if attributes is not None or verbose != 0 or intermediate:
            raise OnnxIsolationError("CMB ONNX adapter does not permit intermediate graph outputs")
        return self._session.run(output_names, feed_inputs)

    def close(self) -> None:
        self._session.close()


def install(task_id: str) -> None:
    """Patch only the task-approved ONNX execution constructor."""
    slug = _task_slug(task_id)
    if np is None:
        raise OnnxIsolationError("The verifier image does not provide NumPy for numeric IPC")
    if slug == "microarch-modeling":
        import onnxruntime
        if not getattr(onnxruntime.InferenceSession, "_relay_onnx_isolated", False):
            onnxruntime.InferenceSession = IsolatedInferenceSession
        return
    import onnx.reference
    from onnx.reference import reference_evaluator
    if not getattr(reference_evaluator.ReferenceEvaluator, "_relay_onnx_isolated", False):
        reference_evaluator.ReferenceEvaluator = IsolatedReferenceEvaluator
        onnx.reference.ReferenceEvaluator = IsolatedReferenceEvaluator


def _set_child_limits(kind: str) -> None:
    memory = 1536 * 1024 * 1024
    cpu_seconds = 250 if kind == "ort" else 1700
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _read_child_frame(stream: Any) -> tuple[dict[str, Any], bytes]:
    def read_exact(size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = stream.read(size - len(result))
            if not chunk:
                break
            result.extend(chunk)
        return bytes(result)

    raw = read_exact(4)
    if len(raw) != 4:
        raise OnnxIsolationError("ONNX parent closed the request channel")
    header_size = struct.unpack("!I", raw)[0]
    if not 0 < header_size <= MAX_FRAME_JSON_BYTES:
        raise OnnxIsolationError("ONNX parent sent invalid frame metadata")
    header_bytes = read_exact(header_size)
    if len(header_bytes) != header_size:
        raise OnnxIsolationError("ONNX parent sent truncated frame metadata")
    header = json.loads(header_bytes)
    payload_size = header.get("payload_bytes")
    if type(payload_size) is not int or payload_size < 0 or payload_size > MAX_INPUT_BYTES:
        raise OnnxIsolationError("ONNX parent frame exceeds the input limit")
    payload = read_exact(payload_size)
    if len(payload) != payload_size:
        raise OnnxIsolationError("ONNX parent sent truncated numeric data")
    return header, payload


def _child_send(header: dict[str, Any], payload: bytes = b"") -> None:
    encoded = json.dumps({**header, "payload_bytes": len(payload)},
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("!I", len(encoded)) + encoded + payload)
    sys.stdout.buffer.flush()


def _child_main(kind: str, model: Path) -> int:
    if np is None:
        _child_send({"status": "error", "error": "numpy_missing"})
        return 125
    expected = os.environ.get("RELAY_ONNX_EXPECTED_SHA256", "")
    if not expected or _digest(model) != expected:
        raise OnnxIsolationError("ONNX child model hash does not match the trusted parent")
    _set_child_limits(kind)
    if kind == "ort":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        evaluator = ort.InferenceSession(str(model), sess_options=options,
                                         providers=["CPUExecutionProvider"])
    elif kind == "reference":
        import onnx
        from onnx.reference import ReferenceEvaluator
        graph = onnx.load(model, load_external_data=False)
        evaluator = ReferenceEvaluator(graph)
    else:
        raise OnnxIsolationError("Unknown isolated ONNX backend")
    _child_send({"status": "ready", "model_sha256": expected})
    while True:
        try:
            header, payload = _read_child_frame(sys.stdin.buffer)
        except OnnxIsolationError as error:
            if "closed the request channel" in str(error):
                return 0
            raise
        if header.get("op") != "run" or type(header.get("request_id")) is not int:
            raise OnnxIsolationError("ONNX parent sent an unsupported request")
        feeds = {}
        offset = 0
        for descriptor in header.get("feeds", []):
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("name"), str):
                raise OnnxIsolationError("ONNX parent sent an invalid feed name")
            size = descriptor.get("nbytes")
            if type(size) is not int or size < 0 or offset + size > len(payload):
                raise OnnxIsolationError("ONNX parent sent an invalid feed byte range")
            feeds[descriptor["name"]] = _decode_array(
                descriptor, payload[offset:offset + size], maximum=MAX_INPUT_BYTES)
            offset += size
        if offset != len(payload):
            raise OnnxIsolationError("ONNX parent sent trailing feed bytes")
        names = header.get("output_names")
        if names is not None and (not isinstance(names, list) or any(not isinstance(x, str) for x in names)):
            raise OnnxIsolationError("ONNX parent requested invalid output names")
        outputs = evaluator.run(names, feeds)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        descriptors = []
        chunks = []
        total = 0
        for index, value in enumerate(outputs):
            array, raw = _array_bytes(value, maximum=MAX_OUTPUT_BYTES)
            total += len(raw)
            if total > MAX_OUTPUT_BYTES:
                raise OnnxIsolationError("ONNX output exceeds the adapter byte limit")
            descriptors.append({"name": str(index), "dtype": array.dtype.str,
                                "shape": list(array.shape), "nbytes": len(raw)})
            chunks.append(raw)
        _child_send({"status": "ok", "request_id": header["request_id"],
                     "arrays": descriptors}, b"".join(chunks))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--kind", choices=("ort", "reference"))
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if not args.child or args.kind is None or args.model is None:
        parser.error("This module is a trusted parent adapter or isolated child, not a user CLI")
    try:
        return _child_main(args.kind, args.model)
    except BaseException as error:
        try:
            _child_send({"status": "error", "error": type(error).__name__})
        except Exception:
            pass
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
