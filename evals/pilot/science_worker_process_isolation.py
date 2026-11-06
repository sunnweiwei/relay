"""Filesystem boundary for the existing Foraging and Noisy solver workers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TASKS = {"foraging-cognitive-model", "noisy-blackbox-optimization"}
CANONICAL_PREFIX = "terminal-bench-science/"
BWRAP_PATH = "/opt/relay-tools/bwrap"
FORAGING_WORKER = "/tests/worker.py"
NOISY_WORKER = "/tests/solver_worker.py"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)


class WorkerLaunch:
    def __init__(self, command: list[str], environment: dict[str, str],
                 fds: tuple[int, ...], dispatcher: Any):
        self.command = command
        self.environment = environment
        self.pass_fds = fds
        self._dispatcher = dispatcher

    def close(self) -> None:
        dispatcher_fd = self._dispatcher.fileno()
        for descriptor in self.pass_fds:
            if descriptor == dispatcher_fd:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._dispatcher.close()


def task_slug(task_id: str) -> str:
    slug = task_id.removeprefix(CANONICAL_PREFIX) if task_id.startswith(CANONICAL_PREFIX) else task_id
    if task_slug_is_canonical(task_id) and slug in TASKS:
        return slug
    if task_id in TASKS:
        return task_id
    raise ValueError(f"No audited Python worker interface for {task_id!r}")


def task_slug_is_canonical(task_id: str) -> bool:
    return task_id.startswith(CANONICAL_PREFIX) and task_id.count("/") == 1


def _argv_spec(task_id: str, argv: Sequence[str], cwd: str | None,
               environment: Mapping[str, str]) -> tuple[str, str] | None:
    try:
        slug = task_slug(task_id)
    except ValueError:
        return None
    args = [os.fspath(value) for value in argv]
    if len(args) != 3 or not Path(args[0]).is_absolute():
        return None
    if slug == "foraging-cognitive-model":
        if (args[1:] != ["-B", FORAGING_WORKER]
                or (cwd is not None and PurePosixPath(cwd) != PurePosixPath("/app"))
                or environment.get("VERIFY_MODEL", "/app/model.py") != "/app/model.py"):
            return None
        return slug, "/app/model.py"
    if args[1] != NOISY_WORKER:
        return None
    candidate = PurePosixPath(args[2])
    root = candidate.parent
    if (root.parent != PurePosixPath("/tmp") or not root.name.startswith("dfo-solver-")
            or candidate.name != "solver.py"):
        return None
    if cwd is not None and PurePosixPath(cwd) != root:
        return None
    return slug, str(candidate)


def _worker_spec(task_id: str, argv: Sequence[str], cwd: str | None,
                 environment: Mapping[str, str], kwargs: Mapping[str, Any]) -> tuple[str, str] | None:
    spec = _argv_spec(task_id, argv, cwd, environment)
    if spec is None:
        return None
    slug, _candidate = spec
    if slug == "foraging-cognitive-model":
        if (kwargs.get("preexec_fn") is not None
                or kwargs.get("stdin") != subprocess.PIPE
                or kwargs.get("stdout") != subprocess.PIPE
                or kwargs.get("text") is not True or kwargs.get("bufsize") != 1):
            return None
    elif (kwargs.get("preexec_fn") is None or kwargs.get("start_new_session") is not True
          or kwargs.get("stdin") != subprocess.PIPE
          or kwargs.get("stdout") != subprocess.PIPE
          or kwargs.get("stderr") != subprocess.DEVNULL or kwargs.get("bufsize") != 0):
        return None
    return spec


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if not isinstance(argv, (list, tuple)):
        return False
    environment = kwargs.get("env") or os.environ
    return _worker_spec(task_id, argv, kwargs.get("cwd"), environment, kwargs) is not None


def _source_path(source_root: Path, target: str) -> Path:
    path = Path(target)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("worker paths must be absolute and normalized")
    resolved = source_root / path.as_posix().lstrip("/")
    if resolved.is_symlink() or not resolved.exists():
        raise FileNotFoundError("worker source path is unavailable")
    return resolved


def _runtime_mount(executable: str, source_root: Path) -> tuple[Path, str] | None:
    path = Path(executable)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("worker Python path must be absolute")
    resolved = path.resolve()
    if any(resolved.is_relative_to(Path(prefix)) for prefix in ("/usr/bin", "/usr/local/bin")):
        return None
    root = Path("/opt/relay-runtime")
    if resolved.is_relative_to(root):
        source = _source_path(source_root, str(root))
        if not source.is_dir():
            raise FileNotFoundError("portable worker Python runtime is unavailable")
        return source, str(root)
    raise ValueError("worker Python must be system Python or /opt/relay-runtime Python")


def _clean_environment(environment: Mapping[str, str], slug: str) -> dict[str, str]:
    result = {
        key: value for key, value in environment.items()
        if not key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED"))
        and key not in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH",
                        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID",
                        "OPENAI_PROJECT_ID", "ANTHROPIC_API_KEY"}
    }
    result["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if slug == "foraging-cognitive-model":
        result.update(HOME="/tmp", TMPDIR="/tmp", VERIFY_MODEL="/app/model.py")
    return result


def _dispatcher(slug: str, worker: str, candidate: str) -> Any:
    if slug == "foraging-cognitive-model":
        check = "if sys.argv[1:] != ['-B', " + repr(worker) + "]: raise SystemExit(125)\n"
    else:
        check = ("if sys.argv[1:] != [" + repr(worker) + ", "
                 + repr(candidate) + "]: raise SystemExit(125)\n")
    limits = ""
    if slug == "noisy-blackbox-optimization":
        limits = (
            "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))\n"
            "os.umask(0o077)\n"
        )
    source = ("import os, resource, sys\n" + check + limits
              + "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n").encode()
    handle = tempfile.TemporaryFile()
    handle.write(source)
    handle.flush()
    os.lseek(handle.fileno(), 0, os.SEEK_SET)
    return handle


def build_worker_command(
    *, task_id: str, argv: Sequence[str], environment: Mapping[str, str],
    cwd: str | None, uid: int, gid: int, bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"), system_root: Path = Path("/"),
) -> WorkerLaunch:
    spec = _argv_spec(task_id, argv, cwd, environment)
    if spec is None:
        raise ValueError("process is not an audited submitted-code worker")
    slug, candidate = spec
    args = [os.fspath(value) for value in argv]
    worker_path = FORAGING_WORKER if slug == "foraging-cognitive-model" else NOISY_WORKER
    source_root, system_root = Path(source_root), Path(system_root)
    worker_source = _source_path(source_root, worker_path)
    candidate_source = _source_path(source_root, candidate)
    if not worker_source.is_file() or not candidate_source.is_file():
        raise FileNotFoundError("worker or candidate source is not a regular file")
    runtime = _runtime_mount(args[0], source_root)
    worker_fd = os.open(worker_source, os.O_RDONLY)
    candidate_fd = os.open(candidate_source, os.O_RDONLY)
    dispatcher = _dispatcher(slug, worker_path, candidate)
    descriptors = (worker_fd, candidate_fd, dispatcher.fileno())
    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
        "--cap-drop", "ALL", "--die-with-parent",
    ]
    for path in SYSTEM_PATHS:
        host_path = system_root / path.lstrip("/")
        if host_path.is_symlink():
            command += ["--symlink", os.readlink(host_path), path]
        elif host_path.exists():
            command += ["--ro-bind", str(host_path), path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    command += ["--dir", "/opt"]
    if runtime is not None:
        source, target = runtime
        command += ["--ro-bind", str(source), target]
    command += ["--dir", "/tests", "--ro-bind-data", str(worker_fd), worker_path]
    if slug == "foraging-cognitive-model":
        command += ["--tmpfs", "/app", "--chmod", "0777", "/app",
                    "--ro-bind-data", str(candidate_fd), candidate]
        chdir = "/app"
    else:
        root = str(PurePosixPath(candidate).parent)
        command += ["--dir", root, "--ro-bind-data", str(candidate_fd), candidate]
        chdir = cwd or root
    command += ["--ro-bind-data", str(dispatcher.fileno()), "/opt/relay-worker-exec.py"]
    command += ["--remount-ro", "/", "--remount-ro", "/dev", "--chdir", chdir]
    command += ["--", args[0], "/opt/relay-worker-exec.py", *args[1:]]
    return WorkerLaunch(command, _clean_environment(environment, slug), descriptors, dispatcher)
