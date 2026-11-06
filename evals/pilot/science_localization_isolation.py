"""Outer bubblewrap boundary for the three JSON solver-worker tasks.

The official test runners keep their own function signatures, serialization,
validation, result-size caps, and call deadlines. This module only wraps their
exact ``/tests/solver_worker.py`` child in a private filesystem and PID/network
namespace.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TASK_PREFIXES = {
    "baseline-free-localization": "baseline-free-worker-",
    "guided-wave-localization": "guided-wave-worker-",
    "virtual-baseline-localization": "virtual-baseline-worker-",
}
CANONICAL_PREFIX = "terminal-bench-science/"
BWRAP_PATH = "/opt/relay-tools/bwrap"
WORKER_PATH = "/tests/solver_worker.py"
EXEC_PATH = "/tests/relay_localization_exec.py"
UV_CACHE_ROOT = Path("/tmp/uv-cache")
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)


class WorkerLaunch:
    """Prepared isolated command and inherited read-only dispatcher files."""

    def __init__(self, command: list[str], environment: dict[str, str], fds: tuple[int, ...],
                 dispatcher_file: Any):
        self.command = command
        self.environment = environment
        self.pass_fds = fds
        self._dispatcher_file = dispatcher_file

    def close(self) -> None:
        dispatcher_fd = self._dispatcher_file.fileno()
        for descriptor in self.pass_fds:
            if descriptor == dispatcher_fd:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._dispatcher_file.close()


def _dispatcher_fd() -> tuple[int, Any]:
    """Create the small in-namespace limit shim without a writable mount."""
    source = (
        "import os, resource, sys\n"
        "if len(sys.argv) != 3 or sys.argv[1] != '/tests/solver_worker.py':\n"
        "    raise SystemExit(125)\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (4, 4))\n"
        "os.umask(0o077)\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    ).encode()
    handle = tempfile.TemporaryFile()
    handle.write(source)
    handle.flush()
    os.lseek(handle.fileno(), 0, os.SEEK_SET)
    return handle.fileno(), handle


def task_slug(task_id: str) -> str:
    if task_id in TASK_PREFIXES:
        return task_id
    if task_id.startswith(CANONICAL_PREFIX) and task_id.count("/") == 1:
        slug = task_id.removeprefix(CANONICAL_PREFIX)
        if slug in TASK_PREFIXES:
            return slug
    raise ValueError(f"No audited localization worker for {task_id!r}")


def _worker_paths(
    task_id: str, argv: Sequence[str], cwd: str | None,
) -> tuple[str, str] | None:
    try:
        slug = task_slug(task_id)
    except ValueError:
        return None
    args = [os.fspath(value) for value in argv]
    if len(args) != 3 or args[1] != WORKER_PATH:
        return None
    if not args[0].startswith("/") or not args[2].startswith("/"):
        return None
    worker_temp = PurePosixPath(args[2]).parent
    if (worker_temp.parent != PurePosixPath("/tmp")
            or not worker_temp.name.startswith(TASK_PREFIXES[slug])
            or PurePosixPath(args[2]).name != "solution.py"):
        return None
    if cwd is not None and PurePosixPath(cwd) != worker_temp:
        return None
    return slug, str(worker_temp)


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match only the official task runners' exact worker invocation."""
    if not isinstance(argv, (list, tuple)):
        return False
    if kwargs.get("preexec_fn") is None:
        return False
    return _worker_paths(task_id, argv, kwargs.get("cwd")) is not None


def _absolute_source(source_root: Path, target: str) -> Path:
    path = Path(target)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("localization worker paths must be absolute and normalized")
    resolved = source_root / path.as_posix().lstrip("/")
    if resolved.is_symlink() or not resolved.exists():
        raise FileNotFoundError("localization worker source path is unavailable")
    return resolved


def _python_runtime_roots(executable: str, source_root: Path) -> list[tuple[Path, str]]:
    """Return the current temporary Python environment and its uv packages."""
    if not executable.startswith("/") or ".." in Path(executable).parts:
        raise ValueError("localization worker Python path is not absolute")
    lexical = Path(executable)
    if any(lexical.is_relative_to(Path(prefix)) for prefix in ("/usr/bin", "/usr/local/bin")):
        return []
    prefix = Path(sys.prefix).resolve()
    if (not prefix.is_relative_to(Path("/tmp")) or prefix == Path("/tmp")
            or not prefix.is_dir()):
        raise ValueError("localization worker Python must be system Python or a uv venv under /tmp")
    # uv's temporary interpreter may keep package payloads in separate
    # /tmp/uv-cache/archive-v0 directories. Preserve only those exact
    # site-package paths present in the verifier's active sys.path.
    roots = [(_absolute_source(source_root, str(prefix)), str(prefix))]
    targets = {str(prefix)}
    cache_root = UV_CACHE_ROOT
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            continue
        candidate = candidate.resolve()
        if not candidate.is_relative_to(cache_root) or not candidate.is_dir():
            continue
        if any(candidate == Path(target) or candidate.is_relative_to(Path(target))
               for target in targets):
            continue
        roots.append((_absolute_source(source_root, str(candidate)), str(candidate)))
        targets.add(str(candidate))
    return roots


def _clean_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in environment.items():
        if key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED")):
            continue
        if key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}:
            continue
        if key in {"PATH", "PYTHONPATH"}:
            value = ":".join(
                entry for entry in value.split(":")
                if entry and not entry.startswith(("/opt/relay-isolation", "/opt/relay-tools"))
            )
            if not value:
                continue
        result[key] = value
    return result


def build_worker_command(
    *, task_id: str, argv: Sequence[str], environment: Mapping[str, str],
    cwd: str | None, uid: int, gid: int,
    bubblewrap_new_session: bool = True,
    bwrap: str = BWRAP_PATH, source_root: Path = Path("/"),
    system_root: Path = Path("/"),
) -> WorkerLaunch:
    """Wrap the existing trusted task worker without changing its JSON calls."""
    paths = _worker_paths(task_id, argv, cwd)
    if paths is None:
        raise ValueError("process is not an audited localization solver worker")
    _slug, tempdir = paths
    args = [os.fspath(value) for value in argv]
    source_root = Path(source_root)
    system_root = Path(system_root)
    worker_source = _absolute_source(source_root, WORKER_PATH)
    candidate_source = _absolute_source(source_root, args[2])
    temp_source = _absolute_source(source_root, tempdir)
    if not worker_source.is_file() or not candidate_source.is_file() or not temp_source.is_dir():
        raise FileNotFoundError("localization worker or candidate source is not a regular file")
    try:
        candidate_source.relative_to(temp_source)
    except ValueError as error:
        raise ValueError("candidate source is outside its private worker directory") from error
    runtime_roots = _python_runtime_roots(args[0], source_root)
    worker_fd = os.open(worker_source, os.O_RDONLY)
    try:
        dispatcher_fd, dispatcher_file = _dispatcher_fd()
    except BaseException:
        os.close(worker_fd)
        raise

    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
        "--cap-drop", "ALL", "--die-with-parent",
    ]
    if bubblewrap_new_session:
        command.append("--new-session")
    for path in SYSTEM_PATHS:
        host_path = system_root / path.lstrip("/")
        if host_path.is_symlink():
            command += ["--symlink", os.readlink(host_path), path]
        elif host_path.exists():
            command += ["--ro-bind", str(host_path), path]

    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for source, target in runtime_roots:
        target_path = PurePosixPath(target)
        ancestors: list[PurePosixPath] = []
        cursor = target_path.parent
        while cursor != PurePosixPath("/"):
            if cursor != PurePosixPath("/tmp"):
                ancestors.append(cursor)
            cursor = cursor.parent
        for directory in reversed(ancestors):
            command += ["--dir", str(directory)]
        command += ["--dir", target]
        command += ["--ro-bind", str(source), target]

    command += [
        "--dir", "/app", "--dir", "/tests", "--dir", "/logs", "--dir", "/results",
        "--dir", "/home", "--dir", "/var/tmp", "--dir", "/run", "--dir", "/run/lock",
    ]
    command += ["--dir", tempdir, "--bind", str(temp_source), tempdir]
    command += ["--ro-bind-data", str(worker_fd), WORKER_PATH]
    command += ["--ro-bind-data", str(dispatcher_fd), EXEC_PATH]
    command += ["--remount-ro", "/", "--remount-ro", "/dev", "--chdir", cwd or tempdir]
    command += ["--", args[0], EXEC_PATH, WORKER_PATH, args[2]]
    return WorkerLaunch(command, _clean_environment(environment), (worker_fd, dispatcher_fd), dispatcher_file)
