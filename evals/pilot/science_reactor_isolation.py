"""Run the reactor controller host in a minimal, unprivileged child namespace."""
from __future__ import annotations

import os
import pwd
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_SLUG = "reactor-safety-control"
TASK_ID = "terminal-bench-science/reactor-safety-control"
CONTROLLER_HOST = "/usr/local/lib/reactor_controller_host.py"
SUBMISSION_DIR = "/app/submission"
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
_SCRATCH = re.compile(r"/tmp/reactor-controller-[A-Za-z0-9_-]+\Z")
_BOOTSTRAP = (
    "import os,sys; scratch=sys.argv[1]; argv=sys.argv[2:]; "
    "os.mkdir(scratch,0o700); os.chmod(scratch,0o700); os.chdir(scratch); "
    "os.execv(argv[0],argv)"
)


class ReactorIsolationError(RuntimeError):
    """The Reactor child command or its filesystem boundary is invalid."""


@dataclass(frozen=True)
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]


def task_slug(task_id: str) -> str:
    """Accept the bare slug used by create_bundle or its canonical task ID."""
    if task_id == TASK_SLUG:
        return TASK_SLUG
    if task_id == TASK_ID:
        return TASK_SLUG
    raise ReactorIsolationError(f"No Reactor candidate adapter for task {task_id!r}")


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match only the verifier's exact controller_host Popen call."""
    try:
        task_slug(task_id)
    except ReactorIsolationError:
        return False
    return (isinstance(argv, (list, tuple)) and len(argv) == 3
            and os.fspath(argv[0]) == sys.executable
            and os.fspath(argv[1]) == CONTROLLER_HOST
            and os.fspath(argv[2]) == SUBMISSION_DIR)


def official_unprivileged_ids() -> tuple[int, int]:
    """Mirror the verifier's nobody-then-daemon demotion when it runs as root."""
    if os.getuid() != 0:
        return os.getuid(), os.getgid()
    for name in ("nobody", "daemon"):
        try:
            account = pwd.getpwnam(name)
            return account.pw_uid, account.pw_gid
        except KeyError:
            continue
    return os.getuid(), os.getgid()


def _child_environment(environment: Mapping[str, str], scratch: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in environment.items():
        if key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED")):
            continue
        if key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}:
            continue
        if key in {"PATH", "PYTHONPATH"}:
            value = ":".join(part for part in value.split(":")
                             if part and not part.startswith(("/opt/relay-isolation", "/opt/relay-tools")))
            if not value:
                continue
        result[key] = value
    # The outer verifier creates a fresh private scratch directory for each
    # controller process. The child creates the same logical path in a fresh
    # tmpfs, so neither it nor the verifier shares arbitrary host /tmp files.
    result.update({"TMPDIR": scratch, "TMP": scratch, "TEMP": scratch})
    return result


def prepare_candidate(
    task_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str | os.PathLike[str] | None,
    uid: int | None = None,
    gid: int | None = None,
    bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"),
    system_paths: Sequence[str] = SYSTEM_PATHS,
) -> CandidateLaunch:
    """Build a boundary for the official line-oriented controller subprocess.

    The controller host keeps its original stdin/stdout protocol and executes
    the same controller file. The child sees public /app inputs and its
    submitted directory, but not verifier fixtures, reward logs, or network.
    """
    task_slug(task_id)
    args = [os.fspath(value) for value in argv]
    expected = [sys.executable, CONTROLLER_HOST, SUBMISSION_DIR]
    if args != expected:
        raise ReactorIsolationError("Unexpected Reactor controller host argv")
    if cwd is None:
        raise ReactorIsolationError("Reactor controller launch is missing its scratch cwd")
    scratch = os.fspath(cwd)
    if not _SCRATCH.fullmatch(scratch):
        raise ReactorIsolationError("Reactor controller cwd is outside its private scratch slot")
    scratch_path = source_root / scratch.lstrip("/")
    if not scratch_path.is_dir() or scratch_path.is_symlink():
        raise ReactorIsolationError("Reactor controller scratch directory is missing or invalid")
    submission = source_root / SUBMISSION_DIR.lstrip("/")
    if not submission.is_dir() or submission.is_symlink():
        raise ReactorIsolationError("Reactor submitted controller directory is missing or invalid")
    if uid is None or gid is None:
        default_uid, default_gid = official_unprivileged_ids()
        uid = default_uid if uid is None else uid
        gid = default_gid if gid is None else gid

    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
        "--cap-drop", "ALL", "--die-with-parent", "--new-session",
    ]
    for path in system_paths:
        source = source_root / path.lstrip("/")
        if source.is_symlink():
            command += ["--symlink", os.readlink(source), path]
        elif source.exists():
            command += ["--ro-bind", str(source), path]
    command += ["--proc", "/proc", "--dev", "/dev"]
    # These paths contain verifier-only material in the outer test image.
    command += ["--tmpfs", "/tests", "--tmpfs", "/logs", "--tmpfs", "/tmp",
                "--tmpfs", "/opt", "--chmod", "1777", "/tmp"]
    # /app contains only public task inputs plus the submitted controller.
    # Mount that directory read-only, then make the root filesystem immutable.
    command += ["--ro-bind", str(submission), SUBMISSION_DIR,
                "--remount-ro", "/", "--chdir", "/tmp", "--"]
    command += [sys.executable, "-c", _BOOTSTRAP, scratch, *args]
    return CandidateLaunch(command, _child_environment(environment, scratch))
