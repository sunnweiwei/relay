"""Filesystem and process boundary for genomic-model-ranking's predictor."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TASK_SLUG = "genomic-model-ranking"
TASK_PREFIX = "genomic-ranking-run-"
CANONICAL_TASK = "terminal-bench-science/genomic-model-ranking"
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)


class CandidateLaunch:
    """Prepared command with trusted in-namespace resource-limit shim."""

    def __init__(self, command: list[str], environment: dict[str, str],
                 dispatcher: Any):
        self.command = command
        self.environment = environment
        self._dispatcher = dispatcher
        self.pass_fds = (dispatcher.fileno(),)

    def close(self) -> None:
        self._dispatcher.close()


def task_matches(task_id: str) -> bool:
    return task_id in {TASK_SLUG, CANONICAL_TASK}


def _paths(task_id: str, argv: Sequence[str], cwd: str | None) -> str | None:
    if not task_matches(task_id):
        return None
    args = [os.fspath(value) for value in argv]
    if len(args) != 9 or args[0] != "python3" or args[1] != "-I":
        return None
    if not args[2].endswith("/predict.py"):
        return None
    root = PurePosixPath(args[2]).parent
    if root.parent != PurePosixPath("/tmp") or not root.name.startswith(TASK_PREFIX):
        return None
    expected = ["--input-dir", str(root / "input"), "--output",
                str(root / "output" / "predictions.csv"), "--ranking-output",
                str(root / "output" / "model_ranking.json")]
    if args[3:] != expected:
        return None
    if cwd is not None and PurePosixPath(cwd) != root:
        return None
    return str(root)


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match only the official verifier's predictor subprocess contract."""
    if not isinstance(argv, (list, tuple)) or kwargs.get("preexec_fn") is None:
        return False
    if kwargs.get("stdin") is None or kwargs.get("close_fds") is not True:
        return False
    return _paths(task_id, argv, kwargs.get("cwd")) is not None


def _source_path(root: Path, target: str) -> Path:
    path = Path(target)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("candidate paths must be absolute and normalized")
    resolved = root / path.as_posix().lstrip("/")
    if resolved.is_symlink() or not resolved.exists():
        raise FileNotFoundError("candidate source path is unavailable")
    return resolved


def _dispatcher() -> Any:
    source = (
        "import os, resource, sys\n"
        "if (len(sys.argv) != 9 or sys.argv[1] != '-I' or "
        "    not sys.argv[2].startswith('/tmp/genomic-ranking-run-') or "
        "    not sys.argv[2].endswith('/predict.py') or "
        "    sys.argv[3:] != ['--input-dir', sys.argv[2][:-10] + 'input', "
        "'--output', sys.argv[2][:-10] + 'output/predictions.csv', "
        "'--ranking-output', sys.argv[2][:-10] + 'output/model_ranking.json']):\n"
        "    raise SystemExit(125)\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (40, 40))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (3200000000, 3200000000))\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (25000000, 25000000))\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
        "if hasattr(resource, 'RLIMIT_NPROC'):\n"
        "    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    ).encode()
    handle = tempfile.TemporaryFile()
    handle.write(source)
    handle.flush()
    os.lseek(handle.fileno(), 0, os.SEEK_SET)
    return handle


def build_candidate_command(
    *, task_id: str, argv: Sequence[str], environment: Mapping[str, str],
    cwd: str | None, uid: int, gid: int, bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"), system_root: Path = Path("/"),
) -> CandidateLaunch:
    root_text = _paths(task_id, argv, cwd)
    if root_text is None:
        raise ValueError("process is not the audited genomic predictor")
    args = [os.fspath(value) for value in argv]
    root = Path(root_text)
    source_root, system_root = Path(source_root), Path(system_root)
    root_source = _source_path(source_root, root_text)
    program_source = _source_path(source_root, args[2])
    if not root_source.is_dir() or not program_source.is_file():
        raise FileNotFoundError("genomic run directory or predictor is unavailable")
    for leaf in ("input/examples.csv", "input/candidate_scores.csv",
                 "input/baseline_evaluation.csv"):
        source = _source_path(source_root, str(root / leaf))
        if not source.is_file():
            raise FileNotFoundError("required public predictor input is unavailable")

    dispatcher = _dispatcher()
    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
        "--cap-drop", "ALL", "--die-with-parent", "--new-session",
    ]
    for path in SYSTEM_PATHS:
        host_path = system_root / path.lstrip("/")
        if host_path.is_symlink():
            command += ["--symlink", os.readlink(host_path), path]
        elif host_path.exists():
            command += ["--ro-bind", str(host_path), path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    command += ["--dir", str(root), "--bind", str(root_source), str(root)]
    command += ["--ro-bind-data", str(dispatcher.fileno()), "/tmp/relay-genomic-exec.py"]
    command += ["--remount-ro", "/", "--remount-ro", "/dev", "--chdir", str(root)]
    # The trusted shim applies the verifier's limits after bwrap has created
    # its namespaces, then execs the exact original Python argv.
    command += ["--", args[0], "-I", "/tmp/relay-genomic-exec.py", *args[1:]]
    return CandidateLaunch(command, dict(environment), dispatcher)
