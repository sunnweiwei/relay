"""Contain the AMR solver launches made by its trusted pytest verifier.

The verifier remains trusted and keeps its oracle, reference checks, and score
logic. This module recognizes only the UID-dropped
``/app/solver/amr_poisson3d.py`` invocations and exposes the solver directory,
optional problem packet, and fresh output files to a private child namespace.
"""

from __future__ import annotations

import os
import grp
import pwd
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "terminal-bench-science/amr-poisson-optimize"
TASK_SLUG = "amr-poisson-optimize"
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
_SECRET_KEYS = {
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
    "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
}


@dataclass
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    scratch: Path
    outputs: list[tuple[Path, Path]]

    def export_outputs(self, returncode: int) -> None:
        """Copy only verifier-requested output files back after success."""
        try:
            if returncode == 0:
                for staged, destination in self.outputs:
                    if not staged.is_file() or staged.is_symlink():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".relay-tmp")
                    temporary.write_bytes(staged.read_bytes())
                    os.replace(temporary, destination)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)


def _slug(task_id: str) -> str:
    slug = task_id.rsplit("/", 1)[-1]
    if slug != TASK_SLUG or task_id not in {TASK_ID, TASK_SLUG}:
        raise ValueError(f"No reviewed AMR solver interface for {task_id!r}")
    return slug


def _is_absolute_child(path: str, root: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        return False
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate != root


def _parse_outputs(argv: Sequence[str], app_root: Path) -> tuple[list[Path], Path | None] | None:
    args = [os.fspath(value) for value in argv]
    if len(args) < 2 or Path(args[1]) != app_root / "solver/amr_poisson3d.py":
        return None
    outputs: list[Path] = []
    problem: Path | None = None
    allowed = {"--output", "--solution", "--problems", "--config"}
    i = 2
    seen: set[str] = set()
    while i < len(args):
        flag = args[i]
        if flag not in allowed or flag in seen or i + 1 >= len(args):
            return None
        value = args[i + 1]
        seen.add(flag)
        i += 2
        if flag in {"--output", "--solution"}:
            path = Path(value)
            if not _is_absolute_child(value, app_root):
                return None
            outputs.append(path)
        elif flag == "--problems":
            if not _is_absolute_child(value, app_root):
                return None
            problem = Path(value)
        elif flag == "--config":
            if not value or value.startswith("/") or ".." in Path(value).parts:
                return None
    if not {"--output", "--solution"} <= seen or len(outputs) != 2:
        return None
    return outputs, problem


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match only official candidate solver runs with an explicit UID drop."""
    if task_id not in {TASK_ID, TASK_SLUG} or not isinstance(argv, (list, tuple)):
        return False
    if kwargs.get("user") is None:
        return False  # The reference oracle is deliberately run as trusted root.
    cwd = kwargs.get("cwd")
    if not isinstance(cwd, (str, os.PathLike)):
        return False
    app_root = Path(cwd)
    if not app_root.is_absolute() or ".." in app_root.parts:
        return False
    return _parse_outputs(argv, app_root) is not None


def is_privacy_probe(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match the verifier's exact trusted /tests permission probe."""
    if task_id not in {TASK_ID, TASK_SLUG} or not isinstance(argv, (list, tuple)):
        return False
    args = [os.fspath(value) for value in argv]
    probe = ("import sys\np = sys.argv[1]\ntry:\n    open(p, 'rb').read(1)\n"
             "except PermissionError:\n    print('DENIED'); sys.exit(0)\n"
             "except FileNotFoundError:\n    print('ABSENT'); sys.exit(3)\n"
             "print('READABLE'); sys.exit(1)\n")
    return (len(args) == 4 and Path(args[0]).name.startswith("python") and args[1] == "-c"
            and args[2] == probe
            and args[3] in {"/tests/oracle_solve3d.py", "/tests/beta_field3d.py"}
            and kwargs.get("user") == "grader" and kwargs.get("stdout") is not None
            and kwargs.get("stderr") is not None)


def prepare_privacy_probe(argv: Sequence[str], *, environment: Mapping[str, str],
                          uid: int, gid: int, bwrap: str = BWRAP_PATH,
                          source_root: Path = Path("/")) -> tuple[list[str], dict[str, str]]:
    """Run only the trusted verifier probe with real UID permissions on /tests."""
    args = [os.fspath(value) for value in argv]
    command = [bwrap, "--die-with-parent", "--unshare-user", "--uid", str(uid),
               "--gid", str(gid), "--unshare-pid", "--unshare-ipc", "--unshare-uts",
               "--unshare-net", "--cap-drop", "ALL", "--new-session"]
    for path in SYSTEM_PATHS:
        host = source_root / path.lstrip("/")
        if host.is_symlink():
            command += ["--symlink", os.readlink(host), path]
        elif host.exists():
            command += ["--ro-bind", str(host), path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", "/tests", "--chmod", "000", "/tests",
                "--remount-ro", "/", "--remount-ro", "/dev", "--chdir", "/", "--", *args]
    clean = {key: value for key, value in environment.items()
             if not key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED"))
             and key not in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}}
    return command, clean


def _copy_solver(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"AMR candidate solver directory is absent: {source}")
    for entry in source.rglob("*"):
        if entry.is_symlink():
            raise ValueError("Symlinked AMR solver files are unsupported")
        if not entry.is_file() and not entry.is_dir():
            raise ValueError("Non-regular AMR solver entries are unsupported")
    shutil.copytree(source, destination, symlinks=False)
    for entry in sorted(destination.rglob("*"), reverse=True):
        entry.chmod(0o555 if entry.is_dir() else 0o444)
    destination.chmod(0o555)


def _clean_environment(environment: Mapping[str, str], guard_mount: str | None) -> dict[str, str]:
    env = {
        key: value for key, value in environment.items()
        if key not in _SECRET_KEYS
        and not key.startswith(("OPENAI_", "ANTHROPIC_", "HF_", "HUGGINGFACE_", "RELAY_",
                               "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED"))
    }
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    env["HOME"] = "/app"
    env["TMPDIR"] = "/tmp"
    env["TEMP"] = "/tmp"
    env["TMP"] = "/tmp"
    if guard_mount:
        env["PYTHONPATH"] = guard_mount
    else:
        env.pop("PYTHONPATH", None)
    return env


def prepare_candidate(
    task_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str,
    uid: int | str,
    gid: int | str,
    source_root: Path = Path("/"),
    bwrap: str = BWRAP_PATH,
) -> CandidateLaunch:
    """Build a private `/app` containing only solver inputs and requested outputs."""
    _slug(task_id)
    app_root = Path(cwd)
    parsed = _parse_outputs(argv, app_root)
    if parsed is None:
        raise ValueError("Unexpected AMR candidate solver argv")
    output_paths, problem_path = parsed
    args = [os.fspath(value) for value in argv]
    uid_value = pwd.getpwnam(str(uid)).pw_uid if not str(uid).isdecimal() else int(uid)
    gid_value = grp.getgrnam(str(gid)).gr_gid if not str(gid).isdecimal() else int(gid)
    solver_source = source_root / app_root.relative_to("/") / "solver"

    scratch = Path(tempfile.mkdtemp(prefix="relay-amr-solver-"))
    app_stage = scratch / "app"
    app_stage.mkdir(mode=0o755)
    outputs: list[tuple[Path, Path]] = []
    try:
        solver_stage = app_stage / "solver"
        _copy_solver(solver_source, solver_stage)
        for output in output_paths:
            rel = output.relative_to(app_root)
            staged = app_stage / rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.touch(mode=0o666)
            staged.chmod(0o666)
            outputs.append((staged, source_root / output.relative_to("/")))
        if problem_path is not None:
            problem_source = source_root / problem_path.relative_to("/")
            if not problem_source.is_file() or problem_source.is_symlink():
                raise FileNotFoundError("Verifier-created AMR problem packet is absent")
            staged_problem = app_stage / problem_path.relative_to(app_root)
            staged_problem.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(problem_source, staged_problem)
            staged_problem.chmod(0o444)

        guard_mount: str | None = None
        guard_source: Path | None = None
        guard_value = environment.get("PYTHONPATH")
        if guard_value:
            guard_parts = guard_value.split(os.pathsep)
            if len(guard_parts) != 1 or not Path(guard_parts[0]).is_absolute():
                raise ValueError("AMR verifier import guard must be one absolute directory")
            guard_source = source_root / guard_parts[0].lstrip("/")
            guard_file = guard_source / "sitecustomize.py"
            if not guard_file.is_file() or guard_file.is_symlink():
                raise FileNotFoundError("AMR verifier import guard is absent")
            guard_mount = "/opt/amr-import-guard"

        command = [
            bwrap, "--die-with-parent", "--unshare-user", "--uid", str(uid_value),
            "--gid", str(gid_value), "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--unshare-net", "--cap-drop", "ALL", "--new-session",
        ]
        for path in SYSTEM_PATHS:
            host = source_root / path.lstrip("/")
            if host.is_symlink():
                command += ["--symlink", os.readlink(host), path]
            elif host.exists():
                command += ["--ro-bind", str(host), path]
        command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"]
        for path in ("/app", "/tests", "/logs", "/results", "/home", "/root", "/var",
                     "/usr/project", "/run/secrets"):
            command += ["--dir", path]
        command += ["--bind", str(app_stage), "/app", "--ro-bind", str(solver_stage), "/app/solver"]
        if problem_path is not None:
            command += ["--ro-bind", str(staged_problem), str(problem_path)]
        if guard_source is not None and guard_mount is not None:
            command += ["--dir", "/opt", "--ro-bind", str(guard_source), guard_mount]
        command += ["--remount-ro", "/", "--remount-ro", "/dev"]
        command += ["--chdir", "/app", "--", *args]
        clean_env = _clean_environment(environment, guard_mount)
        return CandidateLaunch(command, clean_env, scratch, outputs)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
