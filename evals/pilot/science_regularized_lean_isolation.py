"""Isolate the official Lean compiler calls for regularized-game-proof.

The task's test.sh copies the submitted Basic.lean into the verifier project,
then invokes Lake for the official build and two trusted checks. This adapter
preserves those exact invocations and the task's official source scanner. It
adds no Lean syntax policy; the OS boundary removes host paths and networking.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


_ORIGINAL_POPEN = subprocess.Popen
TASK = "regularized-game-proof"
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
WORKSPACE = "/workspace"


def _slug(task_id: str) -> str:
    slug = task_id.rsplit("/", 1)[-1]
    if slug != TASK or (task_id != slug and task_id != f"terminal-bench-science/{slug}"):
        raise ValueError(f"No reviewed Lean compiler interface for {task_id!r}")
    return slug


def _command_kind(argv: Any) -> str | None:
    if not isinstance(argv, (list, tuple)) or not argv or Path(os.fspath(argv[0])).name != "lake":
        return None
    args = tuple(os.fspath(item) for item in argv[1:])
    if args == ("build", "--wfail"):
        return "build"
    if args in {("env", "lean", "HiddenCheck.lean"), ("env", "lean", "AxiomCheck.lean")}:
        return args[-1]
    return None


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if task_id.rsplit("/", 1)[-1] != TASK:
        return False
    _slug(task_id)
    return kwargs.get("cwd") == "/verifier/GameProof" and _command_kind(argv) is not None


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if ".lake" in item.relative_to(path).parts:
            continue
        if item.is_file() and not item.is_symlink():
            h.update(item.relative_to(path).as_posix().encode() + b"\0")
            h.update(item.read_bytes() + b"\0")
    return h.hexdigest()


def _copy_project(workspace: Path, stage: Path, *, seed_build: bool) -> None:
    for child in list(stage.iterdir()):
        if child.name != ".lake":
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    for root, dirs, files in os.walk(workspace, followlinks=False):
        rel_root = Path(root).relative_to(workspace)
        dirs[:] = [d for d in dirs if d not in {".lake", ".git", "__pycache__"}
                   and not (Path(root) / d).is_symlink()]
        target_root = stage / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(root) / name
            if src.is_symlink() or name.endswith((".olean", ".ilean", ".trace")):
                continue
            # Hidden check sources enter the isolated project only for their
            # respective official check invocation, never during `lake build`.
            if name in {"HiddenCheck.lean", "AxiomCheck.lean"}:
                continue
            shutil.copy2(src, target_root / name)
    (stage / ".lake/packages").mkdir(parents=True, exist_ok=True)
    (stage / ".lake/build").mkdir(parents=True, exist_ok=True)
    if seed_build:
        source_build = workspace / ".lake/build"
        if source_build.is_dir():
            shutil.copytree(source_build, stage / ".lake/build", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("Basic.olean", "Basic.ilean", "Basic.c",
                                                          "Basic.setup.json"))


def prepare_candidate(
    task_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str | os.PathLike[str],
    bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"),
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    slug = _slug(task_id)
    kind = _command_kind(argv)
    if not is_candidate(task_id, argv, {"cwd": os.fspath(cwd)}):
        raise ValueError("Lean launch does not match the audited task/cwd profile")
    logical_cwd = Path(cwd)
    workspace = (source_root / logical_cwd.relative_to("/")).resolve()
    source_base = source_root.resolve()
    if not workspace.is_relative_to(source_base) or not workspace.is_dir():
        raise FileNotFoundError("Regularized verifier project is missing or escapes its source root")
    basic = workspace / "GameProof/Basic.lean"
    if not basic.is_file():
        raise FileNotFoundError("Officially staged candidate Basic.lean is absent")
    digest = _digest(workspace)
    temp_parent = source_root / "tmp"
    if not temp_parent.is_dir():
        raise FileNotFoundError("Lean verifier has no private temporary directory")
    # The wrapper process is restarted for each official Lake invocation.
    # Keying a private project snapshot by its stable parent shell PID keeps
    # the Lake build outputs available to HiddenCheck and AxiomCheck.
    stage_parent = temp_parent / f"relay-lean-{slug}-{os.getppid()}"
    stage = stage_parent / "project"
    marker = stage_parent / "relay-stage.json"
    new_stage = not stage.exists()
    if new_stage:
        stage_parent.mkdir(mode=0o700, exist_ok=False)
        stage.mkdir(mode=0o700)
        marker.write_text(json.dumps({"task": slug, "source_digest": digest}) + "\n")
        marker.chmod(0o600)
    else:
        if not marker.is_file():
            raise RuntimeError("Existing private Lean stage has no trusted marker")
        state = json.loads(marker.read_text())
        if state != {"task": slug, "source_digest": digest}:
            raise RuntimeError("Candidate Lean project changed during verifier execution")
    _copy_project(workspace, stage, seed_build=new_stage)
    packages = workspace / ".lake/packages"
    if not packages.exists():
        packages = packages.resolve()
    if not packages.is_dir():
        raise FileNotFoundError("Pinned Lean package cache is absent; network fallback is forbidden")
    if kind == "build":
        for suffix in (".olean", ".ilean"):
            (stage / ".lake/build/lib/lean/GameProof/Basic" ).with_suffix(suffix).unlink(missing_ok=True)
        for rel in (".lake/build/ir/GameProof/Basic.c", ".lake/build/ir/GameProof/Basic.setup.json"):
            (stage / rel).unlink(missing_ok=True)
    else:
        checker = workspace / str(kind)
        if not checker.is_file():
            raise FileNotFoundError(f"Official Lean check source is absent: {kind}")
        shutil.copy2(checker, stage / str(kind))

    args = [os.fspath(arg) for arg in argv]
    old_root = os.fspath(logical_cwd)
    rewritten = [WORKSPACE + arg[len(old_root):] if arg == old_root or arg.startswith(old_root + "/") else arg
                 for arg in args]
    command = [bwrap, "--die-with-parent", "--new-session", "--unshare-user", "--uid", "0", "--gid", "0",
               "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--cap-drop", "ALL"]
    for path in SYSTEM_PATHS:
        host = Path(path)
        if host.is_symlink():
            command += ["--symlink", os.readlink(host), path]
        elif host.exists():
            command += ["--ro-bind", str(host), path]
    command += ["--dir", "/root", "--ro-bind", str(source_root / "root/.elan"), "/root/.elan",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", WORKSPACE, "--bind", str(stage), WORKSPACE,
                "--ro-bind", str(packages), WORKSPACE + "/.lake/packages",
                "--remount-ro", "/", "--remount-ro", "/dev", "--chdir", WORKSPACE,
                "--clearenv", "--setenv", "HOME", "/root", "--setenv", "ELAN_HOME", "/root/.elan",
                "--setenv", "PATH", "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin"]
    child_env = {"HOME": "/root", "ELAN_HOME": "/root/.elan",
                 "PATH": "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin"}
    for env_key in ("LANG", "LC_ALL", "TZ"):
        if env_key in environment:
            child_env[env_key] = environment[env_key]
    command += ["--", *rewritten]
    metadata = {"task_id": task_id, "kind": kind, "source_digest": digest,
                "workspace": str(workspace), "stage": str(stage),
                "network_unshared": True, "tests_path_absent": True}
    return command, child_env, metadata


def lake_wrapper_main(argv: Sequence[str]) -> int:
    """Replace the official bare ``lake`` command with its isolated launch."""
    task_id = os.environ.get("RELAY_SCIENCE_TASK_ID", "")
    command, child_env, _ = prepare_candidate(
        task_id, ["lake", *argv], environment=os.environ, cwd=os.getcwd(),
    )
    os.execve(command[0], command, child_env)
    return 127


