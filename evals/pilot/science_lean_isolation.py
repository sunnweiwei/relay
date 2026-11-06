"""Constrain Lean compiler subprocesses used by the three reviewed Science tasks.

The verifier remains the scorer. This adapter gives each Lean invocation a
private project snapshot, a read-only pinned package cache, the task's Lean
toolchain, and no verifier paths, host paths, or network. It intentionally
accepts only the audited task roots and ``lake`` commands.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_ORIGINAL_POPEN = subprocess.Popen


TASKS = frozenset({"finite-free-stam", "onsager-ising-lean", "gen-turan-paths"})
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
WORKSPACE = "/workspace"

def _slug(task_id: str) -> str:
    slug = task_id.rsplit("/", 1)[-1]
    if slug not in TASKS or (task_id != slug and not task_id.endswith("/" + slug)):
        raise ValueError(f"No reviewed Lean compiler interface for {task_id!r}")
    return slug


def _is_lake(argv: Any) -> bool:
    if not isinstance(argv, (list, tuple)) or not argv:
        return False
    first = os.fspath(argv[0])
    return Path(first).name == "lake"


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if task_id.rsplit("/", 1)[-1] not in TASKS:
        return False
    slug = _slug(task_id)
    if not _is_lake(argv):
        return False
    cwd = kwargs.get("cwd")
    if cwd is None:
        return False
    path = Path(os.fspath(cwd))
    if not path.is_absolute() or ".." in path.parts:
        return False
    if slug == "finite-free-stam":
        return path == Path("/app")
    if slug == "gen-turan-paths":
        return path == Path("/root/project")
    return len(path.parts) == 3 and path.parts[:2] == ("/", "tmp") and path.name.startswith("onsager_golden_")


def _editable_files(slug: str, workspace: Path) -> list[Path]:
    if slug == "finite-free-stam":
        root = workspace / "FiniteFree"
        return sorted(root.rglob("*.lean")) if root.is_dir() else []
    if slug == "onsager-ising-lean":
        root = workspace / "GSLean/Onsager"
        return sorted(root.rglob("*.lean")) if root.is_dir() else []
    target = workspace / "GenTuranProj/Submission.lean"
    return [target] if target.is_file() else []


def _source_digest(files: Sequence[Path], workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(workspace).as_posix()
        digest.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _official_source_checks(slug: str) -> None:
    """Run the task's own pre-execution source checks, without adding policy."""
    basename, functions = {
        "finite-free-stam": ("test_outputs.py", ("test_phase6_no_metaprogramming",
                                                   "test_phase7_no_axiom_declarations",
                                                   "test_phase8_no_kernel_bypass")),
        "onsager-ising-lean": ("test_state.py", ("test_no_banned_exploit_syntax",
                                                   "test_no_sorry_source_grep")),
        "gen-turan-paths": ("test_verify.py", ("test_no_banned_exploit_syntax",)),
    }[slug]
    loaded = [candidate for candidate in tuple(sys.modules.values())
              if isinstance(getattr(candidate, "__file__", None), str)
              and Path(candidate.__file__).name == basename]
    # The verifier imports the task's own tests from /tests; prefer that file over any other
    # module that happens to share its basename.
    module = next((candidate for candidate in loaded if Path(candidate.__file__) == Path("/tests") / basename),
                  loaded[0] if loaded else None)
    if module is None:
        raise RuntimeError(f"Official source checker {basename} is not loaded; refusing Lean execution")
    for name in functions:
        check = getattr(module, name, None)
        if check is None:
            raise RuntimeError(f"Official source checker {basename}.{name} is unavailable")
        check()


def _check_candidate_source(slug: str, workspace: Path, *, enforce_official_checks: bool) -> str:
    files = _editable_files(slug, workspace)
    if not files:
        raise FileNotFoundError(f"No candidate Lean source for {slug}")
    if enforce_official_checks:
        _official_source_checks(slug)
    return _source_digest(files, workspace)


def _copy_project(workspace: Path, stage: Path) -> None:
    """Refresh sources while retaining private writable Lake caches and dependencies."""
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
            dst = target_root / name
            shutil.copy2(src, dst)
    package_source = (workspace / ".lake" / "packages").resolve()
    package_stage = stage / ".lake" / "packages"
    if not package_source.is_dir():
        raise FileNotFoundError("Pinned Lean package cache is absent; network fallback is forbidden")
    if not package_stage.is_dir() or not any(package_stage.iterdir()):
        if package_stage.exists():
            shutil.rmtree(package_stage)
        package_stage.mkdir(parents=True)
        # Lake writes package-local lock and compilation hashes during an
        # official build. Keep those writes inside this attempt's private
        # child workspace, never in the verifier image or pinned package tree.
        subprocess.run(
            ["cp", "-a", "--reflink=auto", str(package_source) + "/.", str(package_stage)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            timeout=600,
        )
        for root, directories, files in os.walk(package_stage, followlinks=False):
            path = Path(root)
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
            for name in files:
                item = path / name
                if not item.is_symlink():
                    item.chmod(stat.S_IMODE(item.stat().st_mode) | stat.S_IWUSR)


@dataclass
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    workspace: Path
    stage: Path
    source_digest: str
    canary_result: dict[str, Any]


_STAGES: dict[tuple[str, str], tuple[Path, Path, str, Path, dict[str, Any] | None]] = {}


def _cleanup() -> None:
    for _, stage, _, _, _ in list(_STAGES.values()):
        shutil.rmtree(stage.parent, ignore_errors=True)
    _STAGES.clear()


atexit.register(_cleanup)


def prepare_candidate(
    task_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str | os.PathLike[str],
    bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"),
) -> CandidateLaunch:
    slug = _slug(task_id)
    if not is_candidate(task_id, argv, {"cwd": cwd}):
        raise ValueError("Lean launch does not match its audited task/cwd profile")
    logical_cwd = Path(cwd)
    workspace = (source_root / logical_cwd.relative_to("/")).resolve()
    source_base = source_root.resolve()
    if not workspace.is_relative_to(source_base) or not workspace.is_dir():
        raise ValueError("Lean project workspace is missing or escapes its source root")
    digest = _check_candidate_source(slug, workspace, enforce_official_checks=(source_root.resolve() == Path("/")))
    key = (slug, str(workspace))
    entry = _STAGES.get(key)
    if entry is None:
        # Apptainer gives /tmp and its writable root a small memory-backed
        # quota. Onsager's pinned Lake packages are large, so keep private
        # staging under the verifier's writable /logs mount. The candidate
        # child never mounts /logs. Unit fixtures may use results or tmp.
        logs_parent = source_root / "logs"
        results_parent = source_root / "results"
        if logs_parent.is_dir() and os.access(logs_parent, os.W_OK):
            temp_parent = logs_parent
        elif results_parent.is_dir() and os.access(results_parent, os.W_OK):
            temp_parent = results_parent
        else:
            temp_parent = source_root / "tmp"
        if not temp_parent.is_dir():
            raise FileNotFoundError("Lean verifier has no writable private staging directory")
        stage_parent = Path(tempfile.mkdtemp(prefix=f"relay-lean-{slug}-", dir=temp_parent))
        stage = stage_parent / "project"
        stage.mkdir(mode=0o700)
        _STAGES[key] = (workspace, stage, digest, stage_parent, None)
        entry = _STAGES[key]
    else:
        old_workspace, stage, old_digest, stage_parent, old_canary = entry
        if old_workspace != workspace or old_digest != digest:
            raise RuntimeError("Candidate Lean source changed during the verifier run")
        _STAGES[key] = (workspace, stage, digest, stage_parent, old_canary)
    workspace, stage, digest, stage_parent, canary_result = _STAGES[key]
    _copy_project(workspace, stage)
    packages = stage / ".lake/packages"
    if not packages.is_dir():
        raise FileNotFoundError("Pinned Lean package cache is absent; network fallback is forbidden")

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
                "--remount-ro", "/", "--remount-ro", "/dev", "--chdir", WORKSPACE,
                "--clearenv", "--setenv", "HOME", "/root", "--setenv", "ELAN_HOME", "/root/.elan",
                "--setenv", "PATH", "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin"]
    child_env = {
        "HOME": "/root", "ELAN_HOME": "/root/.elan",
        "PATH": "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin",
    }
    for key_name in ("LANG", "LC_ALL", "TZ"):
        if key_name in environment:
            child_env[key_name] = environment[key_name]
    if canary_result is None:
        target = {
            "finite-free-stam": "/workspace/FiniteFree",
            "onsager-ising-lean": "/workspace/GSLean/Onsager",
            "gen-turan-paths": "/workspace/GenTuranProj/Submission.lean",
        }[slug]
        script = (
            "set -eu; test -e " + target + "; "
            "test ! -e /tests; test ! -e /logs; test ! -e /usr/project; "
            "test ! -e /home; test ! -e /root/.ssh; test ! -e /run/secrets; "
            "test -d /workspace/.lake/packages; "
            "printf SCIENCE_LEAN_ISOLATION_OK"
        )
        probe = command + ["--", "/bin/sh", "-c", script]
        process = _ORIGINAL_POPEN(probe, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=child_env, close_fds=True)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise TimeoutError("Lean filesystem/network boundary canary timed out")
        canary_result = {
            "task_id": task_id,
            "passed": process.returncode == 0 and b"SCIENCE_LEAN_ISOLATION_OK" in stdout,
            "returncode": process.returncode,
            "checked_absent": ["/tests", "/logs", "/usr/project", "/home", "/root/.ssh", "/run/secrets"],
            "candidate_workspace_visible": b"SCIENCE_LEAN_ISOLATION_OK" in stdout,
        }
        logdir = source_root / "logs/verifier"
        if logdir.is_dir():
            record = logdir / "lean-isolation-canary.json"
            temp = record.with_suffix(".json.tmp")
            temp.write_text(json.dumps(canary_result, indent=2) + "\n")
            temp.chmod(0o600)
            temp.replace(record)
        if not canary_result["passed"]:
            raise PermissionError("Lean filesystem/network boundary canary failed")
        _STAGES[key] = (workspace, stage, digest, stage_parent, canary_result)
    command += ["--", *rewritten]
    return CandidateLaunch(command, child_env, workspace, stage, digest, canary_result)

