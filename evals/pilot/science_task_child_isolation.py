"""Exact child boundaries for linked-cell-suppression and Tess transit vetting."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TASKS = frozenset({"linked-cell-suppression", "tess-transit-vetting"})
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
LINKED_ROOT = re.compile(r"/tmp/linked-cell-verifier-[A-Za-z0-9_-]+\Z")
TESS_ROOT = re.compile(r"/run/vetter-runs/invocation-[A-Za-z0-9_-]+\Z")
TESS_UIDS = frozenset(range(65531, 65538))


class CandidateBoundaryError(RuntimeError):
    """The official candidate launch does not match the reviewed profile."""


@dataclass(frozen=True)
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    uid: int
    gid: int


def task_slug(task_id: str) -> str:
    if task_id in TASKS:
        return task_id
    prefix = "terminal-bench-science/"
    if task_id.startswith(prefix) and task_id.count("/") == 1:
        slug = task_id.removeprefix(prefix)
        if slug in TASKS:
            return slug
    raise CandidateBoundaryError(f"No child boundary for Science task {task_id!r}")


def _absolute(value: Any, label: str) -> str:
    result = os.fspath(value)
    path = PurePosixPath(result)
    if not path.is_absolute() or ".." in path.parts:
        raise CandidateBoundaryError(f"{label} is not an absolute normalized path")
    return result


def _uid_from_preexec(preexec_fn: Any, *, allowed: frozenset[int]) -> int:
    closure = getattr(preexec_fn, "__closure__", None)
    captured = []
    for cell in closure or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            captured.append(value)
    matches = [value for value in captured if value in allowed]
    if len(matches) != 1:
        raise CandidateBoundaryError("The official UID preexec closure is not recognized")
    return matches[0]


def _linked_paths(argv: Sequence[Any], cwd: Any) -> tuple[str, str, str, str] | None:
    if len(argv) != 6 or os.fspath(argv[0]) != sys.executable:
        return None
    candidate = _absolute(argv[1], "linked-cell solver")
    root = str(PurePosixPath(candidate).parent)
    if not LINKED_ROOT.fullmatch(root) or PurePosixPath(candidate).name != "protect_tables.py":
        return None
    if cwd is None or _absolute(cwd, "linked-cell cwd") != root:
        return None
    if (os.fspath(argv[2]) != "--instance-dir" or
            _absolute(argv[3], "linked-cell case") != root + "/case" or
            os.fspath(argv[4]) != "--output-file"):
        return None
    output = _absolute(argv[5], "linked-cell output")
    if PurePosixPath(output).parent != PurePosixPath(root + "/output") or PurePosixPath(output).name != "result.csv":
        return None
    return root, candidate, root + "/case", output


def _linked_preexec_valid(preexec_fn: Any) -> bool:
    return (getattr(preexec_fn, "__name__", None) == "_preexec"
            and getattr(preexec_fn, "__closure__", None) in (None, ()))


def _tess_paths_exact(argv: Sequence[Any], cwd: Any) -> tuple[str, str, str, str, str, str, str] | None:
    if len(argv) != 7 or os.fspath(argv[0]) != sys.executable or os.fspath(argv[1]) != "-I":
        return None
    if (os.fspath(argv[2]) != "/app/vetter/vet.py" or os.fspath(argv[3]) != "--manifest"
            or os.fspath(argv[5]) != "--output"):
        return None
    manifest = _absolute(argv[4], "Tess manifest")
    output = _absolute(argv[6], "Tess output")
    root = str(PurePosixPath(manifest).parent.parent)
    if not TESS_ROOT.fullmatch(root) or PurePosixPath(manifest).parent != PurePosixPath(root + "/input"):
        return None
    if not PurePosixPath(manifest).name.startswith("manifest_") or not PurePosixPath(manifest).name.endswith(".json"):
        return None
    if PurePosixPath(output).parent != PurePosixPath(root + "/output"):
        return None
    if not PurePosixPath(output).name.startswith("report_") or not PurePosixPath(output).name.endswith(".json"):
        return None
    if cwd is None or _absolute(cwd, "Tess cwd") != root:
        return None
    return (root, "/app/vetter/vet.py", root + "/input", root + "/output",
            root + "/home", root + "/tmp", root + "/cache")


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    try:
        slug = task_slug(task_id)
    except CandidateBoundaryError:
        return False
    if not isinstance(argv, (list, tuple)) or kwargs.get("preexec_fn") is None:
        return False
    if slug == "linked-cell-suppression":
        return (_linked_paths(argv, kwargs.get("cwd")) is not None
                and _linked_preexec_valid(kwargs.get("preexec_fn")))
    if _tess_paths_exact(argv, kwargs.get("cwd")) is None:
        return False
    try:
        _uid_from_preexec(kwargs.get("preexec_fn"), allowed=TESS_UIDS)
    except CandidateBoundaryError:
        return False
    return True


def _candidate_environment(environment: Mapping[str, str], *, home: str, tmp: str,
                            cache: str | None = None) -> dict[str, str]:
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
    result.update({"HOME": home, "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp})
    if cache is not None:
        result["XDG_CACHE_HOME"] = cache
    return result


def _limited_exec_script(*, cpu: int, address_space: int,
                         file_size: int, nofile: int, nproc: int, core: int = 0) -> str:
    return (
        "import ctypes,os,resource,sys; "
        "libc=ctypes.CDLL(None,use_errno=True); "
        "libc.prctl(38,1,0,0,0); "
        f"resource.setrlimit(resource.RLIMIT_CORE,({core},{core})); "
        f"resource.setrlimit(resource.RLIMIT_CPU,({cpu},{cpu})); "
        f"resource.setrlimit(resource.RLIMIT_FSIZE,({file_size},{file_size})); "
        f"resource.setrlimit(resource.RLIMIT_NOFILE,({nofile},{nofile})); "
        f"resource.setrlimit(resource.RLIMIT_NPROC,({nproc},{nproc})); "
        f"resource.setrlimit(resource.RLIMIT_AS,({address_space},{address_space})); "
        "os.umask(0o077); os.execv(sys.argv[1],sys.argv[1:])"
    )


def _boundary_prefix(*, uid: int, gid: int, source_root: Path, bwrap: str,
                     extra_dirs: Sequence[str], system_paths: Sequence[str] = SYSTEM_PATHS) -> list[str]:
    command = [bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
               "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
               "--cap-drop", "ALL", "--die-with-parent", "--new-session"]
    for path in system_paths:
        source = source_root / path.lstrip("/")
        if source.is_symlink():
            command += ["--symlink", os.readlink(source), path]
        elif source.exists():
            command += ["--ro-bind", str(source), path]
    command += ["--proc", "/proc", "--dev", "/dev"]
    for path in ("/app", "/tests", "/logs", "/tmp", "/var", "/run", "/opt"):
        command += ["--dir", path]
    command += ["--tmpfs", "/tests", "--tmpfs", "/logs", "--tmpfs", "/tmp",
                "--tmpfs", "/var/tmp", "--tmpfs", "/run", "--tmpfs", "/opt"]
    for path in extra_dirs:
        command += ["--dir", path]
    return command


def prepare_linked_cell(
    task_id: str, argv: Sequence[Any], *, environment: Mapping[str, str], cwd: Any,
    preexec_fn: Any, bwrap: str = BWRAP_PATH, source_root: Path = Path("/"),
    system_paths: Sequence[str] = SYSTEM_PATHS,
) -> CandidateLaunch:
    if task_slug(task_id) != "linked-cell-suppression":
        raise CandidateBoundaryError("Not a linked-cell-suppression launch")
    paths = _linked_paths(argv, cwd)
    if paths is None or not _linked_preexec_valid(preexec_fn):
        raise CandidateBoundaryError("Linked-cell launch differs from the exact verifier argv")
    root, candidate, case, output = paths
    expected_output_dir = root + "/output"
    if (environment.get("HOME") != expected_output_dir
            or environment.get("TMPDIR") != expected_output_dir
            or environment.get("PYTHONPATH") != ""):
        raise CandidateBoundaryError("Linked-cell candidate environment differs from the official allowlist")
    root_path = source_root / root.lstrip("/")
    candidate_path = source_root / candidate.lstrip("/")
    case_path = source_root / case.lstrip("/")
    output_path = source_root / (root + "/output").lstrip("/")
    if any(path.is_symlink() or not path.is_dir() for path in (root_path, case_path, output_path)):
        raise CandidateBoundaryError("Linked-cell run directories are missing or invalid")
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise CandidateBoundaryError("Linked-cell solver is missing or invalid")
    if not (case_path / "instance.json").is_file() or not (case_path / "cells.csv").is_file():
        raise CandidateBoundaryError("Linked-cell case clone is incomplete")
    uid, gid = 65534, 65534
    resource_script = _limited_exec_script(cpu=1800, address_space=7 * 1024**3,
                                           file_size=8 * 1024**2, nofile=256, nproc=256)
    prefix = _boundary_prefix(uid=uid, gid=gid, source_root=source_root, bwrap=bwrap,
                              extra_dirs=(root, root + "/case", root + "/output"), system_paths=system_paths)
    prefix += ["--ro-bind", str(candidate_path), candidate,
               "--ro-bind", str(case_path), case,
               "--bind", str(output_path), root + "/output",
               "--remount-ro", "/", "--remount-ro", "/dev",
               "--chdir", os.fspath(cwd), "--clearenv"]
    child_env = _candidate_environment(environment, home=root + "/output", tmp=root + "/output")
    for key, value in sorted(child_env.items()):
        prefix += ["--setenv", key, value]
    prefix += ["--", sys.executable, "-c", resource_script, *[os.fspath(v) for v in argv]]
    return CandidateLaunch(prefix, child_env, uid, gid)


def prepare_tess(
    task_id: str, argv: Sequence[Any], *, environment: Mapping[str, str], cwd: Any,
    preexec_fn: Any, bwrap: str = BWRAP_PATH, source_root: Path = Path("/"),
    system_paths: Sequence[str] = SYSTEM_PATHS,
) -> CandidateLaunch:
    if task_slug(task_id) != "tess-transit-vetting":
        raise CandidateBoundaryError("Not a Tess launch")
    if _tess_paths_exact(argv, cwd) is None:
        raise CandidateBoundaryError("Tess launch differs from the exact verifier argv")
    uid = _uid_from_preexec(preexec_fn, allowed=TESS_UIDS)
    gid = uid
    tess_paths = _tess_paths_exact(argv, cwd)
    if tess_paths is None:
        raise CandidateBoundaryError("Tess launch differs from the exact verifier argv")
    root, candidate_path, input_path, output_path, home_path, tmp_path, cache_path = tess_paths
    expected_env = {"HOME": home_path, "TMPDIR": tmp_path, "XDG_CACHE_HOME": cache_path}
    if any(environment.get(key) != value for key, value in expected_env.items()):
        raise CandidateBoundaryError("Tess candidate environment differs from the official per-run paths")
    manifest = os.fspath(argv[4]); output = os.fspath(argv[6])
    root_path = source_root / root.lstrip("/")
    paths = {name: root_path / name for name in ("input", "output", "home", "tmp", "cache")}
    candidate = source_root / "app/vetter/vet.py"
    if root_path.is_symlink() or not root_path.is_dir() or any(p.is_symlink() or not p.is_dir() for p in paths.values()):
        raise CandidateBoundaryError("Tess invocation directories are missing or invalid")
    if candidate.is_symlink() or not candidate.is_file():
        raise CandidateBoundaryError("Tess submitted vetter is missing or invalid")
    if (candidate_path != "/app/vetter/vet.py" or input_path != root + "/input"
            or output_path != root + "/output" or home_path != root + "/home"
            or tmp_path != root + "/tmp" or cache_path != root + "/cache"):
        raise CandidateBoundaryError("Tess mount mapping differs from the reviewed invocation layout")
    if not (source_root / manifest.lstrip("/")).is_file() or not (source_root / output.lstrip("/")).parent.is_dir():
        raise CandidateBoundaryError("Tess input manifest or output directory is missing")
    resource_script = _limited_exec_script(cpu=150, address_space=3 * 1024**3,
                                           file_size=16 << 20, nofile=128, nproc=64, core=0)
    prefix = _boundary_prefix(uid=uid, gid=gid, source_root=source_root, bwrap=bwrap,
                              extra_dirs=("/app/vetter", "/run/vetter-runs", root,
                                          root + "/input", root + "/output", root + "/home",
                                          root + "/tmp", root + "/cache"),
                              system_paths=system_paths)
    prefix += ["--ro-bind", str(candidate), "/app/vetter/vet.py",
               "--ro-bind", str(paths["input"]), root + "/input",
               "--bind", str(paths["output"]), root + "/output",
               "--bind", str(paths["home"]), root + "/home",
               "--bind", str(paths["tmp"]), root + "/tmp",
               "--bind", str(paths["cache"]), root + "/cache",
               "--remount-ro", "/", "--remount-ro", "/dev",
               "--chdir", root, "--clearenv"]
    child_env = _candidate_environment(environment, home=root + "/home", tmp=root + "/tmp",
                                       cache=root + "/cache")
    for key, value in sorted(child_env.items()):
        prefix += ["--setenv", key, value]
    prefix += ["--", sys.executable, "-c", resource_script, *[os.fspath(v) for v in argv]]
    return CandidateLaunch(prefix, child_env, uid, gid)

