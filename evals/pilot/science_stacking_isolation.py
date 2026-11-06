"""Exact candidate boundary for stacking-disorder-diffraction."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


TASK_SLUG = "stacking-disorder-diffraction"
TASK_IDS = frozenset({TASK_SLUG, "terminal-bench-science/" + TASK_SLUG})
TASKS = frozenset({TASK_SLUG})
REFERENCE_SHA256 = "2affdab091991828b01be1951042b2b3fcf6c1aad5c2bdd66f2629fb7835fdfd"
BWRAP_PATH = "/opt/relay-tools/bwrap"
PACKET_ROOT = re.compile(r"/workspace/stacking_packet_[A-Za-z0-9_-]+/data\Z")
REFERENCE_ROOT = re.compile(r"/workspace/stacking_reference_[A-Za-z0-9_-]+/solver\Z")
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
)


class CandidateBoundaryError(RuntimeError):
    """The official candidate launch does not match the reviewed profile."""


@dataclass(frozen=True)
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    uid: int = 65534
    gid: int = 65534


def _task_matches(task_id: str) -> bool:
    return task_id in TASK_IDS


def _absolute(value: Any, label: str) -> str:
    result = os.fspath(value)
    path = PurePosixPath(result)
    if not path.is_absolute() or ".." in path.parts:
        raise CandidateBoundaryError(f"{label} is not an absolute normalized path")
    return result


def _candidate_paths(argv: Sequence[Any], cwd: Any) -> tuple[str, str, str] | None:
    if len(argv) != 6 or os.fspath(argv[0]) != sys.executable:
        return None
    solver = _absolute(argv[1], "stacking solver")
    if solver != "/workspace/stacking_submission/solve_stacking.py":
        return None
    if (os.fspath(argv[2]) != "--data-dir" or os.fspath(argv[4]) != "--output"
            or cwd is None or _absolute(cwd, "stacking cwd") != "/workspace/stacking_submission"):
        return None
    packet = _absolute(argv[3], "stacking packet")
    if not PACKET_ROOT.fullmatch(packet):
        return None
    output = _absolute(argv[5], "stacking output")
    if output != "/workspace/stacking_case_results/trajectory.csv":
        return None
    return solver, packet, output


def _reference_paths(argv: Sequence[Any], cwd: Any) -> tuple[str, str] | None:
    """Recognize only the verifier's copied reference source with pinned hash."""
    if len(argv) != 6 or os.fspath(argv[0]) != sys.executable:
        return None
    solver = _absolute(argv[1], "stacking reference solver")
    root = str(PurePosixPath(solver).parent)
    if (not REFERENCE_ROOT.fullmatch(root)
            or PurePosixPath(solver).name != "solve_stacking.py"
            or os.fspath(argv[2]) != "--data-dir" or os.fspath(argv[4]) != "--output"
            or cwd is None or _absolute(cwd, "stacking reference cwd") != root):
        return None
    return solver, _absolute(argv[3], "stacking reference packet")


def is_reference_command(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if not _task_matches(task_id) or not isinstance(argv, (list, tuple)):
        return False
    paths = _reference_paths(argv, kwargs.get("cwd"))
    if paths is None:
        return False
    solver, packet = paths
    if not PACKET_ROOT.fullmatch(packet):
        return False
    solver_root = PurePosixPath(solver).parent
    reference_root = solver_root.parent
    output = _absolute(argv[5], "stacking reference output")
    expected_output = str(reference_root / "trajectory.csv")
    env = kwargs.get("env")
    if (output != expected_output
            or kwargs.get("user") != 65534
            or kwargs.get("group") != 65534
            or kwargs.get("extra_groups") != ()
            or kwargs.get("start_new_session") is not True
            or not isinstance(env, Mapping)
            or env.get("HOME") != str(reference_root / "home")
            or env.get("TMPDIR") != str(reference_root / "home" / "tmp")
            or env.get("XDG_CACHE_HOME") != str(reference_root / "home" / "cache")
            or kwargs.get("text") is not True
            or kwargs.get("stdout") != subprocess.PIPE
            or kwargs.get("stderr") != subprocess.PIPE
            or kwargs.get("preexec_fn") is not None):
        return False
    return True


def is_verified_reference(
    task_id: str, argv: Any, kwargs: Mapping[str, Any], *, source_root: Path = Path("/")
) -> bool:
    if not is_reference_command(task_id, argv, kwargs):
        return False
    paths = _reference_paths(argv, kwargs.get("cwd"))
    assert paths is not None
    solver, _packet = paths
    source = Path(source_root) / solver.lstrip("/")
    if source.is_symlink() or not source.is_file():
        return False
    return hashlib.sha256(source.read_bytes()).hexdigest() == REFERENCE_SHA256


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if not _task_matches(task_id) or not isinstance(argv, (list, tuple)):
        return False
    paths = _candidate_paths(argv, kwargs.get("cwd"))
    if paths is None:
        return False
    solver, packet, output = paths
    return (
        kwargs.get("user") == 65534
        and kwargs.get("group") == 65534
        and kwargs.get("extra_groups") == ()
        and kwargs.get("start_new_session") is True
        and kwargs.get("text") is True
        and kwargs.get("stdout") == subprocess.PIPE
        and kwargs.get("stderr") == subprocess.PIPE
        and kwargs.get("preexec_fn") is None
        and kwargs.get("env") is not None
        and str(kwargs["env"].get("HOME", "")) == "/workspace/stacking_solver_home"
        and str(kwargs["env"].get("TMPDIR", "")) == "/workspace/stacking_solver_home/tmp"
        and str(kwargs["env"].get("XDG_CACHE_HOME", "")) == "/workspace/stacking_solver_home/cache"
        and "PYTHONPATH" not in kwargs["env"]
        and "PYTHONHOME" not in kwargs["env"]
        and solver == "/workspace/stacking_submission/solve_stacking.py"
        and output == "/workspace/stacking_case_results/trajectory.csv"
        and PACKET_ROOT.fullmatch(packet) is not None
    )


def _clean_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in environment.items():
        if key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED")):
            continue
        if key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}:
            continue
        if key == "PYTHONPATH":
            continue
        if key == "PATH":
            value = ":".join(part for part in value.split(":")
                             if part and not part.startswith(("/opt/relay-isolation", "/opt/relay-tools")))
        result[key] = value
    result.update({"HOME": "/workspace/stacking_solver_home",
                   "TMPDIR": "/workspace/stacking_solver_home/tmp",
                   "TMP": "/workspace/stacking_solver_home/tmp",
                   "TEMP": "/workspace/stacking_solver_home/tmp",
                   "XDG_CACHE_HOME": "/workspace/stacking_solver_home/cache"})
    return result


def prepare_candidate(
    task_id: str, argv: Sequence[Any], *, environment: Mapping[str, str], cwd: Any,
    uid: int = 65534, gid: int = 65534, bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"), system_paths: Sequence[str] = SYSTEM_PATHS,
) -> CandidateLaunch:
    if not _task_matches(task_id) or not is_candidate(
            task_id, argv, {"user": uid, "group": gid, "extra_groups": (),
                            "start_new_session": True, "cwd": cwd,
                            "env": environment, "text": True,
                            "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}):
        raise CandidateBoundaryError("Stacking solver launch differs from the exact verifier profile")
    paths = _candidate_paths(argv, cwd)
    assert paths is not None
    solver, packet, _output = paths
    root = Path(source_root)
    solver_source = root / "workspace/stacking_submission"
    packet_source = root / packet.lstrip("/")
    packet_parent = packet_source.parent
    output_source = root / "workspace/stacking_case_results"
    home_source = root / "workspace/stacking_solver_home"
    candidate_file = solver_source / "solve_stacking.py"
    if (solver_source.is_symlink() or not solver_source.is_dir()
            or candidate_file.is_symlink() or not candidate_file.is_file()):
        raise CandidateBoundaryError("Stacking solver is missing or invalid")
    if packet_source.is_symlink():
        raise CandidateBoundaryError("Stacking runtime packet is invalid")
    if not packet_source.is_dir() or packet_parent.is_symlink():
        raise CandidateBoundaryError("Stacking runtime packet is missing or invalid")
    if any(path.is_symlink() or not path.is_dir() for path in (output_source, home_source)):
        raise CandidateBoundaryError("Stacking output or home directory is missing or invalid")
    # The official verifier builds the hidden inference packet from the
    # author data. The child receives the transformed public inputs, not the
    # source trajectory profiles used to produce the verifier's private truth.
    for name in ("inverse_parameters.json", "scattering_counts.npz"):
        if not (packet_source / name).is_file():
            raise CandidateBoundaryError("Stacking runtime packet is incomplete")

    command = [bwrap, "--die-with-parent", "--unshare-user", "--uid", str(uid), "--gid", str(gid),
               "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
               "--cap-drop", "ALL"]
    for path in system_paths:
        host = root / path.lstrip("/")
        if host.is_symlink():
            command += ["--symlink", os.readlink(host), path]
        elif host.exists():
            command += ["--ro-bind", str(host), path]
    command += ["--proc", "/proc", "--dev", "/dev"]
    packet_parent = str(PurePosixPath(packet).parent)
    for path in ("/workspace", "/tests", "/logs", "/tmp", "/var/tmp", "/run", "/opt"):
        command += ["--dir", path]
    command += ["--tmpfs", "/workspace", "--tmpfs", "/tests", "--tmpfs", "/logs", "--tmpfs", "/tmp",
                "--tmpfs", "/var/tmp", "--tmpfs", "/run", "--tmpfs", "/opt",
                "--dir", "/workspace/stacking_submission",
                "--dir", packet_parent, "--dir", packet,
                "--dir", "/workspace/stacking_case_results",
                "--dir", "/workspace/stacking_solver_home",
                "--dir", "/workspace/stacking_solver_home/tmp",
                "--dir", "/workspace/stacking_solver_home/cache",
                "--ro-bind", str(solver_source), "/workspace/stacking_submission",
                "--ro-bind", str(packet_source), packet,
                "--bind", str(output_source), "/workspace/stacking_case_results",
                "--bind", str(home_source), "/workspace/stacking_solver_home",
                "--remount-ro", "/", "--remount-ro", "/dev",
                "--chdir", "/workspace/stacking_submission", "--clearenv"]
    clean = _clean_environment(environment)
    for key, value in sorted(clean.items()):
        command += ["--setenv", key, value]
    command += ["--", *[os.fspath(value) for value in argv]]
    return CandidateLaunch(command, clean, uid, gid)


def prepare_reference(
    task_id: str, argv: Sequence[Any], *, environment: Mapping[str, str], cwd: Any,
    uid: int = 65534, gid: int = 65534, bwrap: str = BWRAP_PATH,
    source_root: Path = Path("/"), system_paths: Sequence[str] = SYSTEM_PATHS,
) -> CandidateLaunch:
    """Run the pinned reference through the same boundary as the candidate.

    The verifier still measures the original Popen-to-communicate interval.
    This changes only the reference process setup so namespace startup is
    included symmetrically in the official runtime ratio.
    """
    reference_kwargs = {
        "user": uid, "group": gid, "extra_groups": (),
        "start_new_session": True, "cwd": cwd, "env": environment,
        "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "preexec_fn": None,
    }
    if not is_verified_reference(task_id, argv, reference_kwargs, source_root=source_root):
        raise CandidateBoundaryError("Stacking reference launch differs from the pinned verifier profile")
    paths = _reference_paths(argv, cwd)
    assert paths is not None
    solver, packet = paths
    solver_guest = str(PurePosixPath(solver))
    reference_guest = str(PurePosixPath(solver_guest).parent.parent)
    output_guest = _absolute(argv[5], "stacking reference output")
    home_guest = str(PurePosixPath(reference_guest) / "home")
    root = Path(source_root)
    solver_source = root / str(PurePosixPath(solver_guest).parent).lstrip("/")
    reference_source = solver_source.parent
    packet_source = root / packet.lstrip("/")
    home_source = reference_source / "home"
    solver_file = solver_source / "solve_stacking.py"
    if (solver_source.is_symlink() or not solver_source.is_dir()
            or solver_file.is_symlink() or not solver_file.is_file()
            or hashlib.sha256(solver_file.read_bytes()).hexdigest() != REFERENCE_SHA256):
        raise CandidateBoundaryError("Pinned stacking reference is missing or changed")
    if (reference_source.is_symlink() or not reference_source.is_dir()
            or packet_source.is_symlink() or not packet_source.is_dir()
            or home_source.is_symlink() or not home_source.is_dir()):
        raise CandidateBoundaryError("Stacking reference runtime paths are missing or invalid")
    if {path.name for path in reference_source.iterdir()} - {"solver", "home", "trajectory.csv"}:
        raise CandidateBoundaryError("Stacking reference runtime contains unexpected files")
    if any(path.is_symlink() or not path.is_dir() for path in
           (home_source, home_source / "tmp", home_source / "cache")):
        raise CandidateBoundaryError("Stacking reference home is invalid")

    command = [bwrap, "--die-with-parent", "--unshare-user", "--uid", str(uid), "--gid", str(gid),
               "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net",
               "--cap-drop", "ALL"]
    for path in system_paths:
        host = root / path.lstrip("/")
        if host.is_symlink():
            command += ["--symlink", os.readlink(host), path]
        elif host.exists():
            command += ["--ro-bind", str(host), path]
    command += ["--proc", "/proc", "--dev", "/dev"]
    packet_parent = str(PurePosixPath(packet).parent)
    for path in ("/workspace", "/tests", "/logs", "/tmp", "/var/tmp", "/run", "/opt"):
        command += ["--dir", path]
    command += ["--tmpfs", "/workspace", "--tmpfs", "/tests", "--tmpfs", "/logs", "--tmpfs", "/tmp",
                "--tmpfs", "/var/tmp", "--tmpfs", "/run", "--tmpfs", "/opt",
                "--dir", reference_guest, "--dir", packet_parent, "--dir", packet,
                "--bind", str(reference_source), reference_guest,
                "--ro-bind", str(packet_source), packet,
                "--ro-bind", str(solver_source), str(PurePosixPath(solver_guest).parent),
                "--bind", str(home_source), home_guest,
                "--remount-ro", "/", "--remount-ro", "/dev",
                "--chdir", str(PurePosixPath(solver_guest).parent), "--clearenv"]
    clean = _clean_environment(environment)
    clean.update({"HOME": home_guest, "TMPDIR": home_guest + "/tmp",
                  "TMP": home_guest + "/tmp", "TEMP": home_guest + "/tmp",
                  "XDG_CACHE_HOME": home_guest + "/cache"})
    for key, value in sorted(clean.items()):
        command += ["--setenv", key, value]
    command += ["--", *[os.fspath(value) for value in argv]]
    return CandidateLaunch(command, clean, uid, gid)
