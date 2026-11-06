"""Isolate the public TAMP skill worker in both simulation and verification."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKERS = frozenset({"/tests/tamp_core/skill_runtime.py",
                     "/app/sim_local/tamp_core/skill_runtime.py"})
SYSTEM = ("/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
          "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
          "/opt/libero")


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    return (task_id.rsplit("/", 1)[-1] == "tamp-skill-planning"
            and isinstance(argv, (list, tuple)) and len(argv) == 3
            and argv[1] in WORKERS)


def prepare_candidate(argv: Sequence[str], environment: Mapping[str, str],
                      *, new_session: bool, cwd: str | None = None) -> tuple[list[str], dict[str, str]]:
    if len(argv) != 3 or argv[1] not in WORKERS:
        raise ValueError("Unexpected TAMP skill worker invocation")
    source = Path(argv[2])
    allowed = (str(source) == "/tmp/graded_skills.py" or
               (source.parent == Path("/tmp/tamp_skills") and
                source.name.startswith("skills_") and source.suffix == ".py"))
    if not allowed or not source.is_file() or source.is_symlink():
        raise ValueError("TAMP candidate is outside its official staging path")
    uid, gid = environment.get("TAMP_WORKER_UID", "65534"), environment.get("TAMP_WORKER_GID", "65534")
    if (uid, gid) != ("65534", "65534"):
        raise ValueError("TAMP worker identity differs from the pinned task")
    command = ["/opt/relay-tools/bwrap", "--unshare-user", "--uid", uid, "--gid", gid,
               "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
               "--cap-drop", "ALL", "--die-with-parent"]
    if new_session:
        command += ["--new-session"]
    for name in SYSTEM:
        path = Path(name)
        if path.is_symlink():
            command += ["--symlink", os.readlink(path), name]
        elif path.exists():
            command += ["--ro-bind", name, name]
    command += ["--proc", "/proc", "--dev", "/dev", "--dir", "/tests", "--dir", "/app",
                "--dir", "/logs", "--dir", "/root", "--tmpfs", "/tmp", "--tmpfs", "/run",
                "--ro-bind", argv[1], argv[1], "--ro-bind", str(source), str(source)]
    # The documented skill API permits this public motion library. Mount only
    # that file beside the worker, not adjacent verifier predicates or state.
    library = Path(argv[1]).with_name("vendored_heuristic.py")
    if not library.is_file():
        raise ValueError("TAMP public motion library is missing")
    command += ["--ro-bind", str(library), str(library)]
    workdir = cwd or os.getcwd()
    if workdir not in {"/", "/app", "/tests"}:
        raise ValueError("Unexpected TAMP worker parent working directory")
    command += ["--chdir", workdir, "--", *argv]
    result = {}
    for key, value in environment.items():
        if key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED")):
            continue
        if key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}:
            continue
        if key in {"PATH", "PYTHONPATH"}:
            value = ":".join(part for part in value.split(":") if part and not part.startswith(
                ("/opt/relay-isolation", "/opt/relay-tools")))
            if not value:
                continue
        result[key] = value
    return command, result
