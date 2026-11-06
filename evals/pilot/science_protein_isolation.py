"""Preserve Protein's official Deno worker protocol in a private namespace."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import sys
from typing import Any, Mapping, Sequence


COMMAND = ("deno", "run", "--no-config", "--no-prompt", "--no-check", "--quiet",
           "--unstable-worker-options", "/tests/js_runner.ts")
SYSTEM = ("/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
          "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share")


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    return (task_id.rsplit("/", 1)[-1] == "protein-active-learning"
            and isinstance(argv, (list, tuple)) and tuple(argv) == COMMAND
            and kwargs.get("cwd") == "/tmp/model-run")


def prepare_candidate(argv: Sequence[str], environment: Mapping[str, str],
                      *, new_session: bool = True) -> tuple[list[str], dict[str, str]]:
    if tuple(argv) != COMMAND:
        raise ValueError("Unexpected Protein Deno invocation")
    command = ["/opt/relay-tools/bwrap", "--unshare-user", "--uid", "65532", "--gid", "65532",
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
    command += ["--proc", "/proc", "--dev", "/dev", "--dir", "/tests", "--dir", "/logs",
                "--tmpfs", "/tmp", "--dir", "/tmp/model-run", "--chdir", "/tmp/model-run",
                "--ro-bind", "/tests/js_runner.ts", "/tests/js_runner.ts",
                "--ro-bind", str(Path(__file__).resolve()), "/opt/relay-protein/limits.py",
                sys.executable, "/opt/relay-protein/limits.py", "--exec", *argv]
    # The official caller supplies exactly this small environment. Preserve
    # its values; no trusted Python hooks or fakeroot transport reach Deno.
    child_env = {k: v for k, v in environment.items() if k in {"HOME", "PATH", "NO_COLOR"}}
    return command, child_env


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--exec" or tuple(sys.argv[2:]) != COMMAND:
        raise SystemExit("Unsupported Protein limits dispatcher invocation")
    # Applying NPROC before bwrap forks its namespace helper can prevent setup.
    # Apply the unchanged official limits immediately before the Deno worker.
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
