"""Strict child boundary for the DNA storage encoder and decoder.

The official verifier runs the channel simulator in its own trusted process.
This module only recognizes the two submitted-code launches and stages a
minimal, phase-specific ``/app`` view for each candidate process.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "terminal-bench-science/dna-storage-codec"
TASK_SLUG = "dna-storage-codec"
APP_FILES = frozenset({"encoder.py", "decoder.py"})
OUTPUTS = {
    "encoder": "original_sequences.dat",
    "decoder": "reconstructed.txt",
}
INPUTS = {
    "encoder": "data.txt",
    "decoder": "sequenced.dat",
}
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share", "/usr/include",
)


@dataclass
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    scratch: Path
    phase: str
    destination: Path

    def export_output(self, returncode: int) -> None:
        """Publish only the phase's declared output after a successful child."""
        staged = self.scratch / "app" / OUTPUTS[self.phase]
        try:
            if returncode != 0 or not staged.is_file() or staged.is_symlink():
                return
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.destination.with_name(self.destination.name + ".relay-tmp")
            temporary.write_bytes(staged.read_bytes())
            os.replace(temporary, self.destination)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)


def _phase(argv: Sequence[str]) -> str | None:
    args = [os.fspath(arg) for arg in argv]
    if len(args) != 3 or args[0] != "/usr/bin/python3":
        return None
    if args[1:] == ["/app/encoder.py", "/app/data.txt"]:
        return "encoder"
    if args[1:] == ["/app/decoder.py", "/app/sequenced.dat"]:
        return "decoder"
    return None


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Match only the official UID-dropped encoder/decoder argv forms."""
    if task_id not in {TASK_ID, TASK_SLUG}:
        return False
    phase = _phase(argv) if isinstance(argv, (list, tuple)) else None
    return phase is not None and kwargs.get("user") is not None and kwargs.get("group") is not None


def prepare_candidate(
    task_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str | None,
    source_root: Path = Path("/"),
    bwrap: str = "/opt/relay-tools/bwrap",
    uid: int = 60001,
    gid: int = 60001,
) -> CandidateLaunch:
    """Build a closed, phase-specific namespace for one official launch.

    The verifier's live ``/app`` is never bind-mounted. Candidate files and the
    phase input are copied into a private scratch tree. The candidate can write
    only inside that tree; one declared output is copied back after exit 0.
    """
    if task_id not in {TASK_ID, TASK_SLUG}:
        raise ValueError("DNA codec boundary used for the wrong task")
    args = [os.fspath(item) for item in argv]
    phase = _phase(args)
    if phase is None:
        raise ValueError("Unexpected DNA codec candidate argv")
    if cwd not in {None, "/app"}:
        raise ValueError("DNA codec candidate cwd must remain /app")

    live_app = source_root / "app"
    candidate_files = [live_app / name for name in sorted(APP_FILES)]
    candidate_files = [path for path in candidate_files if path.is_file() and not path.is_symlink()]
    candidate_script = live_app / ("encoder.py" if phase == "encoder" else "decoder.py")
    input_path = live_app / INPUTS[phase]
    if candidate_script not in candidate_files or not input_path.is_file() or input_path.is_symlink():
        raise FileNotFoundError("DNA codec candidate or declared phase input is missing")

    scratch = Path(tempfile.mkdtemp(prefix="relay-dna-codec-"))
    app = scratch / "app"
    app.mkdir(mode=0o700)
    # Preserve candidate helper modules and packages, but never copy the verifier
    # input other than the single phase-specific file or any generated artifacts.
    excluded = {"data.txt", "original_sequences.dat", "sequenced.dat", "reconstructed.txt"}
    for entry in sorted(live_app.rglob("*")):
        if entry.is_symlink():
            shutil.rmtree(scratch, ignore_errors=True)
            raise ValueError("Symlinked DNA codec candidate files are unsupported")
    for entry in sorted(live_app.iterdir()):
        if entry.name in excluded:
            continue
        target = app / entry.name
        if entry.is_symlink():
            shutil.rmtree(scratch, ignore_errors=True)
            raise ValueError("Symlinked DNA codec candidate files are unsupported")
        if entry.is_dir():
            shutil.copytree(entry, target, symlinks=True)
        elif entry.is_file():
            shutil.copy2(entry, target)
    for entry in app.rglob("*"):
        if entry.is_symlink():
            shutil.rmtree(scratch, ignore_errors=True)
            raise ValueError("Symlinked DNA codec helper files are unsupported")

    staged_input = app / INPUTS[phase]
    shutil.copyfile(input_path, staged_input)
    staged_input.chmod(0o444)
    for entry in app.rglob("*"):
        if entry.is_file():
            entry.chmod(0o444)

    # The private copied /app is writable as scratch, then candidate code and
    # the one permitted input are overmounted read-only. Thus temp files do not
    # survive the phase and cannot cross from encoder to decoder.
    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--cap-drop", "ALL",
        "--die-with-parent", "--new-session",
    ]
    for path in SYSTEM_PATHS:
        if Path(path).is_symlink():
            command += ["--symlink", os.readlink(path), path]
        elif Path(path).exists():
            command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/app"]
    for path in ("/tmp", "/var/tmp", "/run/lock", "/tests", "/logs", "/results"):
        command += ["--dir", path]
    command += ["--bind", str(app), "/app"]
    command += ["--ro-bind", str(staged_input), f"/app/{INPUTS[phase]}"]
    # Candidate source files and helper files are read-only, while generated
    # non-output scratch files remain confined to this disposable copied tree.
    for entry in sorted(app.iterdir()):
        if entry.name == INPUTS[phase]:
            continue
        if entry.is_file():
            command += ["--ro-bind", str(entry), f"/app/{entry.name}"]
        elif entry.is_dir():
            command += ["--ro-bind", str(entry), f"/app/{entry.name}"]
    command += ["--remount-ro", "/", "--remount-ro", "/dev"]
    if cwd:
        command += ["--chdir", cwd]
    command += ["--", *args]

    clean_env = {
        key: value for key, value in environment.items()
        if not key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED"))
        and key not in {
            "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH",
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
        }
    }
    clean_env.update(PATH="/usr/local/bin:/usr/bin:/bin", HOME="/app", TMPDIR="/app", TEMP="/app", TMP="/app")
    return CandidateLaunch(command, clean_env, scratch, phase, live_app / OUTPUTS[phase])

