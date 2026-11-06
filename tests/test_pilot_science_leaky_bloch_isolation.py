"""Model-free checks for the exact Leaky-Bloch design replay boundary."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_leaky_bloch_isolation as isolation


def replay_call(sandbox: str):
    runtime = sandbox + "/.replay_runtime_x1"
    return (
        "terminal-bench-science/leaky-bloch-meep",
        [isolation.PRLIMIT_PATH, *isolation.LIMITS, "--", "/usr/bin/setpriv",
         "--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs",
         "--", isolation.INTERPRETER_PATH, sandbox + "/design.py", "--output-dir",
         sandbox + "/reproduced"],
        {"cwd": sandbox, "env": {"HOME": runtime + "/home",
                                  "TMPDIR": runtime + "/tmp",
                                  "XDG_CACHE_HOME": runtime + "/cache",
                                  "PATH": "/opt/conda/envs/pymeep-runtime/bin:/usr/bin:/bin"},
         "start_new_session": True, "umask": 0o077,
         "stdin": subprocess.DEVNULL, "stdout": object(), "stderr": object()},
    )


def test_exact_replay_match_and_near_misses():
    task, argv, kwargs = replay_call("/tmp/reproduced_design_x1")
    assert isolation.is_candidate(task, argv, kwargs)
    assert isolation.is_candidate("leaky-bloch-meep", argv, kwargs)
    assert not isolation.is_candidate("other-benchmark/leaky-bloch-meep", argv, kwargs)
    assert not isolation.is_candidate(task, [*argv[:14], "/tests/test_state.py", *argv[15:]], kwargs)
    assert not isolation.is_candidate(task, argv, {**kwargs, "start_new_session": False})
    assert not isolation.is_candidate(task, argv, {**kwargs, "preexec_fn": lambda: None})


def test_command_mounts_only_replay_script_and_output(tmp_path: Path):
    source = tmp_path / "rootfs"
    sandbox = "/tmp/reproduced_design_x1"
    script = source / sandbox.lstrip("/") / "design.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n")
    output = script.parent / "reproduced"
    output.mkdir()
    runtime = script.parent / ".replay_runtime_x1"
    for name in ("home", "tmp", "cache"):
        (runtime / name).mkdir(parents=True)
    hidden = source / "tests/test_state.py"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("hidden verifier marker")
    (source / "opt/conda/envs/pymeep-runtime/bin").mkdir(parents=True)
    task, argv, kwargs = replay_call(sandbox)
    launch = isolation.build_candidate_command(
        task_id=task, argv=argv, kwargs=kwargs, source_root=source,
        system_root=source, uid=65534, gid=65534, bwrap="/usr/bin/bwrap")
    try:
        command = launch.command
        assert "--unshare-pid" in command and "--unshare-net" in command
        assert "--cap-drop" in command and "ALL" in command
        assert "/opt/conda" in command
        assert str(script) in command and str(output) in command and str(runtime) in command
        assert str(hidden) not in command
        assert "/tests" not in command and "/root/results" not in command
        assert command[-1] == sandbox + "/reproduced"
        launch._dispatcher.seek(0)
        assert "libc.prctl(38,1,0,0,0)" in launch._dispatcher.read().decode()
    finally:
        launch.close()
