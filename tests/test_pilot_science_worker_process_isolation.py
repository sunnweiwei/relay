"""Model-free checks for existing Foraging and Noisy worker boundaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_worker_process_isolation as isolation


def _foraging_args(python: str, worker: str = isolation.FORAGING_WORKER) -> list[str]:
    return [python, "-B", worker]


def _noisy_args(python: str, temp: str = "/tmp/dfo-solver-fixture") -> list[str]:
    return [python, isolation.NOISY_WORKER, temp + "/solver.py"]


def test_exact_worker_launches_match_bare_and_canonical_ids() -> None:
    py = "/opt/relay-runtime/bin/python3"
    for slug in ("foraging-cognitive-model", "noisy-blackbox-optimization"):
        task_id = "terminal-bench-science/" + slug
        if slug == "foraging-cognitive-model":
            args = _foraging_args(py)
            kwargs = {"cwd": "/app", "stdin": subprocess.PIPE,
                      "stdout": subprocess.PIPE, "preexec_fn": None,
                      "text": True, "bufsize": 1}
            env = {"VERIFY_MODEL": "/app/model.py"}
        else:
            args = _noisy_args(py)
            kwargs = {"cwd": "/tmp/dfo-solver-fixture", "preexec_fn": lambda: None,
                      "start_new_session": True, "stdin": subprocess.PIPE,
                      "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL,
                      "bufsize": 0}
            env = {"HOME": "/tmp/dfo-solver-fixture"}
        kwargs["env"] = env
        assert isolation.is_candidate(slug, args, kwargs)
        assert isolation.is_candidate(task_id, args, kwargs)
        assert not isolation.is_candidate("other-benchmark/" + slug, args, kwargs)
        assert not isolation.is_candidate(slug, [*args, "extra"], kwargs)


def test_foraging_match_rejects_alternate_candidate_path_and_preexec() -> None:
    args = _foraging_args("/opt/relay-runtime/bin/python3")
    kwargs = {"cwd": "/app", "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
              "env": {}, "preexec_fn": None, "text": True, "bufsize": 1}
    assert not isolation.is_candidate("foraging-cognitive-model", args,
                                      {**kwargs, "env": {"VERIFY_MODEL": "/app/other.py"}})
    assert not isolation.is_candidate("foraging-cognitive-model", args,
                                      {**kwargs, "preexec_fn": lambda: None})
    assert not isolation.is_candidate("foraging-cognitive-model",
                                      _foraging_args("/opt/relay-runtime/bin/python3", "/tests/test_model.py"),
                                      kwargs)


def test_noisy_match_requires_exact_worker_cwd_and_session() -> None:
    args = _noisy_args("/opt/relay-runtime/bin/python3")
    kwargs = {"cwd": "/tmp/dfo-solver-fixture", "preexec_fn": lambda: None,
              "start_new_session": True, "env": {}, "stdin": subprocess.PIPE,
              "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "bufsize": 0}
    assert not isolation.is_candidate("noisy-blackbox-optimization", args,
                                      {**kwargs, "cwd": "/tmp/other"})
    assert not isolation.is_candidate("noisy-blackbox-optimization", args,
                                      {**kwargs, "start_new_session": False})
    assert not isolation.is_candidate("noisy-blackbox-optimization",
                                      _noisy_args("/opt/relay-runtime/bin/python3", "/tmp/other/solver.py"),
                                      kwargs)


def test_bwrap_foraging_worker_preserves_protocol_and_hides_verifier_paths(tmp_path: Path) -> None:
    bwrap = shutil.which("bwrap")
    python = shutil.which("python3")
    if not bwrap or not python:
        pytest.skip("bubblewrap or system Python is unavailable")
    source = tmp_path / "root"
    (source / "tests").mkdir(parents=True)
    (source / "app").mkdir()
    (source / "tests/hidden.marker").write_text("private")
    worker = '''import importlib.util, json, sys
for line in sys.stdin:
    fields=line.strip().split()
    if fields[0] == "L":
        spec=importlib.util.spec_from_file_location("submission", fields[1])
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        print("ok", flush=True)
    elif fields[0] == "P": print("p %.17g" % module.predict_next(), flush=True)
    elif fields[0] == "Q": break
'''
    (source / "tests/worker.py").write_text(worker)
    candidate = '''from pathlib import Path
def predict_next():
    assert not Path("/tests/hidden.marker").exists()
    assert not Path("/app/task.toml").exists()
    assert not Path("/proc/1/root/tests/hidden.marker").exists()
    return 0.375
'''
    (source / "app/model.py").write_text(candidate)
    (source / "app/model.py").chmod(0o444)
    args = _foraging_args(python)
    launch = isolation.build_worker_command(
        task_id="terminal-bench-science/foraging-cognitive-model", argv=args,
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "VERIFY_MODEL": "/app/model.py"},
        cwd="/app", uid=60001, gid=60001, bwrap=bwrap, source_root=source,
    )
    proc = None
    try:
        proc = subprocess.Popen(launch.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=launch.environment,
                                pass_fds=launch.pass_fds)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write("L /app/model.py\nP\nQ\n")
        proc.stdin.flush()
        replies = [proc.stdout.readline().strip(), proc.stdout.readline().strip()]
        assert proc.wait(timeout=10) == 0
        assert replies == ["ok", "p 0.375"]
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        launch.close()


def test_noisy_command_mounts_only_candidate_and_public_worker(tmp_path: Path) -> None:
    source = tmp_path / "root"
    root = source / "tmp/dfo-solver-fixture"
    root.mkdir(parents=True)
    (root / "solver.py").write_text("def solver(fun,x0): return x0\n")
    (source / "tests").mkdir()
    (source / "tests/solver_worker.py").write_text("pass\n")
    launch = isolation.build_worker_command(
        task_id="noisy-blackbox-optimization", argv=_noisy_args("/usr/local/bin/python3"),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/dfo-solver-fixture"},
        cwd="/tmp/dfo-solver-fixture", uid=60001, gid=60001,
        bwrap="/usr/bin/bwrap", source_root=source,
    )
    try:
        command = launch.command
        dispatcher = launch._dispatcher
        dispatcher.seek(0)
        dispatcher_code = dispatcher.read().decode()
        assert "--unshare-net" in command and "--unshare-pid" in command
        assert "--ro-bind-data" in command
        assert "/tests" in command
        assert "/tmp/dfo-solver-fixture/solver.py" in command
        assert "/tests/test_outputs.py" not in command
        assert "/tests/hidden_config.json" not in command
        assert "RLIMIT_NOFILE, (64, 64)" in dispatcher_code
        assert "RLIMIT_NPROC, (64, 64)" in dispatcher_code
        assert launch.environment["HOME"] == "/tmp/dfo-solver-fixture"
        assert "PYTHONPATH" not in launch.environment
    finally:
        launch.close()


def test_bwrap_noisy_worker_preserves_json_result_and_official_limits(tmp_path: Path) -> None:
    bwrap = shutil.which("bwrap")
    python = shutil.which("python3")
    if not bwrap or not python:
        pytest.skip("bubblewrap or system Python is unavailable")
    source = tmp_path / "root"
    root = source / "tmp/dfo-solver-fixture"
    root.mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "tests/hidden_config.json").write_text("private")
    worker = '''import importlib.util, json, resource, sys
request=json.loads(sys.stdin.readline())
spec=importlib.util.spec_from_file_location("candidate", sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
result=module.solver(lambda x: float(sum(x)), request["x0"])
message={"token":request["result_token"],"op":"result","x":list(result)}
sys.stdout.write(json.dumps(message, separators=(",",":"))+"\\n");sys.stdout.flush()
'''
    (source / "tests/solver_worker.py").write_text(worker)
    candidate = '''import resource
from pathlib import Path
def solver(fun, x0):
    assert not Path("/tests/hidden_config.json").exists()
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (64, 64)
    assert resource.getrlimit(resource.RLIMIT_NPROC) == (64, 64)
    return x0
'''
    (root / "solver.py").write_text(candidate)
    (root / "solver.py").chmod(0o444)
    launch = isolation.build_worker_command(
        task_id=isolation.CANONICAL_PREFIX + "noisy-blackbox-optimization",
        argv=_noisy_args(python),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/dfo-solver-fixture"},
        cwd="/tmp/dfo-solver-fixture", uid=60001, gid=60001,
        bwrap=bwrap, source_root=source,
    )
    proc = None
    try:
        proc = subprocess.Popen(
            launch.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=launch.environment, text=True,
            start_new_session=True, pass_fds=launch.pass_fds,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"eval_token": "e", "result_token": "r", "x0": [1.0, 2.0]}) + "\n")
        proc.stdin.flush()
        response = json.loads(proc.stdout.readline())
        assert proc.wait(timeout=10) == 0
        assert response == {"token": "r", "op": "result", "x": [1.0, 2.0]}
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        launch.close()
