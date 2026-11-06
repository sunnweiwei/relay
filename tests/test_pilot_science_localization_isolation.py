"""Model-free checks for the exact localization solver-worker boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_localization_isolation as isolation


TASK_PREFIXES = {
    "baseline-free-localization": "baseline-free-worker-",
    "guided-wave-localization": "guided-wave-worker-",
    "virtual-baseline-localization": "virtual-baseline-worker-",
}


@pytest.mark.parametrize("slug,prefix", TASK_PREFIXES.items())
def test_exact_worker_match_accepts_bare_and_canonical_ids(slug: str, prefix: str) -> None:
    tempdir=f"/tmp/{prefix}fixture"
    args=["/usr/bin/python3", isolation.WORKER_PATH, tempdir+"/solution.py"]
    kwargs={"cwd":tempdir,"preexec_fn":lambda:None}
    assert isolation.is_candidate(slug,args,kwargs)
    assert isolation.is_candidate("terminal-bench-science/"+slug,args,kwargs)
    assert not isolation.is_candidate("other-benchmark/"+slug,args,kwargs)
    assert not isolation.is_candidate(slug,args,{"cwd":tempdir})
    assert not isolation.is_candidate(slug,[*args[:2],"/tmp/unrelated/solution.py"],kwargs)
    assert not isolation.is_candidate(slug,[*args[:2],tempdir+"/other.py"],kwargs)


def test_command_mounts_only_public_worker_and_private_tempdir(tmp_path: Path) -> None:
    source=tmp_path/"root"
    (source/"tests").mkdir(parents=True)
    (source/"tests/solver_worker.py").write_text("pass\n")
    tempdir=source/"tmp/baseline-free-worker-fixture"
    tempdir.mkdir(parents=True)
    (tempdir/"solution.py").write_text("pass\n")
    (tempdir/"solution.py").chmod(0o444)
    launch=isolation.build_worker_command(
        task_id="terminal-bench-science/baseline-free-localization",
        argv=["/usr/bin/python3",isolation.WORKER_PATH,"/tmp/baseline-free-worker-fixture/solution.py"],
        environment={"PATH":"/usr/bin:/bin","HOME":"/tmp/baseline-free-worker-fixture",
                     "TMPDIR":"/tmp/baseline-free-worker-fixture","RELAY_SECRET":"removed"},
        cwd="/tmp/baseline-free-worker-fixture",uid=60001,gid=60001,
        bwrap="/usr/bin/bwrap",source_root=source,
    )
    command,environment=launch.command,launch.environment
    assert "--unshare-net" in command and "--unshare-pid" in command
    binds=[(command[i+1],command[i+2]) for i,value in enumerate(command[:-2])
           if value in {"--bind","--ro-bind"}]
    assert "--ro-bind-data" in command
    assert isolation.EXEC_PATH in command
    assert len(launch.pass_fds)==2
    assert "--dev" in command and "/dev" in command
    assert str(tempdir) in command
    assert environment["HOME"]=="/tmp/baseline-free-worker-fixture"
    assert "RELAY_SECRET" not in environment
    targets={target for _source,target in binds}
    assert "/tests" not in targets and "/app" not in targets
    launch.close()


def test_command_preserves_the_parent_requested_session_semantics(tmp_path: Path) -> None:
    source=tmp_path/"root"
    (source/"tests").mkdir(parents=True)
    (source/"tests/solver_worker.py").write_text("pass\n")
    tempdir=source/"tmp/baseline-free-worker-session"
    tempdir.mkdir(parents=True)
    (tempdir/"solution.py").write_text("pass\n")
    args=["/usr/bin/python3","/tests/solver_worker.py","/tmp/baseline-free-worker-session/solution.py"]
    common=dict(task_id="baseline-free-localization",argv=args,environment={"PATH":"/usr/bin:/bin"},
                cwd="/tmp/baseline-free-worker-session",uid=60001,gid=60001,bwrap="/usr/bin/bwrap",
                source_root=source)
    isolated_session=isolation.build_worker_command(**common,bubblewrap_new_session=True)
    parent_session=isolation.build_worker_command(**common,bubblewrap_new_session=False)
    try:
        assert "--new-session" in isolated_session.command
        assert "--new-session" not in parent_session.command
    finally:
        isolated_session.close()
        parent_session.close()


def test_command_mounts_uv_environment_when_interpreter_is_a_system_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv=tmp_path/"uv-environment"
    (venv/"bin").mkdir(parents=True)
    (venv/"bin/python").symlink_to("/usr/local/bin/python3.12")
    (venv/"pyvenv.cfg").write_text("home = /usr/local/bin\n")
    cache=tmp_path/"uv-cache/archive-v0/package/lib/python3.12/site-packages"
    cache.mkdir(parents=True)
    monkeypatch.setattr(isolation, "UV_CACHE_ROOT", tmp_path/"uv-cache")
    monkeypatch.setattr(isolation.sys, "prefix", str(venv))
    monkeypatch.setattr(isolation.sys, "path", [str(venv/"lib/python3.12/site-packages"),str(cache)])
    roots=isolation._python_runtime_roots(str(venv/"bin/python"),Path("/"))
    assert roots == [(venv, str(venv)),(cache,str(cache))]
    assert isolation._python_runtime_roots("/usr/local/bin/python3.12",Path("/")) == []


def test_bwrap_worker_returns_json_and_cannot_see_hidden_paths(tmp_path: Path) -> None:
    bwrap=shutil.which("bwrap")
    if not bwrap:
        pytest.skip("bubblewrap is unavailable")
    source=tmp_path/"root"
    tests=source/"tests";tests.mkdir(parents=True)
    (tests/"hidden.marker").write_text("hidden verifier data")
    worker='''import importlib.util, json, sys
req=json.loads(sys.stdin.readline())
spec=importlib.util.spec_from_file_location("candidate", req["candidate"])
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result=mod.localize()
print(json.dumps({"token":req["token"],"location":result}))
'''
    (tests/"solver_worker.py").write_text(worker)
    tempdir=source/"tmp/baseline-free-worker-smoke";tempdir.mkdir(parents=True)
    candidate='''from pathlib import Path
def localize():
    if Path("/tests/hidden.marker").exists(): raise RuntimeError("test path visible")
    if Path("/usr/project/private").exists(): raise RuntimeError("host path visible")
    if Path("/proc/1/root/tests/hidden.marker").exists(): raise RuntimeError("proc path visible")
    return [1.25, 2.5]
'''
    (tempdir/"solution.py").write_text(candidate)
    (tempdir/"solution.py").chmod(0o444)
    env={"PATH":"/usr/bin:/bin","HOME":"/tmp/baseline-free-worker-smoke",
         "TMPDIR":"/tmp/baseline-free-worker-smoke","PYTHONDONTWRITEBYTECODE":"1"}
    launch=isolation.build_worker_command(
        task_id="baseline-free-localization",
        argv=["/usr/bin/python3",isolation.WORKER_PATH,"/tmp/baseline-free-worker-smoke/solution.py"],
        environment=env,cwd="/tmp/baseline-free-worker-smoke",uid=60001,gid=60001,
        bwrap=bwrap,source_root=source,system_root=Path("/"),
    )
    command,clean=launch.command,launch.environment
    process=subprocess.run(command,input=json.dumps({"token":"fixture","candidate":"/tmp/baseline-free-worker-smoke/solution.py"})+"\n",
                           text=True,capture_output=True,env=clean,timeout=20,check=False,
                           pass_fds=launch.pass_fds)
    launch.close()
    assert process.returncode==0, process.stderr
    assert json.loads(process.stdout)=={"token":"fixture","location":[1.25,2.5]}
