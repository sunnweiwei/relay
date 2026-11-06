"""Model-free exact matcher and mount-policy checks for two Python solvers."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.pilot import science_navigation_small_area_isolation as isolation


def callback(module: str):
    namespace: dict[str, object] = {}
    exec(compile("def _demote_solver_process():\n    pass\n", f"/tests/{module}.py", "exec"), namespace)
    function = namespace["_demote_solver_process"]
    function.__module__ = module
    return function


def nav_call(root: str):
    runtime = root + "/runtime"
    output = root + "/output"
    return (
        "terminal-bench-science/navigation-sensor-calibration",
        [sys.executable, "-B", Path(runtime + "/solver.py"), "--input-dir",
         Path(root + "/input"), "--output-dir", Path(output)],
        {"cwd": Path(runtime), "env": {"HOME": output + "/.runtime",
                                  "TMPDIR": output + "/.runtime/tmp",
                                  "XDG_CACHE_HOME": output + "/.runtime/cache",
                                  "PATH": "/usr/local/bin:/usr/bin:/bin"},
         "preexec_fn": callback("test_all"), "start_new_session": True,
         "close_fds": True, "stdin": subprocess.DEVNULL,
         "stdout": object(), "stderr": object()},
    )


def small_call(case: str):
    runtime = case + "/submission"
    return (
        "small-area-equivalence",
        [sys.executable, runtime + "/solve.py", "--data-dir",
         case + "/data", "--output-dir", case + "/output"],
        {"cwd": Path(runtime), "env": {"HOME": case + "/home", "TMPDIR": case + "/home/tmp",
                                  "XDG_CACHE_HOME": case + "/home/cache",
                                  "PATH": "/usr/local/bin:/usr/bin:/bin"},
         "preexec_fn": callback("test_state"),
         "stdout": object(), "stderr": object()},
    )


@pytest.mark.parametrize("task,argv,kwargs", [nav_call("/app/navigation-session-x1"),
                                              small_call("/app/sae_v4_case_x1")])
def test_exact_match_accepts_bare_and_canonical_ids(task, argv, kwargs):
    assert isolation.is_candidate(task, argv, kwargs)
    slug = task.rsplit("/", 1)[-1]
    assert isolation.is_candidate("terminal-bench-science/" + slug, argv, kwargs)
    assert not isolation.is_candidate("other-benchmark/" + slug, argv, kwargs)


def test_probe_and_near_miss_commands_are_not_candidate_launches():
    task, argv, kwargs = nav_call("/app/navigation-session-x1")
    assert not isolation.is_candidate(task, [argv[0], "-B", "-c", "pass"], kwargs)
    assert not isolation.is_candidate(task, [*argv[:2], "/root/results/solver.py", *argv[3:]], kwargs)
    task, argv, kwargs = small_call("/app/sae_v4_case_x1")
    assert not isolation.is_candidate(task, [*argv[:-1], "/tests/data/ground_truth.json"], kwargs)
    assert not isolation.is_candidate(task, argv, {**kwargs, "preexec_fn": None})


def test_exact_navigation_privacy_probe_is_contained_without_verifier_mounts(tmp_path: Path) -> None:
    task, _, _ = nav_call("/app/navigation-session-probe")
    namespace: dict[str, object] = {}
    exec(compile("def _demote_solver_process():\n    pass\n",
                 "/tests/test_all.py", "exec"), namespace)
    callback_fn = namespace["_demote_solver_process"]
    callback_fn.__module__ = "test_all"
    args = [sys.executable, "-B", "-c", isolation.NAV_PRIVACY_PROBE]
    kwargs = {
        "cwd": "/app", "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        "preexec_fn": callback_fn, "start_new_session": True,
    }
    assert isolation.is_privacy_probe(task, args, kwargs)
    assert not isolation.is_privacy_probe(task, [*args[:-1], "pass"], kwargs)
    launch = isolation.prepare_privacy_probe(
        args, environment=kwargs["env"], uid=65532, gid=999,
        source_root=tmp_path, system_root=tmp_path,
    )
    try:
        command = launch.command
        setup = command[:command.index("--", command.index("--chdir"))]
        assert "/tests/cases" not in setup
        assert ["--dir", "/tests"] == [
            setup[i:i + 2] for i, value in enumerate(setup)
            if value == "--dir" and setup[i + 1] == "/tests"
        ][0]
        assert ["--chmod", "0555", "/tests"] == [
            command[i:i + 3] for i, value in enumerate(command)
            if value == "--chmod" and command[i + 2] == "/tests"
        ][0]
        assert ["--chmod", "0555", "/tmp"] == [
            command[i:i + 3] for i, value in enumerate(command)
            if value == "--chmod" and command[i + 2] == "/tmp"
        ][0]
        assert ["--chmod", "0555", "/dev/shm"] == [
            command[i:i + 3] for i, value in enumerate(command)
            if value == "--chmod" and command[i + 2] == "/dev/shm"
        ][0]
        assert "--remount-ro" in command
    finally:
        launch.close()


def _make_source_tree(source: Path, root: str, slug: str) -> dict[str, Path]:
    base = source / root.lstrip("/")
    if slug == "navigation-sensor-calibration":
        candidate = base / "runtime/solver.py"
        input_dir = base / "input"
        output = base / "output"
        home = output / ".runtime"
        tmp, cache = home / "tmp", home / "cache"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("pass\n")
    else:
        candidate = base / "submission"
        input_dir = base / "data"
        output = base / "output"
        home = base / "home"
        tmp, cache = home / "tmp", home / "cache"
        (candidate / "pkg").mkdir(parents=True)
        (candidate / "solve.py").write_text("pass\n")
        (candidate / "pkg/helper.py").write_text("pass\n")
    input_dir.mkdir(parents=True)
    (input_dir / "visible.dat").write_bytes(b"public")
    hidden = source / "tests/hidden.json"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"hidden marker must not be mounted")
    for directory in (output, home, tmp, cache):
        directory.mkdir(parents=True, exist_ok=True)
    return {"candidate": candidate, "input": input_dir, "output": output,
            "home": home, "tmp": tmp, "cache": cache, "hidden": hidden}


@pytest.mark.parametrize("slug,root,call", [
    ("navigation-sensor-calibration", "/app/navigation-session-test", nav_call),
    ("small-area-equivalence", "/app/sae_v4_case_test", small_call),
])
def test_command_mounts_only_submitted_code_public_input_and_scratch(
    tmp_path: Path, monkeypatch, slug: str, root: str, call,
):
    source = tmp_path / "rootfs"
    paths = _make_source_tree(source, root, slug)
    task, argv, kwargs = call(root)
    uid, gid = (65534, 65534) if slug == "small-area-equivalence" else (1001, 1001)
    launch = isolation.build_candidate_command(
        task_id=task, argv=argv, kwargs=kwargs, uid=uid, gid=gid,
        bwrap="/usr/bin/bwrap", source_root=source,
    )
    try:
        command = launch.command
        assert "--unshare-pid" in command and "--unshare-net" in command
        assert "--cap-drop" in command and "ALL" in command
        assert "--ro-bind" in command
        assert "--bind" in command
        assert str(paths["input"]) in command
        assert str(paths["output"]) in command
        assert "/tests" not in command and "/root" not in command
        if slug == "navigation-sensor-calibration":
            assert "--ro-bind-data" in command
            assert str(argv[2]) in command
            assert "--new-session" not in command
            launch._dispatcher.seek(0)
            dispatcher_text = launch._dispatcher.read().decode()
            assert "RLIMIT_FSIZE, (67108864, 67108864)" in dispatcher_text
        else:
            assert "--new-session" in command
            assert str(paths["candidate"]) in command
        assert str(paths["hidden"]) not in command
        assert launch.environment["HOME"] == root + ("/output/.runtime" if slug == "navigation-sensor-calibration" else "/home")
        assert launch.environment["TMPDIR"] == root + ("/output/.runtime/tmp" if slug == "navigation-sensor-calibration" else "/home/tmp")
        output_name = ("calibration.json" if slug == "navigation-sensor-calibration"
                       else "predictive_draws.csv")
        output_file = paths["output"] / output_name
        output_file.write_bytes(b"candidate output")
        outside = source / "outside.dat"
        outside.write_bytes(b"outside")
        symlink = paths["output"] / ("outliers.csv" if slug == "navigation-sensor-calibration"
                                      else "survey_estimates.csv")
        symlink.symlink_to(outside)
        hardlink = paths["output"] / ("trajectory.csv" if slug == "navigation-sensor-calibration"
                                       else "sampling_covariance.csv")
        hardlink.hardlink_to(outside)
        ownership_updates = []
        monkeypatch.setattr(isolation.os, "fchown", lambda descriptor, owner_uid, owner_gid:
                            ownership_updates.append((Path(os.readlink(f"/proc/self/fd/{descriptor}")),
                                                      owner_uid, owner_gid)))
        launch.export_output_ownership(1)
        assert ownership_updates == []
        launch.export_output_ownership(0)
        assert ownership_updates == [(output_file, uid, gid)]
    finally:
        launch.close()
