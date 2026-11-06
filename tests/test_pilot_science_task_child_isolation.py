from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evals.pilot import science_isolation
from evals.pilot import science_task_child_isolation as child


def _preexec() -> None:
    pass


def _candidate_preexec(value: int) -> None:
    del value


def _linked_fixture(root: Path) -> tuple[list[str], dict[str, object], Path]:
    logical_root = Path("/tmp/linked-cell-verifier-unit_001")
    run_root = root / logical_root.relative_to("/")
    case = run_root / "case"
    output = run_root / "output"
    case.mkdir(parents=True)
    output.mkdir()
    (run_root / "protect_tables.py").write_text("print('candidate')\n")
    for name in ("instance.json", "cells.csv", "relations.csv", "occurrences.csv"):
        (case / name).write_text("fixture\n")
    cwd = str(logical_root)
    argv = [sys.executable, str(logical_root / "protect_tables.py"), "--instance-dir",
            str(logical_root / "case"), "--output-file", str(logical_root / "output/result.csv")]
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "preexec_fn": _preexec,
        "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(logical_root / "output"),
                "TMPDIR": str(logical_root / "output"), "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
    }
    return argv, kwargs, root


def test_linked_cell_matches_exact_solver_child_and_both_task_id_forms(tmp_path: Path) -> None:
    argv, kwargs, _ = _linked_fixture(tmp_path)
    for task in ("linked-cell-suppression", "terminal-bench-science/linked-cell-suppression"):
        assert child.is_candidate(task, argv, kwargs)
        assert science_isolation.is_candidate(task, argv, kwargs)


@pytest.mark.parametrize("task", ["linked-cell-suppression", "tess-transit-vetting"])
def test_bundle_contains_shared_child_adapter(task: str, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    science_isolation.create_bundle(task, bundle)
    assert (bundle / "science_task_child_isolation.py").is_file()


def test_linked_cell_builds_readonly_case_and_bounded_output_mounts(tmp_path: Path) -> None:
    argv, kwargs, root = _linked_fixture(tmp_path)
    launch = child.prepare_linked_cell(
        "terminal-bench-science/linked-cell-suppression", argv,
        environment=kwargs["env"], cwd=kwargs["cwd"], preexec_fn=kwargs["preexec_fn"],
        bwrap="/usr/bin/bwrap", source_root=root, system_paths=(),
    )
    command = launch.command
    assert "--unshare-net" in command and "--unshare-pid" in command
    assert command[command.index("--uid") + 1] == "65534"
    assert command[command.index("--ro-bind") + 2] == argv[1]
    case_index = command.index(str(root / "tmp/linked-cell-verifier-unit_001/case"))
    assert command[case_index - 1] == "--ro-bind"
    output_index = command.index(str(root / "tmp/linked-cell-verifier-unit_001/output"))
    assert command[output_index - 1] == "--bind"
    assert command[command.index("--chdir") + 1] == kwargs["cwd"]
    script = command[command.index("-c") + 1]
    for limit in ("1800,1800", "7516192768,7516192768", "8388608,8388608", "256,256"):
        assert limit in script


def test_linked_cell_rejects_nonexact_commands_and_preexec(tmp_path: Path) -> None:
    argv, kwargs, _ = _linked_fixture(tmp_path)
    assert not child.is_candidate("linked-cell-suppression", [*argv, "--extra"], kwargs)
    assert not child.is_candidate("linked-cell-suppression", argv, {**kwargs, "preexec_fn": None})
    assert not child.is_candidate("linked-cell-suppression", argv,
                                  {**kwargs, "preexec_fn": lambda: None})


def _tess_fixture(root: Path, uid: int = 65531) -> tuple[list[str], dict[str, object], Path]:
    app = root / "app/vetter"
    app.mkdir(parents=True)
    (app / "vet.py").write_text("print('candidate')\n")
    logical_root = Path("/run/vetter-runs/invocation-unit_001")
    run_root = root / logical_root.relative_to("/")
    input_dir, output_dir, home, temp, cache = (run_root / x for x in ("input", "output", "home", "tmp", "cache"))
    for path in (input_dir, output_dir, home, temp, cache):
        path.mkdir(parents=True)
    manifest = input_dir / "manifest_0123456789.json"
    manifest.write_text('{"targets": []}\n')
    output = output_dir / "report_0123456789.json"
    cwd = str(logical_root)
    argv = [sys.executable, "-I", "/app/vetter/vet.py", "--manifest",
            str(logical_root / "input" / manifest.name), "--output",
            str(logical_root / "output" / output.name)]
    def preexec_fn() -> None:
        _candidate_preexec(uid)
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "preexec_fn": preexec_fn,
        "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(logical_root / "home"),
                "TMPDIR": str(logical_root / "tmp"), "XDG_CACHE_HOME": str(logical_root / "cache"),
                "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1"},
    }
    return argv, kwargs, root


@pytest.mark.parametrize("task", ["tess-transit-vetting", "terminal-bench-science/tess-transit-vetting"])
def test_tess_matches_exact_child_and_uid_closure(task: str, tmp_path: Path) -> None:
    argv, kwargs, _ = _tess_fixture(tmp_path)
    assert child.is_candidate(task, argv, kwargs)
    assert science_isolation.is_candidate(task, argv, kwargs)


def test_tess_preserves_only_packet_and_declared_writable_directories(tmp_path: Path) -> None:
    argv, kwargs, root = _tess_fixture(tmp_path, uid=65537)
    launch = child.prepare_tess(
        "terminal-bench-science/tess-transit-vetting", argv,
        environment=kwargs["env"], cwd=kwargs["cwd"], preexec_fn=kwargs["preexec_fn"],
        bwrap="/usr/bin/bwrap", source_root=root, system_paths=(),
    )
    command = launch.command
    assert "--unshare-net" in command and "--unshare-ipc" in command
    assert command[command.index("--uid") + 1] == "65537"
    ro_targets = {command[i + 2] for i, value in enumerate(command[:-2]) if value == "--ro-bind"}
    rw_targets = {command[i + 2] for i, value in enumerate(command[:-2]) if value == "--bind"}
    for path in ("/app/vetter/vet.py", kwargs["cwd"] + "/input"):
        assert path in ro_targets
    for path in (kwargs["cwd"] + "/output", kwargs["cwd"] + "/home",
                 kwargs["cwd"] + "/tmp", kwargs["cwd"] + "/cache"):
        assert path in rw_targets
    assert command[command.index("--chdir") + 1] == kwargs["cwd"]
    script = command[command.index("-c") + 1]
    for limit in ("150,150", "3221225472,3221225472", "16777216,16777216", "128,128", "64,64"):
        assert limit in script
    assert launch.environment["HOME"] == kwargs["cwd"] + "/home"


def test_tess_rejects_helper_commands_and_unknown_uid(tmp_path: Path) -> None:
    argv, kwargs, _ = _tess_fixture(tmp_path)
    assert not child.is_candidate("tess-transit-vetting", [*argv, "extra"], kwargs)
    bad = dict(kwargs)
    def unsupported() -> None:
        return None
    bad["preexec_fn"] = unsupported
    assert not child.is_candidate("tess-transit-vetting", argv, bad)

