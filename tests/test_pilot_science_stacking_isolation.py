from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from evals.pilot import science_isolation, science_stacking_isolation as stacking


def _candidate_fixture(root: Path) -> tuple[list[str], dict[str, object], Path]:
    workspace = root / "workspace"
    submission = workspace / "stacking_submission"
    packet = workspace / "stacking_packet_unit001" / "data"
    results = workspace / "stacking_case_results"
    home = workspace / "stacking_solver_home"
    for path in (submission, packet, results, home / "tmp", home / "cache"):
        path.mkdir(parents=True, exist_ok=True)
    (submission / "solve_stacking.py").write_text("print('candidate')\n")
    (packet / "inverse_parameters.json").write_text("{}")
    (packet / "scattering_counts.npz").write_bytes(b"public fixture packet")
    argv = [sys.executable, "/workspace/stacking_submission/solve_stacking.py",
            "--data-dir", "/workspace/stacking_packet_unit001/data", "--output",
            "/workspace/stacking_case_results/trajectory.csv"]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/workspace/stacking_solver_home",
           "TMPDIR": "/workspace/stacking_solver_home/tmp",
           "XDG_CACHE_HOME": "/workspace/stacking_solver_home/cache"}
    kwargs: dict[str, object] = {
        "cwd": "/workspace/stacking_submission", "env": env,
        "user": 65534, "group": 65534, "extra_groups": (), "start_new_session": True,
        "text": True, "stdout": -1, "stderr": -1,
    }
    return argv, kwargs, root


def test_exact_candidate_launch_for_bare_and_canonical_task_ids(tmp_path: Path) -> None:
    argv, kwargs, _ = _candidate_fixture(tmp_path)
    for task_id in ("stacking-disorder-diffraction",
                    "terminal-bench-science/stacking-disorder-diffraction"):
        assert stacking.is_candidate(task_id, argv, kwargs)
        assert science_isolation.is_candidate(task_id, argv, kwargs)


def test_candidate_boundary_keeps_submission_and_packet_readonly(tmp_path: Path) -> None:
    argv, kwargs, root = _candidate_fixture(tmp_path)
    launch = stacking.prepare_candidate(
        "terminal-bench-science/stacking-disorder-diffraction", argv,
        environment=kwargs["env"], cwd=kwargs["cwd"], source_root=root,
        bwrap="/usr/bin/bwrap", system_paths=(),
    )
    cmd = launch.command
    assert "--unshare-net" in cmd and "--unshare-pid" in cmd and "--new-session" not in cmd
    assert cmd[cmd.index("--uid") + 1] == "65534"
    ro_targets = {cmd[i + 2] for i, value in enumerate(cmd[:-2]) if value == "--ro-bind"}
    rw_targets = {cmd[i + 2] for i, value in enumerate(cmd[:-2]) if value == "--bind"}
    assert "/workspace/stacking_submission" in ro_targets
    assert "/workspace/stacking_packet_unit001/data" in ro_targets
    assert "/workspace/stacking_case_results" in rw_targets
    assert "/workspace/stacking_solver_home" in rw_targets
    assert cmd[cmd.index("--chdir") + 1] == "/workspace/stacking_submission"
    assert cmd[cmd.index("--") + 1:] == argv
    workspace_mask = cmd.index("/workspace", cmd.index("--tmpfs"))
    candidate_mount = cmd.index("/workspace/stacking_submission", cmd.index("--ro-bind"))
    assert cmd[workspace_mask - 1] == "--tmpfs"
    assert workspace_mask < candidate_mount


def test_candidate_rejects_wrong_uid_path_env_and_group_policy(tmp_path: Path) -> None:
    argv, kwargs, _ = _candidate_fixture(tmp_path)
    assert not stacking.is_candidate("stacking-disorder-diffraction", [*argv, "extra"], kwargs)
    assert not stacking.is_candidate("stacking-disorder-diffraction", argv,
                                     {**kwargs, "user": 1000})
    assert not stacking.is_candidate("stacking-disorder-diffraction", argv,
                                     {**kwargs, "extra_groups": (1000,)})
    env = dict(kwargs["env"])
    env["PYTHONPATH"] = "/secret"
    assert not stacking.is_candidate("stacking-disorder-diffraction", argv,
                                     {**kwargs, "env": env})


def test_reference_stays_trusted_only_for_pinned_copy_hash(tmp_path: Path, monkeypatch) -> None:
    solver = tmp_path / "workspace/stacking_reference_case001/solver/solve_stacking.py"
    solver.parent.mkdir(parents=True)
    solver.write_bytes(b"synthetic public unit fixture")
    packet = "/workspace/stacking_packet_unit002/data"
    root = "/workspace/stacking_reference_case001"
    argv = [sys.executable, "/workspace/stacking_reference_case001/solver/solve_stacking.py",
            "--data-dir", packet, "--output", root + "/trajectory.csv"]
    env = {"HOME": root + "/home", "TMPDIR": root + "/home/tmp",
           "XDG_CACHE_HOME": root + "/home/cache"}
    kwargs = {"cwd": root + "/solver", "env": env, "user": 65534,
              "group": 65534, "extra_groups": (), "start_new_session": True,
              "text": True, "stdout": -1, "stderr": -1}
    monkeypatch.setattr(stacking, "REFERENCE_SHA256", hashlib.sha256(solver.read_bytes()).hexdigest())
    assert stacking.is_verified_reference("terminal-bench-science/stacking-disorder-diffraction",
                                          argv, kwargs, source_root=tmp_path)
    with monkeypatch.context() as reference_verified:
        reference_verified.setattr(stacking, "is_verified_reference", lambda *args, **kwargs: True)
        assert science_isolation.is_candidate(
            "terminal-bench-science/stacking-disorder-diffraction", argv, kwargs)
    reference_root = solver.parent.parent
    for path in (reference_root / "home/tmp", reference_root / "home/cache"):
        path.mkdir(parents=True, exist_ok=True)
    packet_source = tmp_path / packet.lstrip("/")
    packet_source.mkdir(parents=True)
    (packet_source / "inverse_parameters.json").write_text("{}")
    (packet_source / "scattering_counts.npz").write_bytes(b"public packet")
    with monkeypatch.context() as reference_verified:
        reference_verified.setattr(stacking, "REFERENCE_SHA256", hashlib.sha256(solver.read_bytes()).hexdigest())
        launch = stacking.prepare_reference(
            "terminal-bench-science/stacking-disorder-diffraction", argv,
            environment=env, cwd=kwargs["cwd"], source_root=tmp_path,
            bwrap="/usr/bin/bwrap", system_paths=())
    cmd = launch.command
    assert "--unshare-net" in cmd and "--unshare-pid" in cmd
    assert cmd[cmd.index("--uid") + 1] == "65534"
    assert cmd[cmd.index("--") + 1:] == argv
    assert cmd[cmd.index("--chdir") + 1] == "/workspace/stacking_reference_case001/solver"
    ro_targets = {cmd[i + 2] for i, value in enumerate(cmd[:-2]) if value == "--ro-bind"}
    rw_targets = {cmd[i + 2] for i, value in enumerate(cmd[:-2]) if value == "--bind"}
    assert "/workspace/stacking_reference_case001/solver" in ro_targets
    assert "/workspace/stacking_packet_unit002/data" in ro_targets
    assert "/workspace/stacking_reference_case001" in rw_targets
    solver.write_bytes(b"modified content")
    assert not stacking.is_verified_reference("stacking-disorder-diffraction", argv, kwargs,
                                              source_root=tmp_path)
    with pytest.raises(ValueError, match="pinned source hash"):
        science_isolation.is_candidate("stacking-disorder-diffraction", argv, kwargs)


def test_bundle_carries_stacking_matcher(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    science_isolation.create_bundle("stacking-disorder-diffraction", bundle)
    assert (bundle / "science_stacking_isolation.py").is_file()

