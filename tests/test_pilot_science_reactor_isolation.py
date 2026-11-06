from __future__ import annotations

from pathlib import Path
import sys

import pytest

from evals.pilot import science_isolation, science_reactor_isolation as reactor


def _fixture_root(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "container-root"
    submission = root / "app/submission"
    submission.mkdir(parents=True)
    (submission / "controller.py").write_text("class Controller: pass\n")
    scratch = root / "tmp/reactor-controller-test_123"
    scratch.mkdir(parents=True)
    return root, "/tmp/reactor-controller-test_123"


def _argv() -> list[str]:
    return [sys.executable, reactor.CONTROLLER_HOST, reactor.SUBMISSION_DIR]


@pytest.mark.parametrize("task_id", [reactor.TASK_SLUG, reactor.TASK_ID])
def test_matches_exact_reactor_host_command_for_both_task_id_forms(task_id: str) -> None:
    assert reactor.is_candidate(task_id, _argv(), {})
    assert science_isolation.is_candidate(task_id, _argv(), {})
    assert science_isolation.task_slug(task_id) == reactor.TASK_SLUG


@pytest.mark.parametrize("task_id", [reactor.TASK_SLUG, reactor.TASK_ID])
def test_bundle_contains_adapter_and_uses_canonical_task_id(tmp_path: Path, task_id: str) -> None:
    metadata = science_isolation.create_bundle(task_id, tmp_path / "bundle")
    assert (tmp_path / "bundle/science_reactor_isolation.py").is_file()
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == reactor.TASK_ID


def test_does_not_match_other_tasks_or_nonexact_commands() -> None:
    assert not reactor.is_candidate("other-task", _argv(), {})
    assert not reactor.is_candidate(reactor.TASK_ID, [*_argv(), "extra"], {})
    assert not reactor.is_candidate(reactor.TASK_ID, [sys.executable, "/tests/helper.py", "/app/submission"], {})


def test_prepare_preserves_controller_protocol_and_adds_minimal_boundary(tmp_path: Path) -> None:
    root, cwd = _fixture_root(tmp_path)
    launch = reactor.prepare_candidate(
        reactor.TASK_ID,
        _argv(),
        environment={"PATH": "/usr/bin:/opt/relay-isolation/bin", "PYTHONPATH": "/opt/relay-isolation",
                     "RELAY_SCIENCE_TASK_ID": reactor.TASK_ID, "TMPDIR": "/tmp/old"},
        cwd=cwd,
        uid=65534,
        gid=65534,
        bwrap="/usr/bin/bwrap",
        source_root=root,
        system_paths=(),
    )
    command = launch.command
    assert command[:4] == ["/usr/bin/bwrap", "--unshare-user", "--uid", "65534"]
    assert "--unshare-net" in command and "--unshare-pid" in command
    assert "--tmpfs" in command
    assert "/tests" in command and "/logs" in command and "/opt" in command
    assert command[command.index("--ro-bind") + 1] == str(root / "app/submission")
    assert command[command.index("--ro-bind") + 2] == "/app/submission"
    assert "--remount-ro" in command
    assert command[-7:] == [sys.executable, "-c", reactor._BOOTSTRAP, cwd, *_argv()]
    assert launch.environment["TMPDIR"] == cwd
    assert launch.environment["TMP"] == cwd
    assert launch.environment["TEMP"] == cwd
    assert "RELAY_SCIENCE_TASK_ID" not in launch.environment
    assert launch.environment["PATH"] == "/usr/bin"
    assert "PYTHONPATH" not in launch.environment


@pytest.mark.parametrize("cwd", [None, "/tmp/elsewhere", "/tmp/reactor-controller-../escape"])
def test_rejects_missing_or_untrusted_scratch_cwd(tmp_path: Path, cwd: str | None) -> None:
    root, _ = _fixture_root(tmp_path)
    with pytest.raises(reactor.ReactorIsolationError):
        reactor.prepare_candidate(reactor.TASK_SLUG, _argv(), environment={}, cwd=cwd,
                                  source_root=root, system_paths=())


def test_rejects_missing_submission_directory(tmp_path: Path) -> None:
    root = tmp_path / "container-root"
    (root / "tmp/reactor-controller-test_123").mkdir(parents=True)
    with pytest.raises(reactor.ReactorIsolationError, match="submitted controller directory"):
        reactor.prepare_candidate(reactor.TASK_ID, _argv(), environment={},
                                  cwd="/tmp/reactor-controller-test_123", source_root=root,
                                  system_paths=())
