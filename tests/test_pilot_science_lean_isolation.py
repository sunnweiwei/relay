"""Model-free bwrap checks for the audited Science Lean compiler adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from evals.pilot import science_lean_isolation as isolation


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "tmp").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "logs/verifier").mkdir(parents=True)
    (tmp_path / "root/.elan/bin").mkdir(parents=True)
    (tmp_path / "root/.elan/toolchains/lean/bin").mkdir(parents=True)
    app = tmp_path / "app"
    (app / "FiniteFree").mkdir(parents=True)
    package = app / ".lake/packages/fixture-package"
    package.mkdir(parents=True)
    (package / "source.txt").write_text("pinned dependency source\n")
    (app / "lakefile.toml").write_text("name = 'fixture'\n")
    (app / "FiniteFree/Goal.lean").write_text("theorem fixture : True := True.intro\n")
    lake = tmp_path / "root/.elan/bin/lake"
    lake.write_text("#!/bin/sh\nset -eu\ntest \"$PWD\" = /workspace\ntest ! -e /tests\ntest ! -e /root/private\nmkdir -p .lake/build\necho ok > .lake/build/marker\necho private-cache > .lake/packages/fixture-package/cache-hash\nprintf 'lean-ok\\n'\n")
    lake.chmod(0o755)
    return tmp_path


def launch(root: Path, argv: list[str] | tuple[str, ...] = ("lake", "build"), cwd: str = "/app"):
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is unavailable")
    launch = isolation.prepare_candidate(
        "terminal-bench-science/finite-free-stam", argv,
        environment={"PATH": "/root/.elan/bin:/usr/bin"}, cwd=cwd,
        bwrap=shutil.which("bwrap") or isolation.BWRAP_PATH, source_root=root,
    )
    return launch


def test_lean_child_hides_tests_and_persists_only_private_build_state(fixture_root: Path) -> None:
    project = fixture_root / "app"
    launch1 = launch(fixture_root)
    assert launch1.stage.parent.parent == fixture_root / "logs"
    assert launch1.canary_result["passed"] is True
    canary_path = fixture_root / "logs/verifier/lean-isolation-canary.json"
    assert '"passed": true' in canary_path.read_text()
    result = subprocess.run(launch1.command, env=launch1.environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "lean-ok\n"
    assert (launch1.stage / ".lake/build/marker").read_text() == "ok\n"
    assert (launch1.stage / ".lake/packages/fixture-package/cache-hash").read_text() == "private-cache\n"
    assert not (project / ".lake/packages/fixture-package/cache-hash").exists()
    assert (project / ".lake/packages/fixture-package/source.txt").read_text() == "pinned dependency source\n"
    assert not (project / ".lake/build/marker").exists()

    launch2 = launch(fixture_root, ("lake", "env", "lean", "Tests/Check.lean"))
    assert launch1.stage == launch2.stage
    result2 = subprocess.run(launch2.command, env=launch2.environment, capture_output=True, text=True, timeout=30)
    assert result2.returncode == 0, result2.stderr
    assert (launch2.stage / ".lake/build/marker").read_text() == "ok\n"


def test_dependency_writes_stay_in_private_stage(fixture_root: Path) -> None:
    launch1 = launch(fixture_root)
    assert launch1.stage.parent.parent == fixture_root / "logs"
    assert launch1.canary_result["passed"] is True
    package_stage = str(launch1.stage / ".lake/packages")
    assert not any(launch1.command[index] == "--ro-bind" and
                   launch1.command[index + 1] == package_stage
                   for index in range(len(launch1.command) - 1))
    result = subprocess.run(launch1.command, env=launch1.environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (launch1.stage / ".lake/packages/fixture-package/cache-hash").exists()
    assert not (fixture_root / "app/.lake/packages/fixture-package/cache-hash").exists()


def test_task_slug_from_relay_environment_is_accepted() -> None:
    for slug in ("finite-free-stam", "onsager-ising-lean", "gen-turan-paths"):
        assert isolation._slug(slug) == slug
        assert isolation._slug("terminal-bench-science/" + slug) == slug
    with pytest.raises(ValueError, match="No reviewed"):
        isolation._slug("unknown-task")


def test_only_reviewed_lean_task_roots_and_lake_commands_are_accepted(fixture_root: Path) -> None:
    assert not isolation.is_candidate("terminal-bench-science/finite-free-stam", ["python3", "x.py"], {"cwd": "/app"})
    assert not isolation.is_candidate("terminal-bench-science/finite-free-stam", ["lake", "build"], {"cwd": "/tests"})
    assert not isolation.is_candidate("terminal-bench-science/onsager-ising-lean", ["lake", "build"], {"cwd": "/tmp/not-onsager"})
    assert not isolation.is_candidate("terminal-bench-science/gen-turan-paths", ["lake", "build"], {"cwd": "/root/project/../tests"})


def test_helper_does_not_add_restrictions_on_officially_allowed_syntax(fixture_root: Path) -> None:
    source = fixture_root / "app/FiniteFree/Goal.lean"
    source.write_text("local notation \"T\" => True\ntheorem fixture : T := by trivial\n")
    assert launch(fixture_root).source_digest


def test_candidate_source_cannot_change_between_verifier_phases(fixture_root: Path) -> None:
    launch(fixture_root)
    source = fixture_root / "app/FiniteFree/Goal.lean"
    source.write_text("theorem fixture : True := by trivial\n")
    with pytest.raises(RuntimeError, match="source changed"):
        launch(fixture_root)


def test_gen_turan_and_onsager_cwd_profiles_are_exact() -> None:
    assert isolation.is_candidate("terminal-bench-science/gen-turan-paths", ["lake", "build"], {"cwd": "/root/project"})
    assert isolation.is_candidate("terminal-bench-science/onsager-ising-lean", ["lake", "build"], {"cwd": "/tmp/onsager_golden_abc123"})


@pytest.mark.parametrize("slug,filename,checks", [
    ("finite-free-stam", "test_outputs.py", ("test_phase6_no_metaprogramming", "test_phase7_no_axiom_declarations", "test_phase8_no_kernel_bypass")),
    ("onsager-ising-lean", "test_state.py", ("test_no_banned_exploit_syntax", "test_no_sorry_source_grep")),
    ("gen-turan-paths", "test_verify.py", ("test_no_banned_exploit_syntax",)),
])
def test_preflight_delegates_only_to_official_source_checks(
    slug: str, filename: str, checks: tuple[str, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    module = types.SimpleNamespace(__file__=f"/tests/{filename}")
    for check in checks:
        setattr(module, check, lambda name=check: called.append(name))
    monkeypatch.setitem(sys.modules, f"lean_adapter_{slug}", module)
    isolation._official_source_checks(slug)
    assert called == list(checks)
