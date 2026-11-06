"""Offline checks for the regularized-game-proof compiler boundary."""

from __future__ import annotations

import shutil
import sys
import subprocess
import os
from pathlib import Path

import pytest

from evals.pilot import science_regularized_lean_isolation as isolation
from evals.pilot.science_isolation import create_bundle


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "tmp").mkdir()
    project = tmp_path / "verifier/GameProof"
    (project / "GameProof").mkdir(parents=True)
    (project / ".lake/packages").mkdir(parents=True)
    (project / ".lake/build/lib/lean").mkdir(parents=True)
    (project / ".lake/build/lib/lean/VerifierSpec.olean").write_bytes(b"trusted-cache")
    (tmp_path / "root/.elan/bin").mkdir(parents=True)
    (tmp_path / "root/.elan/bin/lake").write_text(
        "#!/bin/sh\nset -eu\ntest \"$PWD\" = /workspace\ntest ! -e /tests\n"
        "test ! -e /usr/project\ntest -f GameProof/Basic.lean\n"
        "case \"$*\" in 'build --wfail') test ! -e HiddenCheck.lean;; "
        "'env lean HiddenCheck.lean') test -f HiddenCheck.lean;; esac\n"
        "mkdir -p .lake/build\nprintf ok > .lake/build/marker\n"
    )
    (tmp_path / "root/.elan/bin/lake").chmod(0o755)
    (project / "GameProof/Basic.lean").write_text(
        "import GameProof.Definitions\ntheorem submitted : True := by trivial\n"
    )
    (project / "HiddenCheck.lean").write_text("-- private check fixture\n")
    (project / "AxiomCheck.lean").write_text("-- private axiom fixture\n")
    (project / "lakefile.toml").write_text('name = "GameProof"\n')
    return tmp_path


def _launch(root: Path, args: tuple[str, ...]):
    return isolation.prepare_candidate(
        "terminal-bench-science/regularized-game-proof", ("lake", *args),
        environment={"PATH": "/root/.elan/bin:/usr/bin"}, cwd="/verifier/GameProof",
        bwrap=shutil.which("bwrap") or isolation.BWRAP_PATH, source_root=root,
    )


def test_exact_official_lake_invocations_are_routed() -> None:
    task = "terminal-bench-science/regularized-game-proof"
    cwd = {"cwd": "/verifier/GameProof"}
    assert isolation.is_candidate(task, ["lake", "build", "--wfail"], cwd)
    assert isolation.is_candidate(task, ["/root/.elan/bin/lake", "env", "lean", "HiddenCheck.lean"], cwd)
    assert isolation.is_candidate(task, ["lake", "env", "lean", "AxiomCheck.lean"], cwd)
    assert not isolation.is_candidate(task, ["lake", "build"], cwd)
    assert not isolation.is_candidate(task, ["lake", "env", "lean", "Other.lean"], cwd)
    assert not isolation.is_candidate(task, ["lake", "build", "--wfail"], {"cwd": "/tests"})
    assert not isolation.is_candidate("terminal-bench-science/finite-free-stam",
                                     ["lake", "build", "--wfail"], cwd)


def test_generated_bundle_shadows_bare_lake_with_isolated_launcher(tmp_path: Path) -> None:
    bundle = create_bundle("terminal-bench-science/regularized-game-proof", tmp_path / "bundle")
    launcher = tmp_path / "bundle/bin/lake"
    assert launcher.is_file() and launcher.stat().st_mode & 0o111
    assert "science_regularized_lean_isolation.py" in {
        path.name for path in (tmp_path / "bundle").iterdir()
    }
    assert bundle["environment"]["PATH"].split(":")[:2] == [
        "/opt/relay-isolation/bin", "/root/.elan/bin",
    ]
    assert "lake_wrapper_main" in launcher.read_text()
    python = Path(sys.executable)
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "bundle"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    imported = subprocess.run(
        [str(python), "-c", "import science_isolation; import science_regularized_lean_isolation"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15,
    )
    assert imported.returncode == 0, imported.stderr


def test_hidden_checks_are_omitted_from_build_and_added_only_for_official_check(
    fixture_root: Path,
) -> None:
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    build_cmd, build_env, build_meta = _launch(fixture_root, ("build", "--wfail"))
    assert build_meta["kind"] == "build"
    assert not (Path(build_meta["stage"]) / "HiddenCheck.lean").exists()
    assert (Path(build_meta["stage"]) / ".lake/build/lib/lean/VerifierSpec.olean").read_bytes() == b"trusted-cache"
    built = subprocess.run(build_cmd, env=build_env, text=True, capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr

    check_cmd, check_env, check_meta = _launch(fixture_root, ("env", "lean", "HiddenCheck.lean"))
    assert check_meta["kind"] == "HiddenCheck.lean"
    assert (Path(check_meta["stage"]) / "HiddenCheck.lean").is_file()
    assert (Path(check_meta["stage"]) / ".lake/build/lib/lean/VerifierSpec.olean").read_bytes() == b"trusted-cache"
    checked = subprocess.run(check_cmd, env=check_env, text=True, capture_output=True, timeout=30)
    assert checked.returncode == 0, checked.stderr


def test_source_change_between_official_compiler_calls_is_rejected(fixture_root: Path) -> None:
    _launch(fixture_root, ("build", "--wfail"))
    source = fixture_root / "verifier/GameProof/GameProof/Basic.lean"
    source.write_text("import GameProof.Definitions\ntheorem submitted : True := True.intro\n")
    with pytest.raises(RuntimeError, match="project changed"):
        _launch(fixture_root, ("env", "lean", "HiddenCheck.lean"))

