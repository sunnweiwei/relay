"""Model-free checks for the genomic-model-ranking predictor boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_genomic_isolation as isolation


def _argv(root: str) -> list[str]:
    return [
        "python3", "-I", root + "/predict.py", "--input-dir", root + "/input",
        "--output", root + "/output/predictions.csv", "--ranking-output",
        root + "/output/model_ranking.json",
    ]


def test_exact_predictor_match_accepts_bare_and_canonical_task_ids() -> None:
    root = "/tmp/genomic-ranking-run-fixture"
    argv = _argv(root)
    kwargs = {"cwd": root, "preexec_fn": lambda: None, "stdin": -3, "close_fds": True}
    assert isolation.is_candidate("genomic-model-ranking", argv, kwargs)
    assert isolation.is_candidate(isolation.CANONICAL_TASK, argv, kwargs)
    assert not isolation.is_candidate("other-benchmark/genomic-model-ranking", argv, kwargs)
    assert not isolation.is_candidate("genomic-model-ranking", argv, {"cwd": root})
    assert not isolation.is_candidate("genomic-model-ranking", _argv("/tmp/unrelated"), kwargs)
    assert not isolation.is_candidate("genomic-model-ranking", [*argv, "extra"], kwargs)
    assert not isolation.is_candidate("genomic-model-ranking", ["/usr/bin/python3", *argv[1:]], kwargs)


def test_command_binds_only_exact_public_run_tree_and_applies_limits_after_bwrap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "root"
    root = source / "tmp/genomic-ranking-run-fixture"
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "home").mkdir()
    (root / "predict.py").write_text("pass\n")
    for name in ("examples.csv", "candidate_scores.csv", "baseline_evaluation.csv"):
        (root / "input" / name).write_text("public\n")

    launch = isolation.build_candidate_command(
        task_id=isolation.CANONICAL_TASK,
        argv=_argv("/tmp/genomic-ranking-run-fixture"),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/old"},
        cwd="/tmp/genomic-ranking-run-fixture", uid=65534, gid=65534,
        bwrap="/usr/bin/bwrap", source_root=source,
    )
    command = launch.command
    try:
        assert "--unshare-net" in command and "--unshare-pid" in command
        assert "--uid" in command and command[command.index("--uid") + 1] == "65534"
        assert "--gid" in command and command[command.index("--gid") + 1] == "65534"
        separator = command.index("--")
        assert command[separator:] == [
            "--", "python3", "-I", "/tmp/relay-genomic-exec.py", "-I",
            "/tmp/genomic-ranking-run-fixture/predict.py", "--input-dir",
            "/tmp/genomic-ranking-run-fixture/input", "--output",
            "/tmp/genomic-ranking-run-fixture/output/predictions.csv",
            "--ranking-output",
            "/tmp/genomic-ranking-run-fixture/output/model_ranking.json",
        ]
        assert "--ro-bind-data" in command
        assert "--bind" in command
        assert str(root) in command
        assert launch.environment["HOME"] == "/tmp/old"
        dispatcher = launch._dispatcher
        dispatcher.seek(0)
        code = dispatcher.read().decode()
        assert "RLIMIT_CPU, (40, 40)" in code
        assert "RLIMIT_AS, (3200000000, 3200000000)" in code
        assert "RLIMIT_NOFILE, (64, 64)" in code
        assert "RLIMIT_NPROC, (64, 64)" in code
        assert "sys.argv[3:]" in code
    finally:
        launch.close()


def test_bwrap_predictor_sees_public_packet_only_and_writes_declared_outputs(
    tmp_path: Path,
) -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable")
    source = tmp_path / "root"
    root = source / "tmp/genomic-ranking-run-fixture"
    (root / "input").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "home").mkdir()
    hidden = source / "tests/private"
    hidden.parent.mkdir()
    hidden.write_text("verifier-only")
    for name in ("examples.csv", "candidate_scores.csv", "baseline_evaluation.csv"):
        (root / "input" / name).write_text("public\n")
    (root / "predict.py").write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "root=Path(sys.argv[sys.argv.index('--input-dir')]).parent\n"
        "blocked=not Path('/tests/private').exists() and not Path('/proc/1/root/tests/private').exists()\n"
        "public=sorted(p.name for p in (root/'input').iterdir())\n"
        "(root/'output/predictions.csv').write_text('ok\\n')\n"
        "(root/'output/model_ranking.json').write_text(json.dumps({'blocked':blocked,'public':public}))\n"
    )
    (root / "predict.py").chmod(0o555)
    launch = isolation.build_candidate_command(
        task_id=isolation.TASK_SLUG,
        argv=_argv("/tmp/genomic-ranking-run-fixture"),
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/genomic-ranking-run-fixture/home",
                     "TMPDIR": "/tmp/genomic-ranking-run-fixture/home"},
        cwd="/tmp/genomic-ranking-run-fixture", uid=65534, gid=65534,
        bwrap=bwrap, source_root=source,
    )
    try:
        result = subprocess.run(
            launch.command, env=launch.environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=15, pass_fds=launch.pass_fds,
        )
    finally:
        launch.close()
    assert result.returncode == 0, result.stderr
    assert (root / "output/predictions.csv").read_text() == "ok\n"
    observed = json.loads((root / "output/model_ranking.json").read_text())
    assert observed == {
        "blocked": True,
        "public": ["baseline_evaluation.csv", "candidate_scores.csv", "examples.csv"],
    }
