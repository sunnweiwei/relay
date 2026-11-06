"""Model-free checks for the official Science subprocess boundaries."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.pilot import science_isolation as isolation


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    for path in ("tests", "logs/verifier", "app/solver", "app/sdk", "app/waveguide_submission",
                 "app/waveguide_solver_home", "app/waveguide_case_results", "app/waveguide_packet_1/data",
                 "app/waveguide_ref_results", "tmp/hbv-optimizer", "dev/tbs-solver-scratch/home", "results"):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    for path in ("tests/private", "logs/verifier/reward.txt", "app/waveguide_ref_results/private"):
        (tmp_path / path).write_text("verifier-only fixture\n")
    for path in ("app/protocol.md", "app/sdk/oracle_client.py", "app/regressor.py", "app/training_data.npz",
                 "tests/_sr_runner.py", "tests/run_optimizer.R", "results/optimizer.R", "tmp/input.npz", "tmp/output.npz"):
        (tmp_path / path).touch()
    return tmp_path


def run_candidate(root: Path, task: str, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable")
    command, environment = isolation.candidate_command(
        task, argv, source_root=root, bwrap=bwrap,
        environment={"PATH": "/usr/bin:/bin", "HOME": "/dev/tbs-solver-scratch/home"},
        **kwargs,
    )
    return subprocess.run(command, env=environment, text=True, capture_output=True, timeout=15)


def test_all_candidate_entrypoints_and_trusted_reference() -> None:
    assert isolation.is_candidate("inverse-waveguide-shape", ["python", "solve.py"], {"user": 65534})
    assert isolation.is_candidate("localized-sspd-solver", ["bash", "-c", "..."], {"group": 60001})
    assert isolation.is_candidate("symbolic-regression", ["python", "/tests/_sr_runner.py", "x", "out"], {})
    assert not isolation.is_candidate("inverse-waveguide-shape", ["python", "/tests/reference_submission/solve_waveguide.py"], {})
    assert not isolation.is_candidate("symbolic-regression", ["python", "/tests/test_sr.py"], {})


def test_public_bridge_private_paths_and_parent_process_are_isolated(fixture_root: Path) -> None:
    probe = f"""
import json, os
from pathlib import Path
blocked = []
for path in ['/tests/private', '/logs/verifier/reward.txt', '/app/waveguide_ref_results/private',
             '/proc/{os.getpid()}/root/tests/private', {str(fixture_root / 'tests/private')!r}]:
    try:
        Path(path).read_text()
    except (OSError, ValueError):
        blocked.append(path)
Path('/tmp/output.npz').write_text('candidate output')
print(json.dumps({{'blocked': len(blocked), 'public': Path('/tests/_sr_runner.py').exists(),
                  'uid': os.getuid(), 'pids': [p.name for p in Path('/proc').iterdir() if p.name.isdigit()]}}))
"""
    (fixture_root / "tests/_sr_runner.py").write_text(probe)
    result = run_candidate(fixture_root, "symbolic-regression", [
        "/usr/bin/python3", "/tests/_sr_runner.py", "/tmp/input.npz", "/tmp/output.npz"])
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["blocked"] == 5
    assert evidence["public"] is True
    assert evidence["uid"] == 60001
    assert len(evidence["pids"]) <= 3
    assert (fixture_root / "tmp/output.npz").read_text() == "candidate output"
    assert (fixture_root / "logs/verifier/reward.txt").read_text() == "verifier-only fixture\n"


def test_localized_scratch_limits_and_localhost(fixture_root: Path) -> None:
    import http.server
    import threading

    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    script = f"""
import json, socket
from pathlib import Path
blocked = []
for path in ['/tmp/untracked', '/var/tmp/untracked', '/run/lock/untracked', '/dev/untracked', '/dev/shm/untracked', '/app/solver/changed']:
    try:
        Path(path).write_text('must fail')
    except OSError:
        blocked.append(path)
Path('/dev/tbs-solver-scratch/home/probe').write_text('scratch')
with socket.create_connection(('127.0.0.1', {server.server_port}), timeout=2) as connection:
    connection.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n')
    response = connection.recv(256)
print(json.dumps({{'blocked': len(blocked), 'localhost': response.startswith(b'HTTP/'),
                  'dev_tmpfs': ' - tmpfs ' in next(line for line in Path('/proc/self/mountinfo').read_text().splitlines()
                                                  if line.split()[4] == '/dev')}}))
"""
    (fixture_root / "app/solver/probe.py").write_text(script)
    # This is the official four-process resource limit, not a raised substitute.
    try:
        result = run_candidate(fixture_root, "localized-sspd-solver", [
            "/bin/bash", "--noprofile", "--norc", "-c", "ulimit -u 4; exec /usr/bin/python3 /app/solver/probe.py"],
            cwd="/app/solver")
    finally:
        server.server_close()
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence == {"blocked": 6, "localhost": True, "dev_tmpfs": True}
    assert (fixture_root / "dev/tbs-solver-scratch/home/probe").read_text() == "scratch"


def test_waveguide_packet_is_readonly_and_reference_results_are_hidden(fixture_root: Path) -> None:
    packet = fixture_root / "app/waveguide_packet_1/data/input"
    packet.write_text("runtime packet")
    (fixture_root / "app/waveguide_submission/solve.py").write_text("""
from pathlib import Path
assert Path('/app/waveguide_packet_1/data/input').read_text() == 'runtime packet'
try:
    Path('/app/waveguide_packet_1/data/input').chmod(0o777)
except OSError:
    pass
else:
    raise AssertionError('packet chmod escaped read-only bind')
assert not Path('/app/waveguide_ref_results/private').exists()
Path('/app/waveguide_case_results/out.csv').write_text('result')
""")
    result = run_candidate(fixture_root, "inverse-waveguide-shape", [
        "/usr/bin/python3", "/app/waveguide_submission/solve.py", "--data-dir", "/app/waveguide_packet_1/data",
        "--output", "/app/waveguide_case_results/out.csv"], cwd="/app/waveguide_submission")
    assert result.returncode == 0, result.stderr
    assert (fixture_root / "app/waveguide_case_results/out.csv").read_text() == "result"
    assert packet.read_text() == "runtime packet"


def test_detached_candidate_does_not_survive_main_process(fixture_root: Path) -> None:
    (fixture_root / "app/solver/probe.py").write_text("""
import os, time
from pathlib import Path
child = os.fork()
if child == 0:
    os.setsid()
    os.close(1)
    os.close(2)
    time.sleep(0.5)
    Path('/dev/tbs-solver-scratch/home/survived').write_text('invalid')
    os._exit(0)
os._exit(0)
""")
    result = run_candidate(fixture_root, "localized-sspd-solver", ["/usr/bin/python3", "/app/solver/probe.py"])
    assert result.returncode == 0, result.stderr
    time.sleep(0.7)
    assert not (fixture_root / "dev/tbs-solver-scratch/home/survived").exists()


def test_hbv_bridge_and_input_output_paths(fixture_root: Path) -> None:
    (fixture_root / "tests/run_optimizer.R").write_text("""
test -r /results/optimizer.R || exit 2
test ! -e /tests/private || exit 3
printf done > /tmp/hbv-optimizer/params.rds
""")
    result = run_candidate(fixture_root, "hbv-calibration-1", [
        "/bin/sh", "/tests/run_optimizer.R", "/tmp/hbv-optimizer/input.rds", "/tmp/hbv-optimizer/params.rds"])
    assert result.returncode == 0, result.stderr
    assert (fixture_root / "tmp/hbv-optimizer/params.rds").read_text() == "done"


def test_popen_uid_drop_uses_the_real_boundary(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (fixture_root / "app/solver/probe.py").write_text("""
import os
from pathlib import Path
assert os.getuid() == 65534
assert not Path('/tests/private').exists()
print('candidate completed')
""")
    build = isolation.candidate_command

    def with_fixture(task_id, argv, **kwargs):
        return build(task_id, argv, source_root=fixture_root, bwrap=shutil.which("bwrap"), **kwargs)

    monkeypatch.setattr(isolation, "candidate_command", with_fixture)
    monkeypatch.setenv("RELAY_SCIENCE_TASK_ID", "localized-sspd-solver")
    monkeypatch.setattr(subprocess, "Popen", subprocess.Popen)
    isolation.install()
    result = subprocess.run(
        ["/usr/bin/python3", "/app/solver/probe.py"], user=65534, group=65534,
        extra_groups=(), env={"HOME": "/dev/tbs-solver-scratch/home", "PATH": "/usr/bin:/bin"},
        start_new_session=True, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "candidate completed"


def test_timeout_cleanup_reaps_namespace_before_official_communicate_retry(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (fixture_root / "app/solver/probe.py").write_text("""
import os, time
from pathlib import Path
if os.fork() == 0:
    os.setsid()
    time.sleep(1)
    Path('/dev/tbs-solver-scratch/home/survived-timeout').write_text('invalid')
    os._exit(0)
time.sleep(60)
""")
    build = isolation.candidate_command

    def with_fixture(task_id, argv, **kwargs):
        return build(task_id, argv, source_root=fixture_root, bwrap=shutil.which("bwrap"), **kwargs)

    monkeypatch.setattr(isolation, "candidate_command", with_fixture)
    monkeypatch.setenv("RELAY_SCIENCE_TASK_ID", "localized-sspd-solver")
    monkeypatch.setattr(subprocess, "Popen", subprocess.Popen)
    isolation.install()
    with subprocess.Popen(
        ["/usr/bin/python3", "/app/solver/probe.py"], user=60001, group=60001,
        env={"HOME": "/dev/tbs-solver-scratch/home", "PATH": "/usr/bin:/bin"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as process:
        with pytest.raises(subprocess.TimeoutExpired):
            process.communicate(timeout=0.2)
        # The official verifier retries communicate after its UID cleanup.
        process.communicate(timeout=2)
        assert process.poll() is not None
    time.sleep(1.1)
    assert not (fixture_root / "dev/tbs-solver-scratch/home/survived-timeout").exists()


def test_runuser_dispatches_the_original_argv(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (fixture_root / "tests/run_optimizer.R").write_text("""
test -r /results/optimizer.R || exit 2
test ! -e /tests/private || exit 3
printf done > /tmp/hbv-optimizer/params.rds
""")
    build = isolation.candidate_command
    calls = []

    def with_fixture(task_id, argv, **kwargs):
        kwargs["cwd"] = "/"
        return build(task_id, argv, source_root=fixture_root, bwrap=shutil.which("bwrap"), **kwargs)

    def exec_fixture(executable, command, environment):
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        calls.append(command)

    monkeypatch.setattr(isolation, "candidate_command", with_fixture)
    monkeypatch.setenv("RELAY_SCIENCE_TASK_ID", "hbv-calibration-1")
    monkeypatch.setattr(os, "execve", exec_fixture)
    isolation.runuser_main(["-u", "nobody", "--", "env", "HOME=/tmp/hbv-optimizer", "TMPDIR=/tmp/hbv-optimizer",
                           "/bin/sh", "/tests/run_optimizer.R", "/tmp/hbv-optimizer/input.rds", "/tmp/hbv-optimizer/params.rds"])
    assert len(calls) == 1
    assert (fixture_root / "tmp/hbv-optimizer/params.rds").read_text() == "done"


def test_trusted_adapter_environment_is_removed() -> None:
    environment = isolation.child_environment({
        "RELAY_SCIENCE_TASK_ID": "localized-sspd-solver", "FAKEROOTKEY": "fixture",
        "LD_PRELOAD": "libfakeroot.so", "APPTAINER_BIND": "private", "HOME": "/dev/scratch",
        "PYTHONPATH": "/opt/relay-isolation:/app/sdk", "PATH": "/opt/relay-isolation/bin:/usr/bin",
        "TBS_ORACLE_URL": "http://127.0.0.1:1234",
    })
    assert environment == {"HOME": "/dev/scratch", "PYTHONPATH": "/app/sdk", "PATH": "/usr/bin",
                           "TBS_ORACLE_URL": "http://127.0.0.1:1234"}


def test_bundle_is_self_contained_and_rejects_unknown_tasks(tmp_path: Path) -> None:
    metadata = isolation.create_bundle("terminal-bench-science/hbv-calibration-1", tmp_path / "bundle")
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == "terminal-bench-science/hbv-calibration-1"
    assert (tmp_path / "bundle/bin/runuser").stat().st_mode & 0o111
    assert (tmp_path / "bundle/dna_codec_isolation.py").is_file()
    assert (tmp_path / "bundle/science_traffic_isolation.py").is_file()
    assert (tmp_path / "bundle/science_localization_isolation.py").is_file()
    assert (tmp_path / "bundle/science_genomic_isolation.py").is_file()
    assert (tmp_path / "bundle/science_worker_process_isolation.py").is_file()
    assert (tmp_path / "bundle/science_navigation_small_area_isolation.py").is_file()
    assert (tmp_path / "bundle/science_leaky_bloch_isolation.py").is_file()
    assert (tmp_path / "bundle/science_isolation.py").is_file()
    compile((tmp_path / "bundle/sitecustomize.py").read_text(), "sitecustomize.py", "exec")
    probe = subprocess.run(
        [sys.executable, "-c", "import science_isolation; print(science_isolation.dna_codec_isolation.TASK_ID)"],
        cwd=tmp_path, env={"PYTHONPATH": str(tmp_path / "bundle")},
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "terminal-bench-science/dna-storage-codec"
    with pytest.raises(ValueError, match="No audited"):
        isolation.create_bundle("unreviewed-task", tmp_path / "unsupported")


def _official_preexec(module: str):
    namespace = {"__name__": module}
    exec(compile("def _demote_solver_process(): pass", "/tests/official_runner.py", "exec"), namespace)
    return namespace["_demote_solver_process"]


def test_navigation_and_small_area_hooks_are_exact_and_bundled(tmp_path: Path) -> None:
    from evals.pilot import science_navigation_small_area_isolation as nav_isolation

    for slug in ("navigation-sensor-calibration", "small-area-equivalence"):
        bundle = tmp_path / slug
        isolation.create_bundle("terminal-bench-science/" + slug, bundle)
        assert (bundle / "science_navigation_small_area_isolation.py").is_file()
    nav_root = "/app/navigation-session-fixture"
    nav_args = [sys.executable, "-B", nav_root + "/runtime/solver.py", "--input-dir",
                nav_root + "/input", "--output-dir", nav_root + "/output"]
    nav_kwargs = {
        "cwd": nav_root + "/runtime", "env": {"HOME": nav_root + "/output/.runtime",
            "TMPDIR": nav_root + "/output/.runtime/tmp",
            "XDG_CACHE_HOME": nav_root + "/output/.runtime/cache"},
        "preexec_fn": _official_preexec("test_all"), "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "start_new_session": True, "close_fds": True,
    }
    assert isolation.is_candidate("terminal-bench-science/navigation-sensor-calibration",
                                  nav_args, nav_kwargs)
    assert not isolation.is_candidate("terminal-bench-science/navigation-sensor-calibration",
                                      nav_args, {**nav_kwargs, "umask": 0o077})
    small_root = "/app/sae_v4_case_fixture"
    small_args = [sys.executable, small_root + "/submission/solve.py", "--data-dir",
                  small_root + "/data", "--output-dir", small_root + "/output"]
    small_kwargs = {
        "cwd": Path(small_root + "/submission"), "env": {"HOME": small_root + "/home",
            "TMPDIR": small_root + "/home/tmp", "XDG_CACHE_HOME": small_root + "/home/cache"},
        "preexec_fn": _official_preexec("test_state"),
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
    }
    assert isolation.is_candidate("small-area-equivalence", small_args, small_kwargs)
    probe = [sys.executable, "-B", "-c", nav_isolation.NAV_PRIVACY_PROBE]
    probe_kwargs = {"cwd": "/app", "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
        "env": {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        "preexec_fn": _official_preexec("test_all"), "start_new_session": True}
    assert nav_isolation.is_privacy_probe(
        "terminal-bench-science/navigation-sensor-calibration", probe, probe_kwargs)
    assert not isolation.is_candidate(
        "terminal-bench-science/navigation-sensor-calibration", probe, probe_kwargs)


def test_leaky_bloch_replay_hook_is_exact_and_bundled(tmp_path: Path) -> None:
    from evals.pilot import science_leaky_bloch_isolation as leaky

    bundle = tmp_path / "leaky-bloch-meep"
    isolation.create_bundle("terminal-bench-science/leaky-bloch-meep", bundle)
    assert (bundle / "science_leaky_bloch_isolation.py").is_file()
    sandbox = "/tmp/reproduced_design_fixture"
    runtime = sandbox + "/.replay_runtime_fixture"
    args = [leaky.PRLIMIT_PATH, *leaky.LIMITS, "--", "/usr/bin/setpriv",
            "--reuid=65534", "--regid=65534", "--clear-groups", "--no-new-privs", "--",
            leaky.INTERPRETER_PATH, sandbox + "/design.py", "--output-dir", sandbox + "/output"]
    kwargs = {"cwd": sandbox, "env": {"HOME": runtime + "/home",
        "TMPDIR": runtime + "/tmp", "XDG_CACHE_HOME": runtime + "/cache"},
        "start_new_session": True, "umask": 0o077, "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    assert isolation.is_candidate("terminal-bench-science/leaky-bloch-meep", args, kwargs)
    wrong_script = list(args)
    wrong_script[14] = "/tests/trusted_meep_forward.py"
    assert not isolation.is_candidate("terminal-bench-science/leaky-bloch-meep",
                                      wrong_script, kwargs)


@pytest.mark.parametrize("slug", [
    "baseline-free-localization", "guided-wave-localization", "virtual-baseline-localization",
])
def test_localization_task_bundle_and_candidate_hook(slug: str, tmp_path: Path) -> None:
    metadata = isolation.create_bundle("terminal-bench-science/" + slug, tmp_path / slug)
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == "terminal-bench-science/" + slug
    assert (tmp_path / slug / "science_localization_isolation.py").is_file()
    prefix = isolation.science_localization_isolation.TASK_PREFIXES[slug]
    tempdir = "/tmp/" + prefix + "fixture"
    args = ["/usr/bin/python3", "/tests/solver_worker.py", tempdir + "/solution.py"]
    kwargs = {"cwd": tempdir, "preexec_fn": lambda: None}
    assert isolation.is_candidate("terminal-bench-science/" + slug, args, kwargs)


def test_genomic_model_ranking_bundle_and_candidate_hook(tmp_path: Path) -> None:
    slug = "genomic-model-ranking"
    metadata = isolation.create_bundle(
        "terminal-bench-science/" + slug, tmp_path / "genomic-model-ranking")
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == "terminal-bench-science/" + slug
    assert (tmp_path / slug / "science_genomic_isolation.py").is_file()
    root = "/tmp/genomic-ranking-run-fixture"
    args = ["python3", "-I", root + "/predict.py", "--input-dir", root + "/input",
            "--output", root + "/output/predictions.csv", "--ranking-output",
            root + "/output/model_ranking.json"]
    kwargs = {"cwd": root, "preexec_fn": lambda: None, "stdin": subprocess.DEVNULL,
              "close_fds": True}
    assert isolation.is_candidate(slug, args, kwargs)
    assert isolation.is_candidate("terminal-bench-science/" + slug, args, kwargs)


@pytest.mark.parametrize("slug", ["foraging-cognitive-model", "noisy-blackbox-optimization"])
def test_existing_python_worker_bundle_and_candidate_hook(slug: str, tmp_path: Path) -> None:
    metadata = isolation.create_bundle("terminal-bench-science/" + slug, tmp_path / slug)
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == "terminal-bench-science/" + slug
    assert (tmp_path / slug / "science_worker_process_isolation.py").is_file()
    if slug == "foraging-cognitive-model":
        args = ["/opt/relay-runtime/bin/python3", "-B", "/tests/worker.py"]
        kwargs = {"cwd": "/app", "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                  "stderr": object(), "text": True, "bufsize": 1}
    else:
        args = ["/opt/relay-runtime/bin/python3", "/tests/solver_worker.py",
                "/tmp/dfo-solver-fixture/solver.py"]
        kwargs = {"cwd": "/tmp/dfo-solver-fixture", "stdin": subprocess.PIPE,
                  "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL,
                  "bufsize": 0, "preexec_fn": lambda: None, "start_new_session": True}
    assert isolation.is_candidate(slug, args, kwargs)
    assert isolation.is_candidate("terminal-bench-science/" + slug, args, kwargs)


def test_full_science_task_can_use_bundle_but_unknown_candidate_code_fails_closed(tmp_path: Path) -> None:
    metadata = isolation.create_bundle(
        "terminal-bench-science/sparse-network-assimilation", tmp_path / "full-task-bundle")
    assert metadata["environment"]["RELAY_SCIENCE_TASK_ID"] == "terminal-bench-science/sparse-network-assimilation"
    assert isolation.task_slug(metadata["environment"]["RELAY_SCIENCE_TASK_ID"]) == "sparse-network-assimilation"
    with pytest.raises(ValueError, match="no submitted-code execution interface"):
        isolation.candidate_command(
            "terminal-bench-science/sparse-network-assimilation", ["/usr/bin/python3", "/app/solve.py"],
            environment={"PATH": "/usr/bin:/bin"}, source_root=tmp_path)


def test_measure_namespace_launch_overhead(fixture_root: Path, record_property) -> None:
    """Record startup cost; never subtract it from official benchmark timers."""
    direct, isolated = [], []
    for _ in range(5):
        started = time.perf_counter()
        subprocess.run(["/bin/true"], check=True)
        direct.append(time.perf_counter() - started)
        started = time.perf_counter()
        result = run_candidate(fixture_root, "localized-sspd-solver", ["/bin/true"])
        isolated.append(time.perf_counter() - started)
        assert result.returncode == 0, result.stderr
    record_property("median_direct_seconds", statistics.median(direct))
    record_property("median_isolated_seconds", statistics.median(isolated))
    record_property("median_extra_seconds", statistics.median(isolated) - statistics.median(direct))
