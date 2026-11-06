"""Prepare and run the Terminal-Bench-Science traffic-flux pilot verifier.

The task runner is split into three Apptainer invocations. The submitted
solver runs in an invocation that does not bind the private test directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


# The evaluation's working tree (data, runtimes and run outputs); see evals/pilot/README.md.
SCRATCH = Path(os.environ.get("RELAY_PILOT_ROOT", "relay-pilot"))
SCIENCE_ROOT = SCRATCH / "science"
PUBLIC_ROOT = SCIENCE_ROOT / "public"
PRIVATE_TESTS = SCIENCE_ROOT / "private" / "tests"
RUNTIME = SCRATCH / "science-runtime"
IMAGE = SCRATCH / "python312.sif"
SITE_PACKAGES = RUNTIME / ".venv" / "lib" / "python3.12" / "site-packages"


def run_logged(command: Sequence[str], log_path: Path, timeout: float) -> int:
    """Run a command, tee its output, and save the same output to a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # Stop the candidate's whole process group before returning. Keep
            # partial logs, which are often the only evidence for a timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise TimeoutError(f"command timed out after {timeout:.0f}s: {command}") from exc
    output = log_path.read_text(encoding="utf-8", errors="replace")
    lines = output.splitlines(keepends=True)
    if len(lines) <= 80:
        sys.stdout.write(output)
    else:
        sys.stdout.writelines(lines[:8])
        sys.stdout.write(f"... {len(lines) - 58} lines omitted; full log: {log_path}\n")
        sys.stdout.writelines(lines[-50:])
    return returncode


def apptainer_base() -> list[str]:
    if not IMAGE.is_file() or not SITE_PACKAGES.is_dir():
        raise FileNotFoundError("Run the preparation steps in SCIENCE.md first.")
    return [
        "apptainer", "exec", "--cleanenv", "--containall", "--no-home",
        "--net", "--network=none", "--pwd", "/",
        "--bind", f"{SITE_PACKAGES}:/opt/science-deps:ro",
        "--env", "PYTHONPATH=/opt/science-deps",
    ]


def prepare() -> None:
    """Check that the pinned public task and isolated verifier files exist."""
    required = [
        PUBLIC_ROOT / "instruction.md",
        PUBLIC_ROOT / "task.toml",
        PUBLIC_ROOT / "environment" / "traffic_solver.py",
        PUBLIC_ROOT / "environment" / "flux_models.py",
        PUBLIC_ROOT / "environment" / "data" / "episodes.npz",
        PUBLIC_ROOT / "environment" / ".deps" / "numpy" / "__init__.py",
        PRIVATE_TESTS / "harness_export.py",
        PRIVATE_TESTS / "harness_run.py",
        PRIVATE_TESTS / "test_flux.py",
        PRIVATE_TESTS / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing downloaded task files:\n" + "\n".join(missing))
    if (SCIENCE_ROOT / "solution").exists() or (PUBLIC_ROOT / "solution").exists():
        raise RuntimeError("Solution files must not be present in the pilot workspace.")
    print(f"Task public files: {PUBLIC_ROOT}")
    print(f"Verifier-only files: {PRIVATE_TESTS}")
    print(f"Python image: {IMAGE}")
    print(f"Runtime dependencies: {SITE_PACKAGES}")


def verify(artifact: Path | None, run_dir: Path, expected_failure: bool) -> int:
    """Run official export, candidate, and pytest phases with separated mounts."""
    run_dir = run_dir.resolve()
    work = run_dir / "work"
    app = run_dir / "app"
    logs = run_dir / "logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "verifier-result.json"
    started_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, object] = {
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "phase": "prepare",
        "exit_code": None,
        "timeout": False,
        "reward": None,
        "junit": None,
        "stdout_case_count": None,
    }

    def save_result(**updates: object) -> None:
        result.update(updates)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save_result()
    prepare()
    if app.exists():
        shutil.rmtree(app)
    shutil.copytree(PUBLIC_ROOT / "environment", app,
                    ignore=shutil.ignore_patterns("Dockerfile", ".deps"))
    if artifact is not None:
        artifact = artifact.resolve(strict=True)
        shutil.copy2(artifact, app / "traffic_solver.py")

    common = apptainer_base()
    deadline = time.monotonic() + 1800.0

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("official verifier budget of 1800 seconds exhausted")
        return seconds

    # Export only the legal inputs from private tests. No candidate code runs
    # in this phase.
    export_command = common + [
        "--bind", f"{PRIVATE_TESTS}:/tests:ro",
        "--bind", f"{work}:/work",
        str(IMAGE),
        "python", "/tests/harness_export.py",
    ]
    save_result(phase="export")
    try:
        export_status = run_logged(export_command, logs / "export.log", remaining())
    except TimeoutError as exc:
        save_result(status="timeout", phase="export", timeout=True,
                    finished_at=datetime.now(timezone.utc).isoformat(), error=str(exc))
        raise
    if export_status:
        save_result(status="failed", phase="export", exit_code=export_status,
                    finished_at=datetime.now(timezone.utc).isoformat())
        return export_status

    # This is the only phase that executes the submitted solver. It receives
    # /app and exported /work, but private tests are not mounted.
    public_runner = run_dir / "public-runner" / "harness_run.py"
    public_runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRIVATE_TESTS / "harness_run.py", public_runner)
    candidate_command = common + [
        "--bind", f"{app}:/app:ro",
        "--bind", f"{work}:/work",
        "--bind", f"{public_runner}:/harness_run.py:ro",
        str(IMAGE),
        "python", "-c",
        "from pathlib import Path; assert not Path('/tests').exists(); "
        "import runpy; runpy.run_path('/harness_run.py', run_name='__main__')",
    ]
    # This small dispatcher contains no tests or reference data. It is the
    # only verifier-side code mounted into the candidate phase.
    save_result(phase="candidate")
    try:
        candidate_status = run_logged(candidate_command, logs / "candidate.log", remaining())
    except TimeoutError as exc:
        save_result(status="timeout", phase="candidate", timeout=True,
                    finished_at=datetime.now(timezone.utc).isoformat(), error=str(exc))
        raise
    save_result(phase="pytest", candidate_exit_code=candidate_status)
    if candidate_status:
        print(f"candidate phase exited {candidate_status}; continuing to grade any outputs")

    # Pytest is a separate, network-isolated invocation. The candidate phase
    # has exited before this phase mounts the grader files.
    grade_command = common + [
        "--bind", f"{PRIVATE_TESTS}:/tests:ro",
        "--bind", f"{work}:/work",
        "--bind", f"{logs}:/logs/verifier",
        str(IMAGE),
        "python", "-m", "pytest", "--ctrf", "/logs/verifier/ctrf.json",
        "--junitxml", "/logs/verifier/junit.xml",
        "/tests/test_flux.py", "-rA",
    ]
    try:
        grade_status = run_logged(grade_command, logs / "pytest.log", remaining())
    except TimeoutError as exc:
        save_result(status="timeout", phase="pytest", timeout=True,
                    candidate_exit_code=candidate_status,
                    finished_at=datetime.now(timezone.utc).isoformat(), error=str(exc))
        raise
    reward = "1\n" if grade_status == 0 else "0\n"
    (logs / "reward.txt").write_text(reward, encoding="utf-8")
    junit: dict[str, int] | None = None
    junit_path = logs / "junit.xml"
    if junit_path.exists():
        root = ET.parse(junit_path).getroot()
        suites = list(root.iter("testsuite")) or [root]
        junit = {key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
                 for key in ("tests", "failures", "errors", "skipped")}
    stdout_case_count = None
    for line in (logs / "pytest.log").read_text(encoding="utf-8").splitlines():
        counts = re.findall(r"(\d+)\s+(?:passed|failed|errors?|skipped)", line)
        if counts:
            stdout_case_count = sum(map(int, counts))
    save_result(status="completed" if grade_status == 0 else "completed_failed",
                phase="complete", exit_code=grade_status, timeout=False,
                reward=int(grade_status == 0), junit=junit,
                stdout_case_count=stdout_case_count,
                candidate_exit_code=candidate_status,
                finished_at=datetime.now(timezone.utc).isoformat(),
                expected_failure_requested=expected_failure)
    print(f"reward={reward.strip()} (pytest exit {grade_status})")
    print(f"CTR report: {logs / 'ctrf.json'}")
    if expected_failure and grade_status != 0:
        return 0
    return grade_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="check the pinned task/runtime paths")
    verify_parser = subparsers.add_parser("verify", help="run the separated official verifier")
    verify_parser.add_argument("--artifact", type=Path, help="candidate traffic_solver.py")
    verify_parser.add_argument("--run-dir", type=Path, required=True)
    verify_parser.add_argument("--expect-failure", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return 0
    return verify(args.artifact, args.run_dir, args.expect_failure)


if __name__ == "__main__":
    raise SystemExit(main())
