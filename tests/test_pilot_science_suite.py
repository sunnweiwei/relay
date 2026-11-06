"""The Science suite must grade stopped submissions, but never preflights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.pilot import science_matrix, science_suite
from evals.pilot.science import run_logged


@pytest.mark.parametrize("worker_status,requests,graded", [
    ("stopped", 64, True),
    ("preflight_failed", 0, False),
    ("infrastructure_failed", 5, False),
])
def test_submission_grading_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     worker_status: str, requests: int, graded: bool) -> None:
    prompt = tmp_path / "science/runs/fixed-python-compaction-01/work/task.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Frozen task")
    numpy = tmp_path / "science/public/environment/.deps/numpy/__init__.py"
    numpy.parent.mkdir(parents=True)
    numpy.touch()
    output = tmp_path / "matrix/science/model/method"
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "--public" in command:
            worker = output / "worker"
            (worker / "work").mkdir(parents=True)
            (worker / "work/traffic_solver.py").write_text("candidate = True\n")
            (worker / "run.json").write_text(json.dumps({
                "status": worker_status, "harness_requests": requests}))
        else:
            verifier = output / "verifier"
            verifier.mkdir()
            (verifier / "verifier-result.json").write_text(json.dumps({
                "status": "completed_failed", "reward": 0}))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(science_suite.subprocess, "run", fake_run)
    monkeypatch.setattr(science_suite.subprocess, "check_output", lambda *a, **k: "abc\n")
    args = argparse.Namespace(scratch=tmp_path, out=output, python=Path(".venv/bin/python"),
                              model="gpt-6.1-sol", strategy="clm")
    status = science_suite.run_one(args)
    result = json.loads((output / "result.json").read_text())
    assert len(calls) == (2 if graded else 1)
    assert status == (0 if graded else 1)
    if graded:
        assert result["verifier"]["reward"] == 0
        frozen = Path(result["artifact"])
        assert frozen.read_text() == "candidate = True\n"
        assert "--artifact" in calls[-1] and str(frozen) in calls[-1]
    else:
        assert result["status"] == ("worker_infrastructure_failed" if worker_status == "infrastructure_failed"
                                    else "worker_preflight_failed")
        assert "verifier" not in result


def test_output_must_be_hidden_with_old_private_references(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    args = argparse.Namespace(scratch=scratch, out=tmp_path / "exposed-output")
    with pytest.raises(ValueError):
        science_suite.run_one(args)
    assert not args.out.exists()


def test_verifier_timeout_preserves_partial_log(tmp_path: Path) -> None:
    log = tmp_path / "candidate.log"
    with pytest.raises(TimeoutError):
        run_logged([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                   log, timeout=1)
    assert log.read_text() == "started\n"


@pytest.mark.parametrize("reason,expected_status,expected_exit", [
    ("ordinary_plan_usage_not_allowed", "quota_stopped", 2),
    ("worker_isolation_preflight_failed", "infrastructure_failed", 1),
])
def test_matrix_keeps_graded_attempts_and_stops_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
    expected_status: str, expected_exit: int,
) -> None:
    output = tmp_path / "matrix"
    prior = output / "gpt-6-luna/compaction/attempt-001/result.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({"status": "completed_failed"}))
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        attempt = Path(command[command.index("--out") + 1])
        attempt.mkdir(parents=True)
        (attempt / "result.json").write_text(json.dumps({
            "status": "worker_preflight_failed",
            "worker_status": {"error": f"RuntimeError: {reason}"}}))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(science_matrix.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=tmp_path, out=output, python=Path("python3"),
                              models=["gpt-6-luna", "gpt-6.1-sol"], methods=["compaction", "clm"])
    assert science_matrix.run_matrix(args) == expected_exit
    assert len(calls) == 1
    state = json.loads((output / "matrix.json").read_text())
    assert state["status"] == expected_status
    assert state["entries"][0]["status"] == "already_graded"
    assert prior.read_text() == json.dumps({"status": "completed_failed"})


def test_old_capacity_failure_is_unknown_and_retried_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "matrix"
    prior = output / "gpt-6.1-sol/compaction/attempt-001"
    (prior / "worker").mkdir(parents=True)
    (prior / "result.json").write_text(json.dumps({"status": "completed_failed", "verifier": {"reward": 0}}))
    failure = {"status": "stopped", "exit_code": 1, "stop_reason": None, "harness_requests": 3}
    (prior / "worker/run.json").write_text(json.dumps(failure))
    (prior / "worker/codex.jsonl").write_text(json.dumps({
        "type": "turn.failed", "error": {"message": "Selected model is at capacity"}}) + "\n")
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        attempt = Path(command[command.index("--out") + 1])
        attempt.mkdir(parents=True)
        sol = command[command.index("--model") + 1] == "gpt-6.1-sol"
        result = {"status": "worker_infrastructure_failed", "worker_status": {
            "status": "infrastructure_failed", "error": "Selected model is at capacity"}} if sol else {
                "status": "completed_failed", "verifier": {"reward": 0}}
        (attempt / "result.json").write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, int(sol))

    monkeypatch.setattr(science_matrix.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=tmp_path, out=output, python=Path("python3"),
                              models=["gpt-6.1-sol", "gpt-6-luna"], methods=["compaction"])
    assert science_matrix.run_matrix(args) == 1
    assert len(calls) == 2  # One fresh Sol retry, then the independent Luna cell.
    assert not (prior.parent / "attempt-003").exists()
    assert json.loads((prior / "evaluation-audit.json").read_text())["valid"] is False
    assert json.loads((prior / "result.json").read_text())["verifier"]["reward"] == 0  # Raw result retained.
    assert json.loads((output / "matrix.json").read_text())["entries"][-1]["status"] == "completed_failed"


def test_enforced_request_stop_remains_gradable(tmp_path: Path) -> None:
    (tmp_path / "codex.jsonl").write_text(json.dumps({
        "type": "turn.failed", "error": {"message": "request denied"}}) + "\n")
    original = {"status": "stopped", "exit_code": 1, "stop_reason": "model_request_limit"}
    assert science_suite.classify_worker(tmp_path, original) == original


def test_resume_does_not_accept_grade_from_protocol_invalid_worker(tmp_path: Path) -> None:
    (tmp_path / "worker").mkdir()
    original = {"status": "completed_failed", "verifier": {"reward": 0}}
    (tmp_path / "result.json").write_text(json.dumps(original))
    (tmp_path / "worker/run.json").write_text(json.dumps({
        "status": "completed", "exit_code": 0, "protocol_valid": False}))
    result = science_matrix.attempt_result(tmp_path)
    assert result["status"] == "worker_protocol_invalid"
    assert result["status"] not in science_matrix.GRADED
    assert json.loads((tmp_path / "evaluation-audit.json").read_text())["valid"] is False
    assert json.loads((tmp_path / "result.json").read_text()) == original


def test_native_resume_preserves_pure_context_limit_without_repeating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "matrix"
    attempt = output / "gpt-6-luna/clm/attempt-001"
    (attempt / "worker").mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({"status": "completed_failed", "verifier": {"reward": 0}}))
    (attempt / "worker/run.json").write_text(json.dumps({
        "status": "stopped", "exit_code": 1, "stop_reason": "context_budget_exceeded",
        "protocol_valid": False, "requests_over_context_budget": 1,
        "strategy_errors": [], "sandbox_execution_errors": []}))
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        next_attempt = Path(command[command.index("--out") + 1])
        next_attempt.mkdir(parents=True)
        (next_attempt / "result.json").write_text(json.dumps({"status": "completed_failed"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(science_matrix.subprocess, "run", fake_run)
    args = argparse.Namespace(scratch=tmp_path, out=output, python=Path("python3"),
                              models=["gpt-6-luna"], methods=["clm", "folding"])
    assert science_matrix.run_matrix(args) == 1
    assert len(calls) == 1 and calls[0][calls[0].index("--strategy") + 1] == "folding"
    audit = json.loads((attempt / "evaluation-audit.json").read_text())
    assert audit["classification"] == "context_budget_exceeded"
