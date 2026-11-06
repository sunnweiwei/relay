"""Quota-truncated Science work is unscored and cannot silently resume."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_matrix, science_sample_matrix, science_sample_suite, science_suite


QUOTA_REASONS = (
    "plan_quota_reserve_reached", "ordinary_plan_usage_not_allowed",
    "credit_balance_decreased", "account_limit_reached",
    "quota_check_failed:TimeoutError", "quota_metadata_unavailable",
)


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("reason", QUOTA_REASONS)
def test_quota_stop_never_runs_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, reason: str,
) -> None:
    output = tmp_path / "attempt"
    module = science_sample_suite if sampled else science_suite
    args = argparse.Namespace(scratch=tmp_path, out=output, python=Path("python3"),
                              model="gpt-6-luna", strategy="compaction")
    calls: list[list[str]] = []

    def fake_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert "--public" in command, "Quota stop must not launch the native verifier"
        worker = output / "worker"
        (worker / "work").mkdir(parents=True)
        (worker / "work/traffic_solver.py").write_text("candidate = True\n")
        (worker / "run.json").write_text(json.dumps({
            "status": "stopped", "stop_reason": reason, "exit_code": 1,
            "harness_requests": 5, "protocol_valid": True,
        }))
        return subprocess.CompletedProcess(command, 1)

    def refuse_verifier(*args: object, **kwargs: object) -> None:
        pytest.fail("Quota stop must not expose verifier data or grade a partial task")

    monkeypatch.setattr(module.subprocess, "run", fake_worker)
    if sampled:
        args.task, args.images = tmp_path / "task", tmp_path / "images"
        args.task.mkdir()
        args.images.mkdir()
        (args.task / "task.toml").write_text(
            '[task]\nname="task-a"\n[environment]\ncpus=1\n')
        (args.task / "instruction.md").write_text("Write the requested output.")
        (args.images / "build.json").write_text(json.dumps({
            "images": {"environment": {"workdir": "/app"}},
        }))
        monkeypatch.setattr(module, "require_readiness", lambda *args: {})
        monkeypatch.setattr(module, "start_service", lambda *args, **kwargs: (object(), {}))
        monkeypatch.setattr(module, "stop_process", lambda *args: None)
        monkeypatch.setattr(module, "freeze_artifacts", refuse_verifier)
        monkeypatch.setattr(module, "verify", refuse_verifier)
    else:
        prompt = tmp_path / "science/runs/fixed-python-compaction-01/work/task.txt"
        prompt.parent.mkdir(parents=True)
        prompt.write_text("Frozen task")
        numpy = tmp_path / "science/public/environment/.deps/numpy/__init__.py"
        numpy.parent.mkdir(parents=True)
        numpy.touch()
        monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "abc\n")

    assert module.run_one(args) == 2
    assert len(calls) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "worker_quota_stopped"
    assert result["stop_reason"] == reason
    assert result["benchmark_score"] is None
    assert "verifier" not in result
    assert not (output / "verifier").exists()
    audit = json.loads((output / "evaluation-audit.json").read_text())
    assert audit == {"valid": False, "classification": "quota_stopped",
                     "reason": reason, "benchmark_score": None}


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("reason", QUOTA_REASONS)
def test_saved_quota_stop_blocks_resume_before_missing_or_previously_graded_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, reason: str,
) -> None:
    output = tmp_path / "matrix"
    task_root = output / "task-a" if sampled else output
    quota_attempt = task_root / "gpt-6-luna/clm/attempt-001"
    (quota_attempt / "worker").mkdir(parents=True)
    original = {"status": "completed", "verifier": {"reward": 1}}
    (quota_attempt / "result.json").write_text(json.dumps(original))
    (quota_attempt / "worker/run.json").write_text(json.dumps({
        "status": "stopped", "stop_reason": reason, "exit_code": 1,
        "harness_requests": 5, "protocol_valid": True,
    }))
    # An additional valid grade must not hide the quota stop in the same cell.
    valid_attempt = quota_attempt.parent / "attempt-002"
    valid_attempt.mkdir()
    (valid_attempt / "result.json").write_text(json.dumps(original))

    def refuse_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("A saved quota stop must prevent all resumed inference")

    module = science_sample_matrix if sampled else science_matrix
    monkeypatch.setattr(module.subprocess, "run", refuse_launch)
    args = argparse.Namespace(out=output, scratch=tmp_path, models=["gpt-6-luna"],
                              methods=["compaction", "clm"], python=Path("python3"))
    if sampled:
        args.manifest = tmp_path / "manifest.json"
        args.manifest.write_text(json.dumps({"selected_ids": ["task-a"]}))
        args.tasks_root, args.images_root = tmp_path / "tasks", tmp_path / "images"
        monkeypatch.setattr(module, "require_readiness", refuse_launch)

    assert module.run_matrix(args) == 2
    state = json.loads((output / "matrix.json").read_text())
    assert state["status"] == "quota_stopped"
    assert state["stop_reason"] == reason
    assert state["stopped_attempt"] == str(quota_attempt)
    assert state["entries"] == []
    assert not (task_root / "gpt-6-luna/compaction").exists()
    assert not (quota_attempt.parent / "attempt-003").exists()
    assert json.loads((quota_attempt / "result.json").read_text()) == original
    assert science_matrix.attempt_result(quota_attempt)["status"] == "worker_quota_stopped"
    audit = json.loads((quota_attempt / "evaluation-audit.json").read_text())
    assert audit["valid"] is False and audit["classification"] == "quota_stopped"


def test_unrelated_stop_reason_does_not_mask_quota_error() -> None:
    metadata = {"stop_reason": "wall_time_limit", "error": "quota_check_failed:TimeoutError"}
    assert science_suite.quota_stop_reason(metadata) == "quota_check_failed:TimeoutError"
    assert science_matrix.quota_stopped({"worker_status": metadata})


def test_worker_exit_classification_preserves_quota_error_without_stop_reason(tmp_path: Path) -> None:
    metadata = {"status": "stopped", "exit_code": 1, "error": "quota_check_failed:TimeoutError"}
    assert science_suite.classify_worker(tmp_path, metadata) == metadata
    (tmp_path / "worker").mkdir()
    (tmp_path / "worker/run.json").write_text(json.dumps(metadata))
    (tmp_path / "result.json").write_text(json.dumps({"status": "completed", "verifier": {"reward": 1}}))
    result = science_matrix.attempt_result(tmp_path)
    assert result["status"] == "worker_quota_stopped"
    assert result["stop_reason"] == metadata["error"]
