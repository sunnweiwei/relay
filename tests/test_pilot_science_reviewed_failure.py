"""Reviewed Science exclusions authorize continuation, never a retry or a score."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.pilot import failure_policy, science_matrix, science_sample_matrix


def reviewed_attempt(path: Path) -> None:
    (path / "worker/relay").mkdir(parents=True)
    worker = {"status": "completed", "exit_code": 0, "stop_reason": None,
              "protocol_valid": True, "strategy": "rlm_persistent", "strategy_errors": 0,
              "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
              "window": 32000, "provider_input_tokens_per_request": [1000]}
    (path / "worker/run.json").write_text(json.dumps(worker))
    (path / "worker/relay/runtime.log").write_text("")
    (path / "worker/relay/trace.jsonl").write_text(json.dumps({
        "body": {"input": [{"role": "user", "content": "task"}]},
    }) + "\n")
    (path / "worker/upstream-events.jsonl").write_text(json.dumps({
        "event": "response_headers", "path": "/responses", "status": 200,
    }) + "\n")
    (path / "result.json").write_text(json.dumps({"status": "completed_failed", "verifier": {"reward": 0}}))
    (path / "evaluation-audit.json").write_text(json.dumps({
        "valid": False, "classification": "infrastructure_failed",
        "reason": failure_policy.PERSISTENT_SESSION_RESET, "benchmark_score": None,
    }))
    source = Path(failure_policy.__file__).resolve().parents[2] / "relay/strategies/rlm.py"
    review = {"version": 1, "reason": failure_policy.PERSISTENT_SESSION_RESET,
              "score": None, "continue_without_retry": True,
              "reviewed_repaired_rlm_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "evidence_sha256": {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                  for name in failure_policy.REVIEW_EVIDENCE}}
    (path / "continuation-audit.json").write_text(json.dumps(review))


def setup_matrix(tmp_path: Path, sampled: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, Path]:
    args = argparse.Namespace(out=tmp_path / "matrix", scratch=tmp_path, python=Path("python3"),
                              models=["gpt-6-luna"], methods=["rlm_persistent", "compaction"])
    task_root = args.out
    if sampled:
        args.manifest = tmp_path / "manifest.json"
        args.manifest.write_text(json.dumps({"selected_ids": ["task-a"]}))
        args.tasks_root, args.images_root = tmp_path / "tasks", tmp_path / "images"
        monkeypatch.setattr(science_sample_matrix, "require_readiness", lambda *args: {})
        task_root /= "task-a"
    attempt = task_root / "gpt-6-luna/rlm_persistent/attempt-001"
    reviewed_attempt(attempt)
    return args, attempt


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
def test_reviewed_failure_preserves_audit_and_continues_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args, attempt = setup_matrix(tmp_path, sampled, monkeypatch)
    preserved = {name: (attempt / name).read_bytes() for name in [
        "result.json", "evaluation-audit.json", "continuation-audit.json", *failure_policy.REVIEW_EVIDENCE]}
    calls: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command[command.index("--strategy") + 1])
        assert calls[-1] == "compaction", "The reviewed invalid arm must not run again"
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps({"status": "completed_failed"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.run_matrix(args) == 1
    assert calls == ["compaction"]
    assert not (attempt.parent / "attempt-002").exists()
    state = json.loads((args.out / "matrix.json").read_text())
    assert state["status"] == "complete_with_failures"
    assert state["entries"][0]["reason"] == failure_policy.PERSISTENT_SESSION_RESET
    assert state["entries"][0]["benchmark_score"] is None
    assert {name: (attempt / name).read_bytes() for name in preserved} == preserved


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("defect", ["stale_evidence", "malformed", "wrong_repair", "unrelated_audit", "quota"])
def test_unvalidated_review_or_quota_blocks_all_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, defect: str,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args, attempt = setup_matrix(tmp_path, sampled, monkeypatch)
    review = attempt / "continuation-audit.json"
    if defect == "stale_evidence":
        (attempt / "worker/relay/trace.jsonl").write_text('{"changed": true}\n')
        # A different saved grade cannot hide a stale controller decision.
        graded = attempt.parent / "attempt-002"
        graded.mkdir()
        (graded / "result.json").write_text(json.dumps({"status": "completed"}))
    elif defect == "malformed":
        review.write_text("invalid JSON")
    elif defect == "wrong_repair":
        content = json.loads(review.read_text())
        content["reviewed_repaired_rlm_sha256"] = "0" * 64
        review.write_text(json.dumps(content))
    elif defect == "unrelated_audit":
        review.unlink()
        audit = json.loads((attempt / "evaluation-audit.json").read_text())
        audit["reason"] = "unreviewed_error"
        (attempt / "evaluation-audit.json").write_text(json.dumps(audit))
    else:
        worker = json.loads((attempt / "worker/run.json").read_text())
        worker.update(status="stopped", stop_reason="plan_quota_reserve_reached", exit_code=1)
        (attempt / "worker/run.json").write_text(json.dumps(worker))

    def refuse_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("A reviewed label must not bypass missing evidence or a quota stop")

    monkeypatch.setattr(module.subprocess, "run", refuse_launch)
    assert module.run_matrix(args) == (2 if defect == "quota" else 1)
    state = json.loads((args.out / "matrix.json").read_text())
    assert state["status"] == ("quota_stopped" if defect == "quota" else "infrastructure_failed")


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
def test_reviewed_failure_does_not_hide_another_unclassified_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args, attempt = setup_matrix(tmp_path, sampled, monkeypatch)
    other = attempt.parent / "attempt-002"
    other.mkdir()
    (other / "result.json").write_text(json.dumps({"status": "worker_preflight_failed"}))

    def refuse_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("An unrelated failure must still block the queue")

    monkeypatch.setattr(module.subprocess, "run", refuse_launch)
    assert module.run_matrix(args) == 1
    assert json.loads((args.out / "matrix.json").read_text())["status"] == "infrastructure_failed"


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("fresh", [False, True], ids=["resume", "worker_return"])
@pytest.mark.parametrize("new_failure", ["context_budget", "provider_capacity"])
def test_stale_review_blocks_context_skip_and_capacity_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, fresh: bool, new_failure: str,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args, attempt = setup_matrix(tmp_path, sampled, monkeypatch)
    calls: list[str] = []

    def invalidate_review() -> None:
        path = attempt / "worker/run.json"
        worker = json.loads(path.read_text())
        if new_failure == "context_budget":
            worker.update(status="stopped", stop_reason="context_budget_exceeded", exit_code=1,
                          protocol_valid=False, requests_over_context_budget=1,
                          provider_input_tokens_per_request=[32001])
        else:
            worker.update(status="infrastructure_failed", exit_code=1, protocol_valid=False,
                          error="provider capacity unavailable")
            (attempt / "worker/codex.jsonl").write_text(json.dumps({
                "type": "turn.failed", "error": {"message": "provider capacity unavailable"},
            }) + "\n")
        path.write_text(json.dumps(worker))
        result = science_matrix.attempt_result(attempt)
        assert result["reviewed_failure_validation_failed"] is True
        if new_failure == "context_budget":
            assert science_matrix.context_budget_only(result["worker_status"])
        else:
            assert science_matrix.at_capacity(result)

    if fresh:
        shutil.rmtree(attempt)
    else:
        invalidate_review()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert fresh and not calls, "A stale review must block retries and later methods"
        calls.append(command[command.index("--strategy") + 1])
        assert calls == ["rlm_persistent"]
        reviewed_attempt(attempt)
        invalidate_review()
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.run_matrix(args) == 1
    assert calls == (["rlm_persistent"] if fresh else [])
    assert not (attempt.parent / "attempt-002").exists()
    assert json.loads((args.out / "matrix.json").read_text())["status"] == "infrastructure_failed"
