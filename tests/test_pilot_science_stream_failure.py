"""An audited stream loss stays unscored without blocking independent Science arms."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from evals.pilot import science_matrix, science_sample_matrix, science_suite


STREAM_LOG = (
    "rlm failed; forwarding the last context\n"
    "Traceback (most recent call last):\n"
    '  File "proxy.py", line 1, in post\n'
    "    response.read()\n"
    "httpx.RemoteProtocolError: peer closed connection without sending complete message body "
    "(incomplete chunked read)\n"
)


def interrupted_attempt(path: Path) -> dict[str, object]:
    (path / "worker/relay").mkdir(parents=True)
    metadata: dict[str, object] = {
        "status": "completed", "exit_code": 0, "stop_reason": None,
        "strategy": "rlm", "window": 32000, "harness_requests": 2,
        "protocol_valid": False, "strategy_errors": 1,
        "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
        "provider_input_tokens_per_request": [14000, 15000],
    }
    (path / "worker/run.json").write_text(json.dumps(metadata))
    (path / "worker/relay/runtime.log").write_text(STREAM_LOG)
    (path / "result.json").write_text(json.dumps({"status": "worker_protocol_invalid"}))
    return metadata


def matrix_args(tmp_path: Path, sampled: bool, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    args = argparse.Namespace(out=tmp_path / "matrix", scratch=tmp_path,
                              models=["gpt-6-luna"], methods=["rlm", "rlm_persistent"],
                              python=Path("python3"))
    if sampled:
        args.manifest = tmp_path / "manifest.json"
        args.manifest.write_text(json.dumps({"selected_ids": ["task-a"]}))
        args.tasks_root, args.images_root = tmp_path / "tasks", tmp_path / "images"
        monkeypatch.setattr(science_sample_matrix, "require_readiness", lambda *args: {})
    return args


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("resumed", [False, True], ids=["new_failure", "saved_failure"])
def test_stream_failure_is_unscored_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, resumed: bool,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args = matrix_args(tmp_path, sampled, monkeypatch)
    task_root = args.out / "task-a" if sampled else args.out
    prior = task_root / "gpt-6-luna/rlm/attempt-001"
    if resumed:
        interrupted_attempt(prior)
    calls: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        method = command[command.index("--strategy") + 1]
        calls.append(method)
        attempt = Path(command[command.index("--out") + 1])
        if method == "rlm":
            assert not resumed, "The preserved interrupted arm must not consume another attempt"
            interrupted_attempt(attempt)
        else:
            attempt.mkdir(parents=True)
            (attempt / "result.json").write_text(json.dumps({"status": "completed_failed"}))
        return subprocess.CompletedProcess(command, int(method == "rlm"))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.run_matrix(args) == 1
    assert calls == (["rlm_persistent"] if resumed else ["rlm", "rlm_persistent"])
    assert not (prior.parent / "attempt-002").exists()
    audit = json.loads((prior / "evaluation-audit.json").read_text())
    assert audit["classification"] == "infrastructure_failed"
    assert audit["reason"] == "upstream_stream_interrupted"
    assert audit["valid"] is False and audit["benchmark_score"] is None
    assert json.loads((prior / "worker/run.json").read_text())["status"] == "completed"
    assert json.loads((prior / "result.json").read_text()) == {"status": "worker_protocol_invalid"}
    assert (prior / "worker/relay/runtime.log").read_text() == STREAM_LOG
    state = json.loads((args.out / "matrix.json").read_text())
    assert state["status"] == "complete_with_failures"
    assert state["entries"][0]["status"] == "worker_infrastructure_failed"
    assert state["entries"][0]["reason"] == "upstream_stream_interrupted"
    assert state["entries"][1]["status"] == "completed_failed"


@pytest.mark.parametrize("sampled", [False, True], ids=["native", "sample"])
@pytest.mark.parametrize("mixed", [False, True], ids=["unclassified_only", "mixed_with_stream"])
def test_saved_unclassified_failure_still_blocks_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sampled: bool, mixed: bool,
) -> None:
    module = science_sample_matrix if sampled else science_matrix
    args = matrix_args(tmp_path, sampled, monkeypatch)
    task_root = args.out / "task-a" if sampled else args.out
    slot = task_root / "gpt-6-luna/rlm"
    attempt = slot / ("attempt-002" if mixed else "attempt-001")
    if mixed:
        interrupted_attempt(slot / "attempt-001")
    interrupted_attempt(attempt)
    (attempt / "worker/relay/runtime.log").write_text(STREAM_LOG.replace(
        "httpx.RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)",
        "RuntimeError: REPL initialization failed"))

    def refuse_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("A stream failure must not hide an unclassified failure or authorize a retry")

    monkeypatch.setattr(module.subprocess, "run", refuse_launch)
    assert module.run_matrix(args) == 1
    state = json.loads((args.out / "matrix.json").read_text())
    assert state["status"] == "infrastructure_failed"
    assert state["entries"][0]["path"] == str(attempt)


def test_stream_annotation_requires_original_metadata_and_log(tmp_path: Path) -> None:
    raw = interrupted_attempt(tmp_path / "attempt")
    path = tmp_path / "attempt"
    classified = science_suite.classify_worker(path / "worker", raw)
    assert classified["status"] == "infrastructure_failed"
    assert classified["failure_kind"] == "upstream_stream_interrupted"
    result = {"worker_status": classified}
    assert science_matrix.stream_interrupted(path, result)
    (path / "worker/relay/runtime.log").unlink()
    assert not science_matrix.stream_interrupted(path, result)


def test_other_codex_failure_prevents_stream_continuation(tmp_path: Path) -> None:
    raw = interrupted_attempt(tmp_path / "attempt")
    path = tmp_path / "attempt"
    (path / "worker/codex.jsonl").write_text(json.dumps({
        "type": "turn.failed", "error": {"message": "Unrelated worker failure"},
    }) + "\n")
    classified = science_suite.classify_worker(path / "worker", raw)
    assert classified["failure_kind"] == "infrastructure_failed"
    assert not science_matrix.stream_interrupted(path, {"worker_status": classified})
