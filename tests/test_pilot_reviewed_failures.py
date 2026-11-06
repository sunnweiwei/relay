"""Manual terminal-failure reviews must not authorize a retry or hide other errors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.pilot import arc_suite, failure_policy as policy


CAPACITY = {"error": {"type": "rate_limit_error", "code": "turn_rate_limit_exceeded",
                      "message": "Codex is at capacity, please try again later"}}
WARNING = f"upstream refused /backend-api/codex/responses (429): {json.dumps(CAPACITY).encode()!r}\n"
TRACEBACK = ('rlm_persistent failed; forwarding the last context\nTraceback (most recent call last):\n'
             '  File "/repo/relay/core/engine.py", line 464, in complete\n'
             f'RuntimeError: the model request failed with HTTP 429: {CAPACITY!r}\n')


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def bind(attempt: Path, **updates: Any) -> None:
    path = attempt / "continuation-audit.json"
    review = json.loads(path.read_text())
    review.update(evidence_sha256={name: hashlib.sha256((attempt / name).read_bytes()).hexdigest()
                                  for name in policy.REVIEW_EVIDENCE}, **updates)
    write(path, review)


def reviewed_case(attempt: Path, reason: str = policy.AUXILIARY_CAPACITY_EXHAUSTION) -> dict[str, Any]:
    reset = reason == policy.PERSISTENT_SESSION_RESET
    worker = {"status": "completed", "exit_code": 0, "stop_reason": None, "strategy": "rlm_persistent",
              "protocol_valid": reset, "strategy_errors": 0 if reset else 1,
              "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
              "window": 32000, "provider_input_tokens_per_request": [14000]}
    write(attempt / "worker/run.json", worker)
    (attempt / "worker/relay").mkdir()
    (attempt / "worker/relay/runtime.log").write_text("" if reset else WARNING + TRACEBACK)
    write(attempt / "worker/relay/trace.jsonl", {"summary_request": {}, "summary_status": 429, "answer": CAPACITY})
    write(attempt / "worker/upstream-events.jsonl", {"event": "response_headers", "status": 429,
                                                   "path": "/backend-api/codex/responses"})
    write(attempt / "evaluation-audit.json", {"valid": False, "classification": "infrastructure_failed",
                                            "reason": reason, "benchmark_score": None, "capacity_failure": False})
    source = Path(policy.__file__).resolve().parents[2] / "relay/strategies/rlm.py"
    write(attempt / "continuation-audit.json", {"version": 1, "reason": reason, "score": None,
        "continue_without_retry": True, "reviewed_repaired_rlm_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    bind(attempt)
    return worker


@pytest.mark.parametrize("reason", sorted(policy.REVIEWED_TERMINAL_REASONS))
def test_reviewed_terminal_reason_is_returned_without_writing_artifacts(tmp_path: Path, reason: str) -> None:
    reviewed_case(tmp_path, reason)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert policy.reviewed_terminal_failure(tmp_path) == reason
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", policy.REVIEW_EVIDENCE)
def test_stale_raw_evidence_is_rejected(tmp_path: Path, name: str) -> None:
    reviewed_case(tmp_path)
    with (tmp_path / name).open("a") as stream:
        stream.write("\n")
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("updates", [
    {"score": 0}, {"continue_without_retry": False}, {"version": True}, {"version": "1"},
    {"reason": "provider_capacity"}, {"reason": "protocol_invalid"},
])
def test_review_schema_is_explicit(tmp_path: Path, updates: dict[str, Any]) -> None:
    reviewed_case(tmp_path)
    bind(tmp_path, **updates)
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("updates", [
    {"stop_reason": "plan_quota_reserve_reached"}, {"error": "credit_balance_decreased"},
    {"failure_kind": "account_limit_reached"}, {"error": "quota_metadata_unavailable"},
    {"requests_over_context_budget": 1}, {"provider_input_tokens_per_request": [33000]},
    {"sandbox_execution_errors": 1}, {"error": "worker_sandbox_execution_failed"},
    {"strategy_errors": 2}, {"strategy_errors": True}, {"protocol_valid": True},
    {"status": "running"}, {"exit_code": -11}, {"failure_kind": "other_failure"},
])
def test_fresh_hashes_do_not_hide_mixed_or_nonterminal_metadata(tmp_path: Path, updates: dict[str, Any]) -> None:
    worker = reviewed_case(tmp_path)
    write(tmp_path / "worker/run.json", worker | updates)
    bind(tmp_path)
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("log", [
    "", WARNING, TRACEBACK.replace("turn_rate_limit_exceeded", "quota_exhausted"),
    TRACEBACK + "ValueError: unrelated failure\n", WARNING.replace("429", "500") + TRACEBACK,
    TRACEBACK.replace("rlm_persistent failed", "ace failed"),
])
def test_capacity_requires_the_actual_exact_traceback(tmp_path: Path, log: str) -> None:
    reviewed_case(tmp_path)
    (tmp_path / "worker/relay/runtime.log").write_text(log)
    bind(tmp_path)
    assert policy.reviewed_terminal_failure(tmp_path) is None


def test_unrelated_evaluation_audit_blocks_review(tmp_path: Path) -> None:
    reviewed_case(tmp_path)
    write(tmp_path / "evaluation-audit.json", {"valid": False, "classification": "infrastructure_failed",
                                             "reason": "arc_observation_missing"})
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("log", ["", WARNING * 2])
def test_manual_science_stop_and_recovered_warnings_are_allowed_for_reset(tmp_path: Path, log: str) -> None:
    worker = reviewed_case(tmp_path, policy.PERSISTENT_SESSION_RESET)
    write(tmp_path / "worker/run.json", worker | {"status": "infrastructure_failed", "exit_code": -15,
                                                  "error": "harness_exit_-15", "protocol_valid": False})
    (tmp_path / "worker/relay/runtime.log").write_text(log)
    bind(tmp_path, controller_stopped=True)
    assert policy.reviewed_terminal_failure(tmp_path) == policy.PERSISTENT_SESSION_RESET
    bind(tmp_path, controller_stopped=False)
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("change", ["wrong_method", "strategy_error", "unknown_log", "old_repair"])
def test_reset_requires_persistent_clean_execution_and_installed_repair(tmp_path: Path, change: str) -> None:
    worker = reviewed_case(tmp_path, policy.PERSISTENT_SESSION_RESET)
    if change == "wrong_method":
        write(tmp_path / "worker/run.json", worker | {"strategy": "rlm"})
    elif change == "strategy_error":
        write(tmp_path / "worker/run.json", worker | {"strategy_errors": 1})
    elif change == "unknown_log":
        (tmp_path / "worker/relay/runtime.log").write_text("RuntimeError: unknown failure\n")
    bind(tmp_path, **({"reviewed_repaired_rlm_sha256": "0" * 64} if change == "old_repair" else {}))
    assert policy.reviewed_terminal_failure(tmp_path) is None


@pytest.mark.parametrize("reason", sorted(policy.REVIEWED_TERMINAL_REASONS))
def test_arc_resume_skips_reviewed_failure_preserves_raw_summary_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "arc").mkdir(parents=True)
    (scratch / "arc/frozen_prompt.txt").write_text("Frozen prompt")
    out = scratch / "matrix/arc"
    slot = out / "gpt-6.1-sol/rlm_persistent"
    worker = reviewed_case(slot, reason)
    summary = {"worker_status": worker["status"], "protocol_valid": worker["protocol_valid"],
               "scored": worker["protocol_valid"], "error": None, "stop_reason": worker["stop_reason"]}
    write(slot / "summary.json", summary)
    write(slot / "service/arc_trace.jsonl", {"kind": "reset", "action_count": 1,
        "observation": {"state": "NOT_FINISHED", "frame_ascii": "01"}})
    raw = (slot / "summary.json").read_bytes()
    calls = []

    def run(arm: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(arm)
        return {"worker_status": "completed", "protocol_valid": True, "scored": True}

    monkeypatch.setattr(arc_suite, "run_arm", run)
    monkeypatch.setattr(sys, "argv", ["arc_suite", "--scratch", str(scratch), "--python", sys.executable,
        "--out-root", str(out), "--models", "gpt-6.1-sol", "--arms", "rlm_persistent", "ace", "--resume"])
    assert arc_suite.main() == 0
    assert calls == ["ace"]
    assert (slot / "summary.json").read_bytes() == raw
    assert not (out / "capacity-retries.json").exists()
    calls.clear()
    with (slot / "worker/relay/runtime.log").open("a") as stream:
        stream.write("\n")
    assert arc_suite.main() == 2
    assert calls == []
