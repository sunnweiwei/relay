"""Only evidenced auxiliary stream interruptions permit later-arm admission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.pilot.failure_policy import upstream_stream_interrupted


MESSAGE = "peer closed connection without sending complete message body (incomplete chunked read)"
STREAM_LOG = ("rlm failed; forwarding the last context\nTraceback (most recent call last):\n"
              '  File "proxy.py", line 171, in post\n    response = summary_client.post(url)\n'
              f"httpcore.RemoteProtocolError: {MESSAGE}\n\n"
              "The above exception was the direct cause of the following exception:\n\n"
              "Traceback (most recent call last):\n"
              f"httpx.RemoteProtocolError: {MESSAGE}\n")


def interrupted_worker(**updates: Any) -> dict[str, Any]:
    return {"status": "stopped", "exit_code": -15, "stop_reason": "wall_time_limit",
            "strategy": "rlm", "protocol_valid": False, "strategy_errors": 1,
            "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
            "window": 32000, "provider_input_tokens_per_request": [14000], **updates}


@pytest.mark.parametrize("updates", [{}, {"status": "completed", "exit_code": 0, "stop_reason": None}])
def test_interrupted_stream_can_be_skipped_but_not_retried(tmp_path: Path, updates: dict[str, Any]) -> None:
    log = tmp_path / "runtime.log"
    log.write_text(STREAM_LOG)
    assert upstream_stream_interrupted(interrupted_worker(**updates), log)


@pytest.mark.parametrize("updates", [
    {"stop_reason": "plan_quota_reserve_reached"}, {"stop_reason": "account_limit_reached"},
    {"error": "credit_balance_decreased"}, {"error": "quota_metadata_unavailable"},
    {"stop_reason": "context_budget_exceeded"}, {"requests_over_context_budget": 1},
    {"sandbox_execution_errors": 1}, {"error": "worker_sandbox_execution_failed"},
    {"error": "provider_capacity"}, {"failure_kind": "context_budget_exceeded"},
    {"provider_input_tokens_per_request": [33000]}, {"provider_input_tokens_per_request": []},
    {"strategy_errors": 0}, {"strategy_errors": 2}, {"strategy_errors": True},
    {"sandbox_execution_errors": None}, {"requests_over_context_budget": None},
    {"status": "running"}, {"status": "infrastructure_failed"}, {"protocol_valid": True},
    {"exit_code": -11}, {"exit_code": None}, {"stop_reason": None},
    {"strategy": "ace"}, {"window": None},
])
def test_other_or_missing_failure_evidence_blocks_continuation(tmp_path: Path, updates: dict[str, Any]) -> None:
    log = tmp_path / "runtime.log"
    log.write_text(STREAM_LOG)
    assert not upstream_stream_interrupted(interrupted_worker(**updates), log)


@pytest.mark.parametrize("log_text", [
    "", STREAM_LOG.replace("httpx.RemoteProtocolError", "httpx.ReadTimeout"),
    STREAM_LOG.replace(MESSAGE, "Server disconnected"),
    STREAM_LOG + "ValueError: unrelated strategy defect\n",
    "could not prepare the request; forwarding it unchanged\n" + STREAM_LOG,
    STREAM_LOG.replace("httpcore.RemoteProtocolError", "ValueError"),
    STREAM_LOG + STREAM_LOG.replace("rlm failed", "ace failed"),
])
def test_missing_or_mixed_tracebacks_fail_closed(tmp_path: Path, log_text: str) -> None:
    log = tmp_path / "runtime.log"
    log.write_text(log_text)
    assert not upstream_stream_interrupted(interrupted_worker(), log)


def test_missing_runtime_log_is_not_evidence(tmp_path: Path) -> None:
    assert not upstream_stream_interrupted(interrupted_worker(), tmp_path / "absent.log")


def test_every_counted_strategy_failure_must_match(tmp_path: Path) -> None:
    log = tmp_path / "runtime.log"
    log.write_text(STREAM_LOG * 2)
    assert upstream_stream_interrupted(interrupted_worker(strategy_errors=2), log)
    log.write_text(STREAM_LOG + STREAM_LOG.replace("rlm failed", "ace failed"))
    assert not upstream_stream_interrupted(interrupted_worker(strategy_errors=2), log)
