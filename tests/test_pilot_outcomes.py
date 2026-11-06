from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.pilot.outcomes import browse_profile_applied, failure_cause, fallback_counts, outcome


def worker(tmp_path: Path, log: str = "", harness: str = "") -> Path:
    (tmp_path / "relay").mkdir(parents=True, exist_ok=True)
    (tmp_path / "relay/runtime.log").write_text(log)
    (tmp_path / "codex.jsonl").write_text(harness)
    return tmp_path


@pytest.mark.parametrize("log,harness,run,expected", [
    ("Traceback ...\nhttpx.RemoteProtocolError: peer closed connection without sending complete message body\n",
     "", {"stop_reason": "context_management_failed"}, "upstream_connection_lost"),
    ("RuntimeError: the model request failed with HTTP 429: {'detail': 'Rate limit exceeded'}\n",
     "", {"stop_reason": "context_management_failed"}, "rate_limited"),
    ("RuntimeError: the RLM gave no answer\n", "", {"stop_reason": "context_management_failed"}, "method_error"),
    ("", '{"type":"error","message":"Invalid prompt: your prompt was flagged"}', {}, "provider_refusal"),
    ("", '{"type":"response.incomplete","reason":"max_output_tokens"}', {}, "max_output_tokens"),
    ("", "", {"stop_reason": "quota_window_changed"}, "quota_stop"),
    ("", "", {"stop_reason": "input_accounting_failed"}, "input_accounting"),
    ("", "", {"stop_reason": "upstream_request_timeout"}, "upstream_connection_lost"),
    ("", "", {}, "worker_infrastructure"),
])
def test_failure_cause_reads_the_run_records(tmp_path: Path, log: str, harness: str,
                                             run: dict, expected: str) -> None:
    assert failure_cause(worker(tmp_path, log, harness), run) == expected


def test_verifier_timeout_is_a_task_failure(tmp_path: Path) -> None:
    cause = failure_cause(worker(tmp_path), {"stop_reason": "model_request_limit"}, verifier_timeout=True)
    assert (cause, outcome(False, cause)) == ("verifier_timeout", "task_failure")


def test_outcome_policy() -> None:
    assert outcome(True, None) == "scored"
    assert outcome(False, "method_error") == "task_failure"
    assert outcome(False, "provider_refusal") == "task_failure"
    assert outcome(False, "upstream_connection_lost") == "infrastructure"
    assert outcome(False, "quota_stop") == "infrastructure"


def test_fallback_counts_separate_main_auxiliary_and_exhausted_recovery() -> None:
    run = {"overflow_fallback_events": [
        {"scope": "main", "dropped_items": 6, "truncated_text_items": 0},
        {"scope": "main", "dropped_items": 0, "truncated_text_items": 0},  # (fit without a change)
        {"scope": "summary", "dropped_items": 0, "truncated_text_items": 1},
        {"scope": "main", "action": "method_recovery_exhausted"},
    ]}
    assert fallback_counts(run) == {"main_truncations": 1, "auxiliary_truncations": 1, "recovery_exhausted": 1}
    assert fallback_counts({}) == {"main_truncations": 0, "auxiliary_truncations": 0, "recovery_exhausted": 0}


@pytest.mark.parametrize("text,applied", [("You are Codex, an agent ...", False), ("You are an agent solving ...", True)])
def test_browse_profile_applied_reads_the_first_forwarded_request(tmp_path: Path, text: str, applied: bool) -> None:
    body = {"input": [{"role": "developer", "content": [{"type": "input_text", "text": text}]}]}
    (tmp_path / "relay").mkdir()
    (tmp_path / "relay/trace.jsonl").write_text(json.dumps({"forwarded": body}) + "\n")
    assert browse_profile_applied(tmp_path) is applied
    assert browse_profile_applied(tmp_path / "missing") is None
