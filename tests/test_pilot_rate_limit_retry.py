from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from evals.pilot.rate_limit_retry import build_retry_ledger, classify_attempt


RESPONSES = "/backend-api/codex/responses"


def make_attempt(root: Path, *, statuses=(200, 429), stop="context_management_failed",
                 failure="context_management_failed", strategy_errors=0,
                 error_payload='{"detail":"Rate limit exceeded"}',
                 summary_payload=None, summary=True) -> Path:
    (root / "relay").mkdir(parents=True)
    run = {"status": "stopped", "stop_reason": stop, "exit_code": -15,
           "failure_kind": failure, "strategy_errors": strategy_errors,
           "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
           "error": None}
    (root / "run.json").write_text(json.dumps(run))
    (root / "relay/runtime.log").write_text(
        f"upstream refused {RESPONSES} (429): {error_payload!r}\n"
        "RuntimeError: 429 Too Many Requests: rate limit exceeded\n")
    upstream = [
        {"event": "response_headers", "path": RESPONSES, "status": status}
        for status in statuses
    ]
    (root / "upstream-events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in upstream))
    trace = ([{"path": RESPONSES, "summary_status": status,
               "answer": (summary_payload if summary_payload is not None else
                          {"detail": "Rate limit exceeded"}) if status == 429 else None}
              for status in statuses] if summary else [{"path": RESPONSES, "body": {}}])
    (root / "relay/trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace))
    return root


def test_terminal_429_suffix_is_eligible_and_evidence_is_hashed(tmp_path: Path) -> None:
    attempt = make_attempt(tmp_path / "attempt")
    result = classify_attempt(attempt)
    assert result["eligible"] is True
    assert result["reason"] == "pure_terminal_http_429"
    assert {Path(path).name for path in result["evidence_sha256"]} >= {
        "run.json", "runtime.log", "upstream-events.jsonl", "trace.jsonl"}
    assert all(len(digest) == 64 for digest in result["evidence_sha256"].values())


@pytest.mark.parametrize("kwargs,reason", [
    ({"statuses": (429, 200)}, "429_was_incidental"),
    ({"stop": "model_request_limit", "failure": "model_request_limit"}, "local_execution_cap"),
    ({"statuses": (200, 500, 429)}, "unrelated_5xx_present"),
    ({"strategy_errors": 1}, "strategy_errors_present_or_unaccounted"),
    ({"failure": "folding_failed", "stop": "strategy_error"}, "terminal_failure_not_attributed_to_429"),
    ({"error_payload": '{"error":{"code":"insufficient_quota","message":"quota reached"}}'},
     "account_or_quota_limit_response"),
    ({"summary_payload": {"error": {"code": "plan_limit", "message": "account usage limit"}}},
     "account_or_quota_limit_response"),
])
def test_excludes_incidental_caps_account_limits_and_other_failures(
    tmp_path: Path, kwargs: dict, reason: str,
) -> None:
    result = classify_attempt(make_attempt(tmp_path / "attempt", **kwargs))
    assert result["eligible"] is False
    assert result["reason"] == reason


def test_terminal_main_response_429_can_be_proven_without_summary_request(tmp_path: Path) -> None:
    result = classify_attempt(make_attempt(tmp_path / "attempt", summary=False))
    assert result["eligible"] is True


def test_unclassified_summary_error_payload_fails_closed(tmp_path: Path) -> None:
    result = classify_attempt(make_attempt(tmp_path / "attempt", summary_payload=None))
    # The helper creates an explicit generic detail payload for ordinary fixtures;
    # mutate just the structured terminal error record to be unclassifiable.
    trace = tmp_path / "attempt/relay/trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[-1]["answer"] = {"unexpected": "shape"}
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = classify_attempt(tmp_path / "attempt")
    assert result["eligible"] is False
    assert result["reason"] == "unclassified_summary_429_payload"


LIVE_ROOT = Path(os.environ.get("RELAY_PILOT_ROOT", "/nonexistent")) / "full-browse-20261030/results/gpt-6-luna"


@pytest.mark.parametrize("task_id,method", [
    ("786", "selfcompact"), ("885", "compaction"), ("420", "selfcompact"),
    ("18", "folding"), ("18", "rlm"),
])
def test_preserved_terminal_429_failure_fixtures(task_id: str, method: str) -> None:
    run_dir = LIVE_ROOT / task_id / method
    if not run_dir.exists():
        pytest.skip("Preserved Browse failure fixture is unavailable on this machine")
    result = classify_attempt(run_dir)
    assert result["eligible"] is True
    assert result["reason"] == "pure_terminal_http_429"


@pytest.mark.parametrize("task_id,method", [("885", "folding"), ("786", "rlm")])
def test_preserved_incidental_429_fixtures_are_not_eligible(task_id: str, method: str) -> None:
    run_dir = LIVE_ROOT / task_id / method
    if not run_dir.exists():
        pytest.skip("Preserved Browse negative fixture is unavailable on this machine")
    result = classify_attempt(run_dir)
    assert result["eligible"] is False
    assert result["reason"] == "429_was_incidental"


def _state(task_id: str, method: str, run_dir: Path, score: int | None) -> dict:
    return {"benchmark": "BrowseComp-Plus", "status": "complete", "protocol": {},
            "tasks": [{"task_id": task_id, "cells": [{
                "task_id": task_id, "method": method, "status": "finished",
                "raw_run": str(run_dir / "run.json"), "raw_score": score,
            }]}]}


def test_ledger_keeps_attempt_artifacts_and_nulls_only_retry_method_score(tmp_path: Path) -> None:
    first = make_attempt(tmp_path / "first")
    second = make_attempt(tmp_path / "second", statuses=(200, 429, 200))
    historical = tmp_path / "historical.json"
    current = tmp_path / "current.json"
    historical.write_text(json.dumps(_state("1", "acm", first, None)))
    current.write_text(json.dumps(_state("2", "clm", second, 1)))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"source_states": [str(historical)],
                                "source_state_sha256": {
                                    str(historical): hashlib.sha256(historical.read_bytes()).hexdigest()}}))
    first_grade = first / "browse-grade.json"
    first_grade.write_text('{"strict_answer_score":null}\n')
    original_grade_hash = hashlib.sha256(first_grade.read_bytes()).hexdigest()

    ledger_path = tmp_path / "retry-ledger.json"
    ledger = build_retry_ledger(plan, current, ledger_path)
    assert ledger["finished_cells_audited"] == 2
    assert ledger["retry_eligible"] == 1
    rows = {row["task_id"]: row for row in ledger["cells"]}
    assert rows["1"]["method_score"] is None
    assert rows["1"]["retry_eligible"] is True
    assert rows["2"]["method_score"] == 1
    assert rows["2"]["retry_eligible"] is False
    assert hashlib.sha256(first_grade.read_bytes()).hexdigest() == original_grade_hash
    assert json.loads(ledger_path.read_text())["inference_repeated"] is False
