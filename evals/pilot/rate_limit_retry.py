"""Offline audit for attempts terminated by a pure upstream HTTP 429.

This module only reads preserved artifacts. It never reads task questions,
answers, references, or model messages and never changes a score or attempt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


_RETRY_STOP_REASONS = {"context_management_failed", "http_429", "upstream_http_429"}
_ACCOUNT_MARKERS = (
    "account", "quota", "credit", "ordinary plan", "weekly limit", "monthly limit",
    "billing", "payment required", "insufficient_quota", "balance", "usage_limit",
    "plan_limit", "spend_limit",
)
_REFUSED = re.compile(
    r"upstream refused (?:(?:GET|POST|PUT|PATCH|DELETE|HEAD) )?(\S+)(?: \S+)? \((\d{3})\): (.*)$"
)
_TERMINAL_429 = re.compile(r"^(?:RuntimeError|HTTPStatusError|APIStatusError):.*\b429\b", re.I)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]] | None:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _response_rows(rows: list[dict[str, Any]]) -> list[tuple[str, int]] | None:
    responses = []
    for row in rows:
        if row.get("event") != "response_headers":
            continue
        path, status = row.get("path"), row.get("status")
        if not isinstance(path, str) or type(status) is not int:
            return None
        responses.append((path, status))
    return responses


def _payload_account_limited(payload_text: str) -> bool | None:
    """Return None when a 429 body cannot be safely identified."""
    try:
        value = ast.literal_eval(payload_text.strip())
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        obj = json.loads(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    # Inspect only error fields. Never return or log their text.
    error = obj.get("error", obj)
    return _error_account_limited(error)


def _error_account_limited(error: Any) -> bool | None:
    if isinstance(error, dict) and isinstance(error.get("error"), dict):
        error = error["error"]
    if not isinstance(error, dict) or not any(key in error for key in ("type", "code", "message", "detail")):
        return None
    detail = " ".join(str(error.get(key, "")) for key in ("type", "code", "message", "detail")).casefold()
    return any(marker in detail for marker in _ACCOUNT_MARKERS)


def _runtime_log_evidence(path: Path) -> tuple[str | None, bool, bool, int]:
    """Return terminal class, account limit, other errors, and response 429 refusals."""
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return None, False, True, 0
    terminal = lines[-1] if lines else ""
    terminal_lower = terminal.casefold()
    terminal_429 = bool(_TERMINAL_429.search(terminal) and
                        ("rate limit" in terminal_lower or "too many requests" in terminal_lower))
    account_limited = any(marker in terminal_lower for marker in _ACCOUNT_MARKERS)
    malformed_429 = False
    unrelated_refusal = False
    response_429_refusals = 0
    for line in lines:
        match = _REFUSED.search(line)
        if not match:
            continue
        endpoint, status_text, payload = match.groups()
        status = int(status_text)
        if status != 429 or not endpoint.endswith(("/responses", "/models")):
            unrelated_refusal = True
            continue
        if endpoint.endswith("/responses"):
            response_429_refusals += 1
        limited = _payload_account_limited(payload)
        if limited is None:
            malformed_429 = True
        elif limited:
            account_limited = True
    return ("terminal_429" if terminal_429 else None,
            account_limited, malformed_429 or unrelated_refusal, response_429_refusals)


def _fail(reason: str, evidence: dict[str, str]) -> dict[str, Any]:
    return {"eligible": False, "reason": reason, "evidence_sha256": evidence}


def classify_attempt(run_dir: Path) -> dict[str, Any]:
    """Classify one raw Browse or Science attempt without reading its answer.

    Browse accepts a raw worker directory. Science may pass a wrapper directory
    whose ``worker/`` child contains ``run.json`` and transport evidence.
    Eligible attempts must end in an evidenced upstream responses HTTP 429,
    with the terminal worker exception matching that 429 and no cap, quota,
    method, context-budget, sandbox, or unrelated HTTP failure.
    """
    run_dir = Path(run_dir).resolve()
    worker = run_dir / "worker" if (run_dir / "worker/run.json").is_file() else run_dir
    run_path = worker / "run.json"
    runtime_log = worker / "relay/runtime.log"
    upstream_path = worker / "upstream-events.jsonl"
    trace_path = worker / "relay/trace.jsonl"
    optional = [run_dir / "result.json", run_dir / "verifier.json", run_dir / "evaluation-audit.json",
                run_dir / "grade.json", run_dir / "browse-grade.json", run_dir / "score.json",
                worker / "browse-grade.json", worker / "codex.stderr", worker / "relay/events.jsonl"]
    required = [run_path, runtime_log, upstream_path, trace_path]
    evidence: dict[str, str] = {}
    if run_path.is_file():
        evidence[str(run_path.resolve())] = _sha256(run_path)
    else:
        return _fail("required_run_metadata_missing", evidence)
    try:
        run = json.loads(run_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _fail("invalid_run_metadata", evidence)
    if not isinstance(run, dict):
        return _fail("invalid_run_metadata", evidence)

    # A terminal 429 must be the actual failure label, not an incidental response.
    status, stop, failure = run.get("status"), run.get("stop_reason"), run.get("failure_kind")
    if stop in {"wall_time_limit", "model_request_limit"}:
        return _fail("local_execution_cap", evidence)
    if status != "stopped":
        return _fail("attempt_not_stopped", evidence)
    if stop not in _RETRY_STOP_REASONS or failure not in _RETRY_STOP_REASONS:
        return _fail("terminal_failure_not_attributed_to_429", evidence)
    if run.get("error") not in (None, ""):
        return _fail("worker_error_present", evidence)
    for field in ("strategy_errors", "sandbox_execution_errors", "requests_over_context_budget"):
        value = run.get(field)
        if type(value) is not int or value != 0:
            return _fail(f"{field}_present_or_unaccounted", evidence)
    quota_text = " ".join(str(run.get(key) or "") for key in
                          ("stop_reason", "failure_kind", "error", "quota_stop_reason")).casefold()
    if any(marker in quota_text for marker in ("quota", "credit", "ordinary_plan", "account_limit")):
        return _fail("account_or_quota_limit", evidence)
    if type(run.get("exit_code")) is not int or run["exit_code"] not in (0, 1, -9, -15):
        return _fail("unexpected_worker_exit", evidence)

    evidence_paths = [*required, *(path for path in optional if path.is_file())]
    if any(not path.is_file() for path in required):
        return _fail("required_transport_evidence_missing", evidence)
    evidence.update({str(path.resolve()): _sha256(path) for path in evidence_paths if path.is_file()})

    upstream = _jsonl(upstream_path)
    trace = _jsonl(trace_path)
    if upstream is None or trace is None:
        return _fail("invalid_transport_log", evidence)
    responses = _response_rows(upstream)
    if responses is None or not responses:
        return _fail("response_status_evidence_missing", evidence)
    endpoint_rows = [(path, code) for path, code in responses if path.endswith("/responses")]
    if not endpoint_rows:
        return _fail("responses_endpoint_evidence_missing", evidence)
    if any(code >= 500 for _, code in responses):
        return _fail("unrelated_5xx_present", evidence)
    if any(code >= 400 and not (code == 429 and path.endswith(("/responses", "/models")))
           for path, code in responses):
        return _fail("unrelated_http_error_present", evidence)
    if endpoint_rows[-1][1] != 429:
        return _fail("429_was_incidental", evidence)
    last_success = max((i for i, (_, code) in enumerate(endpoint_rows) if code == 200), default=-1)
    if not endpoint_rows[last_success + 1:] or any(code != 429 for _, code in endpoint_rows[last_success + 1:]):
        return _fail("429_not_terminal_suffix", evidence)

    summary_429 = [row for row in trace if isinstance(row.get("path"), str)
                 and row["path"].endswith("/responses") and row.get("summary_status") == 429]
    if any(row.get("summary_status") not in (200, 429)
                            for row in trace if isinstance(row.get("path"), str)
                            and row["path"].endswith("/responses")
                            and "summary_status" in row):
        return _fail("proxy_trace_does_not_confirm_terminal_429", evidence)
    last_summary = [row["summary_status"] for row in trace
                    if isinstance(row.get("path"), str) and row["path"].endswith("/responses")
                    and type(row.get("summary_status")) is int]
    # Direct main-call 429s have no summary_status. For those, the terminal
    # upstream response plus proxy refusal and terminal client exception below
    # provide the full evidence chain.
    if summary_429 and (not last_summary or last_summary[-1] != 429):
        return _fail("proxy_trace_429_was_incidental", evidence)
    if not summary_429 and endpoint_rows[-1][1] != 429:
        return _fail("proxy_trace_does_not_confirm_terminal_429", evidence)
    for row in summary_429:
        limited = _error_account_limited(row.get("answer"))
        if limited is None:
            return _fail("unclassified_summary_429_payload", evidence)
        if limited:
            return _fail("account_or_quota_limit_response", evidence)

    terminal_class, account_limited, malformed_or_other, response_429_refusals = _runtime_log_evidence(runtime_log)
    if account_limited:
        return _fail("account_or_quota_limit_response", evidence)
    if malformed_or_other:
        return _fail("unclassified_or_unrelated_proxy_error", evidence)
    if terminal_class != "terminal_429":
        return _fail("terminal_exception_not_http_429", evidence)
    if not summary_429 and response_429_refusals < 1:
        return _fail("main_429_lacks_proxy_refusal_evidence", evidence)
    return {"eligible": True, "reason": "pure_terminal_http_429", "evidence_sha256": evidence}


def build_retry_ledger(scale_plan_path: Path, current_state_path: Path,
                       output_path: Path) -> dict[str, Any]:
    """Audit historical and current finished cells; write an immutable-derived ledger."""
    plan = json.loads(Path(scale_plan_path).read_text())
    current_state_path = Path(current_state_path).resolve()
    current_before = _sha256(current_state_path)
    states = [*plan["source_states"], str(current_state_path)]
    seen: set[tuple[str, str]] = set()
    ledger = []
    for source_state in states:
        state_path = Path(source_state).resolve()
        state_hash = _sha256(state_path)
        state = json.loads(state_path.read_text())
        expected_hash = plan.get("source_state_sha256", {}).get(source_state)
        if expected_hash and expected_hash != state_hash:
            raise ValueError(f"Historical state changed: {state_path}")
        for task in state.get("tasks", []):
            for cell in task.get("cells", []):
                if cell.get("status") != "finished":
                    continue
                key = (str(cell["task_id"]), str(cell["method"]))
                if key in seen:
                    raise ValueError(f"Duplicate finished cell across frozen source states: {key}")
                seen.add(key)
                raw_run = Path(cell["raw_run"]).resolve()
                result = classify_attempt(raw_run.parent)
                ledger.append({
                    "task_id": key[0], "method": key[1], "source_state": str(state_path),
                    "source_state_sha256": state_hash, "original_run_dir": str(raw_run.parent),
                    "original_run_sha256": result["evidence_sha256"].get(str(raw_run)),
                    "original_raw_score": cell.get("raw_score"),
                    "method_score": None if result["eligible"] else cell.get("raw_score"),
                    "retry_eligible": result["eligible"],
                    "retry_reason": result["reason"], "evidence_sha256": result["evidence_sha256"],
                })
    if _sha256(current_state_path) != current_before:
        raise ValueError("Current batch state changed during the read-only audit")
    ledger.sort(key=lambda row: (not row["task_id"].isdigit(),
                                 int(row["task_id"]) if row["task_id"].isdigit() else row["task_id"],
                                 row["method"]))
    output = {"version": 1, "source_states": states, "finished_cells_audited": len(ledger),
              "retry_eligible": sum(row["retry_eligible"] for row in ledger),
              "current_state_sha256": current_before,
              "method_score_policy": "null for every attempt selected for retry; original score artifacts remain unchanged",
              "inference_repeated": False, "cells": ledger}
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    return output
