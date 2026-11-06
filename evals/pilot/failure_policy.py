"""Narrow continuation policy for preserved, unscored worker failures."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


UPSTREAM_STREAM_INTERRUPTED = "upstream_stream_interrupted"
AUXILIARY_CAPACITY_EXHAUSTION = "auxiliary_capacity_exhaustion"
PERSISTENT_SESSION_RESET = "rlm_persistent_session_reset"
REVIEWED_TERMINAL_REASONS = frozenset({AUXILIARY_CAPACITY_EXHAUSTION, PERSISTENT_SESSION_RESET})
REVIEW_EVIDENCE = (
    "worker/run.json", "worker/relay/runtime.log", "worker/relay/trace.jsonl",
    "worker/upstream-events.jsonl",
)
_INCOMPLETE_BODY = "peer closed connection without sending complete message body (incomplete chunked read)"
_STRATEGY_FAILURE = re.compile(r"(?m)^([a-zA-Z0-9_]+) failed; forwarding the last context$")
_EXCEPTION = re.compile(r"(?m)^[\w.]+(?:Error|Exception): .*$")


def upstream_stream_interrupted(worker: dict[str, Any], runtime_log: Path) -> bool:
    """Require raw metadata and matching tracebacks; never authorize a retry or score.

    A saved label alone is insufficient. Missing counters, mixed failures, an
    unfinished worker, or any quota/context/sandbox stop fail closed.
    """
    if worker.get("status") not in {"completed", "stopped"} or worker.get("protocol_valid") is not False:
        return False
    if worker.get("error") not in (None, "") or worker.get("failure_kind") not in (None, ""):
        return False
    reason = worker.get("stop_reason")
    if reason not in (None, "", "wall_time_limit", "model_request_limit"):
        return False
    if type(worker.get("exit_code")) is not int:
        return False
    if reason in (None, "") and worker["exit_code"] != 0:
        return False
    if worker["status"] == "completed" and (reason or worker["exit_code"] != 0):
        return False
    if worker["status"] == "stopped" and (not reason or worker["exit_code"] not in (0, 1, -9, -15)):
        return False
    for field in ("requests_over_context_budget", "sandbox_execution_errors"):
        if type(worker.get(field)) is not int or worker[field] != 0:
            return False
    errors, window = worker.get("strategy_errors"), worker.get("window")
    if type(errors) is not int or errors < 1 or type(window) is not int or window < 1:
        return False
    sizes = worker.get("provider_input_tokens_per_request")
    if not isinstance(sizes, list) or not sizes or any(type(n) is not int or not 0 < n <= window for n in sizes):
        return False
    strategy = worker.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        return False
    try:
        log = runtime_log.read_text()
    except (OSError, UnicodeError):
        return False
    failures = list(_STRATEGY_FAILURE.finditer(log))
    if len(failures) != errors or log[:failures[0].start()].strip():
        return False
    allowed = {f"{name}: {_INCOMPLETE_BODY}" for name in ("httpcore.RemoteProtocolError", "httpx.RemoteProtocolError")}
    for index, failure in enumerate(failures):
        end = failures[index + 1].start() if index + 1 < len(failures) else len(log)
        block = log[failure.end():end].strip()
        if (failure.group(1) != strategy or not block.startswith("Traceback (most recent call last):")
                or not block.endswith(f"httpx.RemoteProtocolError: {_INCOMPLETE_BODY}")):
            return False
        exceptions = _EXCEPTION.findall(block)
        if not exceptions or any(exception not in allowed for exception in exceptions):
            return False
    return True


def _capacity_error(value: Any) -> bool:
    return isinstance(value, dict) and value.get("error") == {
        "type": "rate_limit_error", "code": "turn_rate_limit_exceeded",
        "message": "Codex is at capacity, please try again later",
    }


def _without_capacity_warnings(log: str) -> str | None:
    """Remove only the exact recovered native capacity-warning shape."""
    lines = []
    for line in log.splitlines():
        if line.startswith("upstream refused "):
            match = re.fullmatch(r"upstream refused /backend-api/codex/responses \(429\): (b.+)", line)
            if not match or not _capacity_error(json.loads(ast.literal_eval(match.group(1)))):
                return None
        else:
            lines.append(line)
    return "\n".join(lines)


def _capacity_tracebacks(log: str, strategy: str, errors: int) -> bool:
    """Every strategy exception must be the same evidenced auxiliary 429."""
    log = _without_capacity_warnings(log)
    if log is None:
        return False
    failures = list(_STRATEGY_FAILURE.finditer(log))
    if len(failures) != errors or log[:failures[0].start()].strip():
        return False
    prefix = "RuntimeError: the model request failed with HTTP 429: "
    for index, failure in enumerate(failures):
        end = failures[index + 1].start() if index + 1 < len(failures) else len(log)
        block = log[failure.end():end].strip()
        exceptions = _EXCEPTION.findall(block)
        if (failure.group(1) != strategy or len(exceptions) != 1
                or not block.startswith("Traceback (most recent call last):")
                or not block.endswith(exceptions[0]) or not exceptions[0].startswith(prefix)
                or "/relay/core/engine.py" not in block or "in complete" not in block
                or not _capacity_error(ast.literal_eval(exceptions[0][len(prefix):]))):
            return False
    return True


def reviewed_terminal_failure(attempt_root: Path) -> str | None:
    """Validate a controller review; authorize only later cells, never a score or retry.

    Worker and scoring artifacts remain immutable. A specific invalid evaluation
    audit must agree with the review, whose hashes bind all raw failure evidence.
    Persistent-session exclusions additionally require the reviewed source repair
    to be installed before any continuation.
    """
    try:
        review = json.loads((attempt_root / "continuation-audit.json").read_text())
        audit = json.loads((attempt_root / "evaluation-audit.json").read_text())
        if not isinstance(review, dict) or not isinstance(audit, dict):
            return None
        reason = review.get("reason")
        if (type(review.get("version")) is not int or review["version"] != 1 or reason not in REVIEWED_TERMINAL_REASONS
                or "score" not in review or review["score"] is not None
                or review.get("continue_without_retry") is not True
                or audit.get("valid") is not False or audit.get("classification") != "infrastructure_failed"
                or audit.get("reason") != reason or audit.get("benchmark_score") is not None):
            return None
        hashes = review.get("evidence_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(REVIEW_EVIDENCE):
            return None
        contents = {name: (attempt_root / name).read_bytes() for name in REVIEW_EVIDENCE}
        if any(hashes[name] != hashlib.sha256(value).hexdigest() for name, value in contents.items()):
            return None
        if any(not contents[name] for name in REVIEW_EVIDENCE if not name.endswith("runtime.log")):
            return None
        worker = json.loads(contents["worker/run.json"])
        if not isinstance(worker, dict):
            return None
        trace = [json.loads(line) for line in contents["worker/relay/trace.jsonl"].splitlines()]
        upstream = [json.loads(line) for line in contents["worker/upstream-events.jsonl"].splitlines()]
        if (not trace or not upstream or any(not isinstance(row, dict) for row in trace + upstream)
                or not any("summary_request" in row or "body" in row for row in trace)
                or not any(row.get("event") in {"request", "response_headers"}
                           and str(row.get("path", "")).endswith("/responses") for row in upstream)):
            return None
        failures = " ".join(str(worker.get(key) or "") for key in ("stop_reason", "error", "failure_kind"))
        if any(term in failures.lower() for term in ("quota", "credit", "ordinary_plan", "account_limit")):
            return None
        if any(type(worker.get(key)) is not int or worker[key] != 0
               for key in ("requests_over_context_budget", "sandbox_execution_errors")):
            return None
        errors, window, sizes = worker.get("strategy_errors"), worker.get("window"), worker.get("provider_input_tokens_per_request")
        if (type(errors) is not int or errors < 0 or type(window) is not int or window < 1
                or not isinstance(sizes, list) or not sizes
                or any(type(size) is not int or not 0 < size <= window for size in sizes)):
            return None
        status, stop, code = worker.get("status"), worker.get("stop_reason"), worker.get("exit_code")
        ordinary_terminal = type(code) is int and (
            (status == "completed" and not stop and code == 0)
            or (status == "stopped" and stop in {"wall_time_limit", "model_request_limit"}
                and code in (0, 1, -9, -15)))
        controller_stopped = (review.get("controller_stopped") is True and status == "infrastructure_failed"
                              and type(code) is int and code in (-9, -15) and not stop
                              and worker.get("error") == f"harness_exit_{code}")
        if (not (ordinary_terminal or (reason == PERSISTENT_SESSION_RESET and controller_stopped))
                or worker.get("failure_kind") not in (None, "")
                or (not controller_stopped and worker.get("error") not in (None, ""))):
            return None
        log = contents["worker/relay/runtime.log"].decode()
        if reason == PERSISTENT_SESSION_RESET:
            cleaned_log = _without_capacity_warnings(log)
            if (worker.get("strategy") != "rlm_persistent" or errors or cleaned_log is None or cleaned_log.strip()
                    or worker.get("protocol_valid") is not (False if controller_stopped else True)):
                return None
            repaired_hash = review.get("reviewed_repaired_rlm_sha256")
            source = Path(__file__).resolve().parents[2] / "relay/strategies/rlm.py"
            if repaired_hash != hashlib.sha256(source.read_bytes()).hexdigest():
                return None
        else:
            if (worker.get("protocol_valid") is not False or errors < 1
                    or not _capacity_tracebacks(log, str(worker.get("strategy") or ""), errors)):
                return None
            if not any(row.get("summary_status") == 429 and "summary_request" in row
                       and _capacity_error(row.get("answer")) for row in trace):
                return None
            if not any(row.get("event") == "response_headers" and row.get("status") == 429
                       and str(row.get("path", "")).endswith("/responses") for row in upstream):
                return None
        return reason
    except (OSError, ValueError, SyntaxError, TypeError, AttributeError):
        return None
