"""One outcome policy for every benchmark's ended cells.

A cell ends in one of three outcomes:

- ``scored``: the benchmark's own grader produced a score.
- ``task_failure``: the run ended without a score for a reason that belongs to the run
  itself (the method raised an error, the model's output was cut at ``max_output_tokens``,
  the provider refused the content the method sent, or the verifier timed out on the
  submitted artifact). It counts as a score of 0.
- ``infrastructure``: the run ended without a score because the service around it failed
  (the upstream dropped a connection, rate limits outlasted the retries, a quota guard
  stopped the run, or the input counter could not account for a request). It is excluded
  from scores and reported as a count.

Causes are read from the run's own records (``run.json``, Relay's runtime log and the
harness output), never from its score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

INFRASTRUCTURE = {"upstream_connection_lost", "rate_limited", "upstream_error", "quota_stop",
                  "input_accounting", "worker_infrastructure"}
TASK_FAILURE = {"provider_refusal", "max_output_tokens", "verifier_timeout", "method_error"}

_EXCEPTION = re.compile(r"^(?:httpx\.)?(\w+(?:Error|Exception|Timeout)): ?(.*)$", re.M)
_TRANSPORT = {"RemoteProtocolError", "ReadError", "WriteError", "ConnectError", "ConnectTimeout",
              "ReadTimeout", "WriteTimeout", "PoolTimeout", "NetworkError", "CloseError"}


def _text(path: Path, tail: int = 200_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - tail))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def failure_cause(worker: Path, run: dict[str, Any], *, verifier_timeout: bool = False) -> str:
    """Why a run that has no score ended; `worker` holds run.json, relay/ and codex output."""
    stop = str(run.get("stop_reason") or "")
    if stop.startswith("quota") or run.get("failure_kind") == "quota_stopped":
        return "quota_stop"
    if stop == "input_accounting_failed":
        return "input_accounting"
    if verifier_timeout:
        return "verifier_timeout"
    if stop == "upstream_request_timeout":
        return "upstream_connection_lost"
    log = _text(worker / "relay/runtime.log")
    harness = _text(worker / "codex.jsonl") + _text(worker / "codex.stderr")
    errors = _EXCEPTION.findall(log)
    if errors:
        kind, message = errors[-1]
        if kind in _TRANSPORT:
            return "upstream_connection_lost"
        if kind == "RuntimeError" and ("HTTP 429" in message or "Rate limit" in message):
            return "rate_limited"
        if kind == "RuntimeError" and re.search(r"HTTP 5\d\d", message):
            return "upstream_error"
    if "flagged" in harness and "Invalid prompt" in harness:
        return "provider_refusal"
    if "max_output_tokens" in harness:
        return "max_output_tokens"
    if stop in ("context_management_failed", "strategy_error") or errors:
        return "method_error"
    return "worker_infrastructure"


def outcome(scored: bool, cause: str | None) -> str:
    if scored:
        return "scored"
    return "infrastructure" if cause in INFRASTRUCTURE else "task_failure"


def fallback_counts(run: dict[str, Any]) -> dict[str, int]:
    """How often Relay's common overflow fallback changed a request in this run.

    Main-request truncations drop whole old tool groups (or shorten old text) after the
    method's own forced recovery could not fit the request; summary truncations shorten
    the method's own auxiliary request."""
    main = summary = exhausted = 0
    for event in run.get("overflow_fallback_events") or []:
        if event.get("action") == "method_recovery_exhausted":
            exhausted += 1
        elif event.get("dropped_items") or event.get("truncated_text_items"):
            if event.get("scope") == "main":
                main += 1
            else:
                summary += 1
    return {"main_truncations": main, "auxiliary_truncations": summary, "recovery_exhausted": exhausted}


def browse_profile_applied(worker: Path) -> bool | None:
    """Whether the first forwarded request carried the Browse profile instead of Codex's prompt."""
    try:
        with (worker / "relay/trace.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                body = record.get("forwarded") or record.get("body")
                if not isinstance(body, dict) or "input" not in body:
                    continue
                texts = [part.get("text", "") for item in body["input"] if item.get("role") == "developer"
                         for part in (item.get("content") if isinstance(item.get("content"), list) else [])
                         if isinstance(part, dict)]
                return not any(text.startswith("You are Codex") for text in texts)
    except (OSError, ValueError):
        return None
    return None
