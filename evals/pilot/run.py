"""Run a bounded Codex worker through an isolated, quota-guarded Relay.

This is experimental evaluation plumbing, not a modification of Relay's methods.
Use the external pilot environment's Python. No model API keys are supported.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

MODEL = "gpt-6-luna"
MODELS = ("gpt-6-luna", "gpt-6.1-sol")
REPO = Path(__file__).resolve().parents[2]
WEEKLY_RESET_TOLERANCE_SECONDS = 5


@dataclass
class RequestAccounting:
    """One admitted task request, with upstream evidence collected across threads."""

    main_requests: int = 0
    main_statuses: list[int] = field(default_factory=list)
    auxiliary_requests: int = 0
    auxiliary_statuses: list[int] = field(default_factory=list)
    client_status: int | None = None
    account_rejected: bool = False
    other_guard_rejected: bool = False
    finalized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_REQUEST_ACCOUNTING: contextvars.ContextVar[RequestAccounting | None] = contextvars.ContextVar(
    "pilot_request_accounting", default=None
)
_ACCOUNT_REJECTION_REASONS = {
    "ordinary_plan_usage_not_allowed", "plan_quota_reserve_reached", "account_limit_reached",
    "credit_balance_decreased", "quota_window_changed", "insufficient_quota",
    "usage_limit_reached", "plan_limit_reached", "spend_limit_reached",
}


def same_weekly_reset(observed: Any, expected: int | str) -> bool:
    """Allow timestamp rounding around the frozen anchor, never a rolling reset.

    The account service has reported adjacent seconds for the same weekly
    window. Keep the original baseline and thresholds; a real new week must
    still stop the experiment.
    """
    if type(observed) is not int:
        return False
    try:
        return abs(observed - int(expected)) <= WEEKLY_RESET_TOLERANCE_SECONDS
    except (ValueError, TypeError):
        return False


def source_fingerprint() -> dict[str, Any]:
    """Identify uncommitted evaluator and method fixes used by each new worker."""
    paths = sorted([*REPO.glob("relay/**/*.py"), *REPO.glob("evals/pilot/*.py")])
    hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "file_count": len(hashes), "files": hashes}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def closed_admission(out: Path, root: Path) -> Path | None:
    """A controller can drain one experiment tree without killing active workers."""
    if root != out and root not in out.parents:
        return None
    for parent in (out, *out.parents):
        marker = parent / ".stop-new-workers"
        if marker.is_file():
            return marker
        if parent == root:
            break
    return None


class CodexRPC:
    """Read-only account metadata; never starts an inference turn."""

    def __init__(self, env: dict[str, str]) -> None:
        self.process = subprocess.Popen(
            ["codex", "app-server"], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.serial = 0
        self.buffer = b""
        self.lock = threading.Lock()
        try:
            self.call("initialize", {"clientInfo": {"name": "relay-pilot-quota", "version": "0.1"}})
            self._send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def _send(self, value: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(value).encode() + b"\n")
        self.process.stdin.flush()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            self.serial += 1
            self._send({"id": self.serial, "method": method, "params": params or {}})
            deadline = time.monotonic() + 20
            assert self.process.stdout is not None
            while time.monotonic() < deadline:
                while b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    message = json.loads(line)
                    if message.get("id") == self.serial:
                        if "error" in message:
                            raise RuntimeError(f"{method}: {message['error']}")
                        return message["result"]
                if select.select([self.process.stdout], [], [], 0.5)[0]:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError("quota metadata server exited")
                    self.buffer += chunk
            raise TimeoutError(f"{method} timed out; inference must stop")

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class QuotaGate:
    def __init__(self, rpc: CodexRPC, path: Path, max_requests: int, start: float,
                 input_limit: int | None = None) -> None:
        self.rpc, self.path = rpc, path
        self.max_requests, self.start = max_requests, start
        self.requests = 0  # charged requests plus active reservations; refunded 429s release a slot
        self.inflight_requests = 0
        self.raw_request_attempts = 0
        self.provider_main_429_responses = 0
        self.provider_auxiliary_429_responses = 0
        self.request_count_refunds = 0
        self.last_checked = 0.0
        self.balance: Decimal | None = None
        self.quota_mode = os.environ.get("RELAY_PILOT_QUOTA_MODE", "poll")
        if self.quota_mode not in {"poll", "account_enforced"}:
            raise ValueError("Unknown quota policy")
        if (self.quota_mode == "account_enforced"
                and os.environ.get("RELAY_PILOT_ACCOUNT_NO_OVERAGE_CONFIRMED") != "1"):
            raise ValueError("Account-enforced mode requires confirmed no-overage account settings")
        self.used: float | None = None if self.quota_mode == "account_enforced" else 100.0
        self.reason: str | None = None
        self.lock = threading.Lock()
        self.input_limit = input_limit
        self.start_percent = float(os.environ.get("RELAY_PILOT_QUOTA_START_PERCENT", "80"))
        self.stop_percent = float(os.environ.get("RELAY_PILOT_QUOTA_STOP_PERCENT", "85"))
        if not 0 < self.start_percent <= self.stop_percent < 100:
            raise ValueError("Quota thresholds must satisfy 0 < start <= stop < 100")
        self.weekly_reset = os.environ.get("RELAY_PILOT_QUOTA_WEEKLY_RESET")
        self.all_call_budget = os.environ.get("RELAY_PILOT_ALL_CALL_BUDGET") == "1"
        self.expected_effort = os.environ.get("RELAY_PILOT_EXPECTED_EFFORT")
        self.auxiliary_inputs: list[int] = []
        self.input_budget: Any = None  # set by the Relay setup, shared with every dispatch guard

    @property
    def dispatch_input_limit(self) -> int | None:
        return self.input_budget.limit_for({}) if self.input_budget else self.input_limit

    def record_raw_request_attempt(self) -> None:
        with self.lock:
            self.raw_request_attempts += 1

    def admit_task_request(self) -> bool:
        """Reserve one task-request slot atomically; caller finalizes it after Relay responds."""
        if not self.check():
            return False
        with self.lock:
            if self.reason:
                return False
            if self.requests >= self.max_requests:
                # A full set of active reservations may still include a 429
                # that will be refunded when its response finishes.
                if self.inflight_requests == 0:
                    self.reason = "model_request_limit"
                return False
            self.requests += 1
            self.inflight_requests += 1
            return True

    def upstream_request_started(self, kind: str) -> None:
        request = _REQUEST_ACCOUNTING.get()
        if request is None:
            return
        with request.lock:
            if kind == "main":
                request.main_requests += 1
            elif kind == "auxiliary":
                request.auxiliary_requests += 1
            else:
                raise ValueError(f"unknown upstream request kind: {kind}")

    def upstream_response_seen(self, kind: str, status: int) -> None:
        request = _REQUEST_ACCOUNTING.get()
        if request is None:
            return
        with request.lock:
            statuses = request.main_statuses if kind == "main" else request.auxiliary_statuses
            statuses.append(status)
        if status == 429:
            with self.lock:
                if kind == "main":
                    self.provider_main_429_responses += 1
                else:
                    self.provider_auxiliary_429_responses += 1

    @staticmethod
    def mark_account_rejection() -> None:
        request = _REQUEST_ACCOUNTING.get()
        if request is not None:
            with request.lock:
                request.account_rejected = True

    @staticmethod
    def mark_other_guard_rejection() -> None:
        request = _REQUEST_ACCOUNTING.get()
        if request is not None:
            with request.lock:
                request.other_guard_rejected = True

    def finish_task_request(self, request: RequestAccounting) -> None:
        """Refund only provider 429 work that did not produce a task generation."""
        with self.lock:
            quota_reason = self.reason in _ACCOUNT_REJECTION_REASONS
        with request.lock:
            if request.finalized:
                return
            request.finalized = True
            quota_rejected = request.account_rejected or quota_reason
            other_guard_rejected = request.other_guard_rejected
            main_count = request.main_requests
            main_statuses = tuple(request.main_statuses)
            auxiliary_count = request.auxiliary_requests
            auxiliary_statuses = tuple(request.auxiliary_statuses)
            client_status = request.client_status
        if not quota_rejected and not other_guard_rejected:
            main_429 = (main_count > 0 and len(main_statuses) == main_count
                        and all(status == 429 for status in main_statuses))
            auxiliary_only_429 = (client_status is not None and client_status >= 400
                                  and main_count == 0 and auxiliary_count > 0
                                  and len(auxiliary_statuses) == auxiliary_count
                                  and all(status == 429 for status in auxiliary_statuses))
            if main_429 or auxiliary_only_429:
                with self.lock:
                    self.requests = max(0, self.requests - 1)
                    self.request_count_refunds += 1
        with self.lock:
            self.inflight_requests = max(0, self.inflight_requests - 1)

    def observe_auxiliary_tokens(self, tokens: int | None) -> None:
        if tokens is None:
            return
        with self.lock:
            self.auxiliary_inputs.append(tokens)
            if (self.all_call_budget and self.dispatch_input_limit is not None
                    and tokens > self.dispatch_input_limit and self.reason is None):
                self.reason = ("provider_input_budget_mismatch" if self.input_budget else
                               "auxiliary_context_budget_exceeded")

    def observe_input_tokens(self, tokens: int | None) -> None:
        """Stop further spend once actual provider usage breaks the declared window."""
        if tokens is not None and self.dispatch_input_limit is not None and tokens > self.dispatch_input_limit:
            with self.lock:
                if self.reason is None:
                    self.reason = "provider_input_budget_mismatch" if self.input_budget else "context_budget_exceeded"

    def check(self, *, force: bool = False, starting: bool = False) -> bool:
        with self.lock:
            if self.reason:
                return False
            # The operator has confirmed (RELAY_PILOT_ACCOUNT_NO_OVERAGE_CONFIRMED) that the
            # account rejects paid overage. This mode makes no account usage calls; model and
            # input guards remain.
            if self.quota_mode == "account_enforced":
                return True
            if not force and time.monotonic() - self.last_checked < 10:
                return True
            try:
                result = self.rpc.call("account/rateLimits/read")
                limits = result["rateLimits"]
                windows = [limits.get(key) for key in ("primary", "secondary")]
                self.used = max(float(w["usedPercent"]) for w in windows if w)
                if self.weekly_reset:
                    weekly = [w for w in windows if w and w.get("windowDurationMins") == 10080]
                    if len(weekly) != 1 or not same_weekly_reset(weekly[0].get("resetsAt"), self.weekly_reset):
                        self.reason = "quota_window_changed"
                    else:
                        self.used = float(weekly[0]["usedPercent"])
                credits = limits.get("credits") or {}
                balance = Decimal(str(credits["balance"])) if credits.get("balance") is not None else None
                if not result.get("ordinaryUsageAllowed", False):
                    self.reason = "ordinary_plan_usage_not_allowed"
                elif self.used >= (self.start_percent if starting else self.stop_percent):
                    self.reason = "plan_quota_reserve_reached"
                elif limits.get("spendControlReached") or limits.get("rateLimitReachedType"):
                    self.reason = "account_limit_reached"
                elif self.balance is not None and balance is not None and balance < self.balance:
                    self.reason = "credit_balance_decreased"
                if self.balance is None:
                    self.balance = balance
                # Preserve window identity for cost reviews without logging the
                # account response, credit balance, or other private metadata.
                public_windows = {
                    name: {key: window[key] for key in ("usedPercent", "windowDurationMins", "resetsAt")
                           if key in window}
                    for name in ("primary", "secondary")
                    if (window := limits.get(name))
                }
                record = {"checked_unix": time.time(),
                          "elapsed_seconds": time.monotonic() - self.start,
                          "used_percent": self.used,
                          "windows": public_windows,
                          "ordinary_usage_allowed": result.get("ordinaryUsageAllowed"),
                          "credit_balance_unchanged": balance == self.balance,
                          "stop_reason": self.reason}
                with self.path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                self.last_checked = time.monotonic()
            except Exception as exc:
                self.reason = f"quota_check_failed:{type(exc).__name__}"
            return self.reason is None


def config(home: Path, work: Path, root: Path, readable: list[Path],
           model: str = MODEL, denied: list[Path] | None = None) -> None:
    # Canonical paths matter: a project root may also be reachable through a symlink.
    extra_denied = json.loads(os.environ.get("RELAY_PILOT_EXTRA_DENY_PATHS", "[]"))
    if not isinstance(extra_denied, list) or any(
            not isinstance(path, str) or not Path(path).is_absolute() for path in extra_denied):
        raise ValueError("Additional hidden paths must be a JSON list of absolute paths")
    deny = [root, REPO.resolve(), Path.home() / ".codex-hosts",
            Path.home() / ".codex/auth.json", Path.home() / ".codex/config.toml",
            Path.home() / ".codex/sessions", Path.home() / ".ssh",
            Path.home() / ".cache/huggingface", Path("/tmp/relay-bcp-183-private"),
            *(denied or []), *(Path(path) for path in extra_denied)]
    # Redundant denied mounts below a hidden root require mkdir inside a
    # read-only mount. Keep a nested denial only when a readable/writable
    # exception between it and the ancestor makes that denial necessary.
    deny = list(dict.fromkeys(path.resolve() for path in deny))
    exceptions = [path.resolve() for path in (*readable, home / "tmp", work)]
    deny = [path for path in deny if not any(
        parent != path and path.is_relative_to(parent)
        and not any(allow.is_relative_to(parent) and path.is_relative_to(allow)
                    for allow in exceptions)
        for parent in deny)]
    fs = {":root": "read", **{str(p.resolve()): "deny" for p in deny},
          **{str(p.resolve()): "read" for p in readable},
          str(home / "tmp"): "write", str(work): "write"}
    text = (
        f'model = "{model}"\nmodel_reasoning_effort = "medium"\n'
        'forced_login_method = "chatgpt"\napproval_policy = "never"\n'
        'default_permissions = "pilot"\nweb_search = "disabled"\n'
        'project_doc_max_bytes = 0\nmodel_context_window = 1000000000\n'
        'model_auto_compact_token_limit = 1000000000\n'
        '[features]\nuse_legacy_landlock = false\nmulti_agent = false\n'
        '[permissions.pilot.filesystem]\n'
        + "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in fs.items())
        + '[permissions.pilot.network]\nenabled = true\n'
    )
    (home / "config.toml").write_text(text)


def metrics(out: Path, home: Path, window: int) -> dict[str, Any]:
    usage: list[dict[str, Any]] = []
    compacted = 0
    sandbox_errors = 0
    for path in home.glob("sessions/**/rollout-*.jsonl"):
        content = path.read_text()
        sandbox_errors += content.count("bwrap: execvp")
        for line in content.splitlines():
            event = json.loads(line)
            compacted += event.get("type") == "compacted"
            payload = event.get("payload") or {}
            if payload.get("type") == "token_count" and payload.get("info"):
                last = payload["info"].get("last_token_usage") or {}
                total = payload["info"].get("total_token_usage") or {}
                usage.append({"last": last, "total": total})
    sizes: list[int] = []
    auxiliary_sizes: list[int] = []
    changes: list[dict[str, Any]] = []
    summaries = 0
    trace = out / "relay/trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            if row.get("usage") is not None:
                sizes.append(row["usage"])
            summaries += "summary_request" in row
            answer = row.get("answer")
            if "summary_request" in row and isinstance(answer, dict):
                auxiliary_usage = answer.get("usage")
                if isinstance(auxiliary_usage, dict) and auxiliary_usage.get("input_tokens") is not None:
                    auxiliary_sizes.append(int(auxiliary_usage["input_tokens"]))
    events = out / "relay/events.jsonl"
    if events.exists():
        changes = [json.loads(line) for line in events.read_text().splitlines()]
    relay_log = out / "relay/runtime.log"
    errors = relay_log.read_text().count("failed; forwarding the last context") if relay_log.exists() else 0
    over_budget = sum(n > window for n in sizes)
    return {"provider_input_tokens_per_request": sizes,
            "peak_provider_input_tokens": max(sizes, default=0),
            "requests_over_context_budget": over_budget,
            "context_budget_scope": "main_agent_requests",
            "auxiliary_peak_input_tokens": max(auxiliary_sizes, default=0),
            "auxiliary_requests_above_main_window": sum(n > window for n in auxiliary_sizes),
            "strategy_errors": errors,
            "sandbox_execution_errors": sandbox_errors,
            "protocol_valid": bool(sizes) and not over_budget and not errors and not sandbox_errors,
            "context_changes": len(changes), "summary_requests": summaries,
            "codex_native_compactions": compacted,
            "codex_final_total_usage": usage[-1]["total"] if usage else None,
            "usage_note": "Window compliance concerns main-agent requests. Auxiliary requests use the same model's provider window and are measured separately. Codex totals exclude their usage; raw Relay traces preserve it."}


def harness_failure(out: Path) -> str | None:
    """An aborted provider/harness turn is not a scored benchmark failure."""
    path = out / "codex.jsonl"
    if not path.is_file():
        return None
    failure = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.failed":
            error = event.get("error") or {}
            failure = str(error.get("message", error) if isinstance(error, dict) else error)[:500]
    return failure


def check_upstream_request(request: Any, gate: QuotaGate, model: str) -> None:
    """Guard every inference request, including strategy-owned model calls."""
    from relay.transport.proxy import _decompress

    if request.method != "POST":
        return
    if request.url.host != "chatgpt.com":
        raise RuntimeError("Only the ChatGPT plan endpoint is permitted")
    body = json.loads(_decompress(request.content, request.headers.get("content-encoding", "")))
    if body.get("model") != model:
        raise RuntimeError("Unexpected model; no fallback permitted")
    if gate.expected_effort and (body.get("reasoning") or {}).get("effort") != gate.expected_effort:
        gate.reason = "model_reasoning_effort_mismatch"
        raise RuntimeError(gate.reason)
    if gate.input_budget is not None:
        from relay.protocols import OpenAIResponses

        # A local rejection is recoverable. Do not poison the quota gate until
        # the proxy has exhausted the method's declared recovery path.
        gate.input_budget.check(OpenAIResponses(), body, "upstream_dispatch")
    if not gate.check():
        if gate.reason in _ACCOUNT_REJECTION_REASONS:
            gate.mark_account_rejection()
        raise RuntimeError(gate.reason)


def observe_proxy_error(gate: QuotaGate, content: bytes, status: int | None = None) -> None:
    """Stop outer-harness retries only after local recovery has been exhausted."""
    try:
        error = json.loads(content).get("error", {})
    except (ValueError, AttributeError):
        return
    if not isinstance(error, dict):
        return
    if status == 429:
        code = str(error.get("code") or error.get("type") or "").casefold()
        message = str(error.get("message") or "").casefold()
        if code in _ACCOUNT_REJECTION_REASONS or any(
                marker in f"{code} {message}"
                for marker in ("quota", "credit", "plan_limit", "usage_limit", "spend_limit", "billing")):
            gate.mark_account_rejection()
    if error.get("source") == "relay_input_budget":
        gate.reason = gate.reason or "context_budget_exhausted"
    elif error.get("code") in {"context_management_failed", "unsupported_budget_request"}:
        gate.reason = gate.reason or "context_management_failed"
    elif error.get("code") in {"upstream_request_timeout", "input_accounting_failed"}:
        gate.reason = gate.reason or error["code"]
    elif error.get("code") in {"insufficient_quota", "usage_limit_reached",
                                "plan_limit_reached", "spend_limit_reached"}:
        gate.reason = gate.reason or "account_limit_reached"


def guarded_relay_app(gate: QuotaGate, inner: Any) -> Any:
    """Apply the task-request cap and reconcile each admitted request after Relay responds."""
    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        accounting = None
        accounting_token = None
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"].endswith("/responses"):
            gate.record_raw_request_attempt()
            if not await asyncio.to_thread(gate.admit_task_request):
                content = json.dumps({"error": {"message": gate.reason or "request_slot_temporarily_reserved"}}).encode()
                await send({"type": "http.response.start", "status": 429,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": content})
                return
            accounting = RequestAccounting()
            accounting_token = _REQUEST_ACCOUNTING.set(accounting)
        response_status: int | None = None

        async def checked_send(message: dict[str, Any]) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                if accounting is not None:
                    with accounting.lock:
                        accounting.client_status = response_status
            if message["type"] == "http.response.body" and not message.get("more_body"):
                observe_proxy_error(gate, message.get("body", b""), response_status)
            await send(message)

        try:
            await inner(scope, receive, checked_send)
        finally:
            if accounting is not None:
                gate.finish_task_request(accounting)
                assert accounting_token is not None
                _REQUEST_ACCOUNTING.reset(accounting_token)

    return guarded


async def serve_relay(gate: QuotaGate, port: int, model: str = MODEL) -> None:
    import httpx
    import uvicorn
    from relay.transport.proxy import _UsageTap, create_app, engine_from_settings
    from relay.protocols import OpenAIResponses

    engine = engine_from_settings()
    gate.input_budget = engine.input_budget
    original_record = engine.record

    def record_usage(exchange: Any, tokens: int | None) -> None:
        original_record(exchange, tokens)
        gate.observe_input_tokens(tokens)

    engine.record = record_usage
    inner = create_app(engine=engine)
    original_lifespan = inner.router.lifespan_context

    def upstream_event(kind: str, request: Any, response: Any = None) -> None:
        record = {"elapsed_seconds": time.monotonic() - gate.start,
                  "event": kind, "path": request.url.path}
        if response is not None:
            record.update(status=response.status_code,
                          content_type=response.headers.get("content-type"))
        with (gate.path.parent / "upstream-events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def check_upstream(request: Any, kind: str) -> None:
        try:
            check_upstream_request(request, gate, model)
        except Exception:
            if gate.reason in _ACCOUNT_REJECTION_REASONS:
                gate.mark_account_rejection()
            else:
                gate.mark_other_guard_rejection()
            raise
        gate.upstream_request_started(kind)
        upstream_event("request", request)

    def response_seen(response: Any, kind: str) -> None:
        upstream_event("response_headers", response.request, response)
        gate.upstream_response_seen(kind, response.status_code)

    class SampledStream(httpx.SyncByteStream):
        """Keep a bounded body sample when a strategy response never finishes."""

        def __init__(self, stream: Any, status: int) -> None:
            self.stream = stream
            self.status = status
            self.path = gate.path.parent / f"upstream-body-{time.time_ns()}.sample"

        def __iter__(self) -> Any:
            saved = 0
            error_body = bytearray()
            tap = _UsageTap(OpenAIResponses())
            try:
                with self.path.open("wb") as sample:
                    for chunk in self.stream:
                        tap.feed(chunk)
                        if self.status >= 400 and len(error_body) < 65536:
                            error_body.extend(chunk[:65536 - len(error_body)])
                        if saved < 65536:
                            part = chunk[:65536 - saved]
                            sample.write(part)
                            sample.flush()
                            saved += len(part)
                        yield chunk
            finally:
                tap.feed(b"\n")
                gate.observe_auxiliary_tokens(tap.tokens)
                if self.status >= 400:
                    observe_proxy_error(gate, bytes(error_body), self.status)

        def close(self) -> None:
            self.stream.close()

    def response_seen_sync(response: Any) -> None:
        response_seen(response, "auxiliary")
        response.stream = SampledStream(response.stream, response.status_code)

    async def check_upstream_async(request: Any) -> None:
        await asyncio.to_thread(check_upstream, request, "main")

    async def response_seen_async(response: Any) -> None:
        response_seen(response, "main")

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        async with original_lifespan(app):
            app.state.client.event_hooks["request"].append(check_upstream_async)
            app.state.summary_client.event_hooks["request"].append(
                lambda request: check_upstream(request, "auxiliary"))
            app.state.client.event_hooks["response"].append(response_seen_async)
            app.state.summary_client.event_hooks["response"].append(response_seen_sync)
            yield

    inner.router.lifespan_context = lifespan

    guarded = guarded_relay_app(gate, inner)
    await uvicorn.Server(uvicorn.Config(guarded, host="127.0.0.1", port=port, log_level="warning")).serve()


def run(args: argparse.Namespace) -> int:
    """Remove copied authentication even when setup fails before the run loop."""
    existed = args.out.exists()
    try:
        return _run(args)
    finally:
        if not existed:
            (args.out / "codex/auth.json").unlink(missing_ok=True)


def resolve_input_budget_counter(cli_counter: str | None, env_counter: str | None,
                                 all_call_budget: str | None) -> str | None:
    """Resolve the explicit counter opt-in while preserving the text-only default."""
    from relay.core.budget import (MEDIA_INPUT_BUDGET_COUNTER, InputBudget,
                                   PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)

    if env_counter is not None and env_counter != PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER:
        raise ValueError("RELAY_PILOT_INPUT_BUDGET_COUNTER accepts only the versioned PNG/JPEG policy")
    if env_counter is not None and cli_counter not in (None, env_counter):
        raise ValueError("The CLI and environment input-budget counters conflict")
    counter = cli_counter or env_counter
    if counter is not None and counter not in {
            InputBudget.name, MEDIA_INPUT_BUDGET_COUNTER, PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER}:
        raise ValueError(f"Unsupported input budget counter: {counter}")
    if counter is not None and all_call_budget != "1":
        raise ValueError("Explicit input-budget counters require RELAY_PILOT_ALL_CALL_BUDGET=1")
    return counter


def _run(args: argparse.Namespace) -> int:
    import httpx
    import socket
    from relay.core.budget import InputBudget

    input_budget_counter = resolve_input_budget_counter(
        getattr(args, "input_budget_counter", None),
        os.environ.get("RELAY_PILOT_INPUT_BUDGET_COUNTER"),
        os.environ.get("RELAY_PILOT_ALL_CALL_BUDGET"),
    )

    out, root = args.out.resolve(), args.root.resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory; never overwrite a trajectory")
    out.mkdir(parents=True, mode=0o700)
    if marker := closed_admission(out, root):
        write_json(out / "run.json", {"model": args.model, "strategy": args.strategy,
                   "window": args.window, "max_requests": args.max_requests,
                   "seconds_limit": args.seconds, "harness": "codex", "reasoning_effort": "medium",
                   "status": "preflight_failed", "stop_reason": "experiment_admission_closed",
                   "inference_started": False, "harness_requests": 0, "elapsed_seconds": 0,
                   "admission_marker": str(marker)})
        return 2
    home, work, relay = out / "codex", out / "work", out / "relay"
    for directory in (home, work, relay):
        directory.mkdir(mode=0o700)
    (home / "tmp").mkdir()
    handler = logging.FileHandler(relay / "runtime.log")
    logging.getLogger("relay").addHandler(handler)
    original = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    auth = json.loads((original / "auth.json").read_text())
    if auth.get("auth_mode") != "chatgpt" or not auth.get("tokens"):
        raise RuntimeError("Only an existing ChatGPT plan login is permitted")
    auth["tokens"]["refresh_token"] = "voided-by-relay-pilot"
    write_json(home / "auth.json", auth)
    (home / "auth.json").chmod(0o600)
    is_rlm = args.strategy in {"rlm", "rlm_persistent"}
    config(home, work, root, args.read_path, args.model, args.deny_path)
    if args.public:
        shutil.copytree(args.public, work, dirs_exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in
           {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}}
    # The parent app's helper lives under a denied per-host directory. Let the
    # isolated CLI create its own helper instead of inheriting that PATH entry.
    env["PATH"] = os.pathsep.join(part for part in env.get("PATH", "").split(os.pathsep)
                                  if "/tmp/arg0" not in part)
    helper_dir = work / ".bin"
    helper_dir.mkdir()
    codex_binary = Path(shutil.which("codex") or "codex").resolve(strict=True)
    (helper_dir / "codex-linux-sandbox").symlink_to(codex_binary)
    env["PATH"] = str(helper_dir) + os.pathsep + env["PATH"]
    env.update({"CODEX_HOME": str(home), "RELAY_HOME": str(relay), "TMPDIR": str(work / "tmp")})
    (work / "tmp").mkdir()
    prompt = args.prompt.read_text()
    (work / "task.txt").write_text(prompt)
    start = time.monotonic()
    rpc = CodexRPC(env)
    gate = QuotaGate(rpc, out / "quota.jsonl", args.max_requests, start, input_limit=args.window)
    process: subprocess.Popen[Any] | None = None
    cleanup_rlm = None
    result: dict[str, Any] = {"model": args.model, "strategy": args.strategy, "window": args.window,
                              "max_requests": args.max_requests, "seconds_limit": args.seconds,
                              "harness": "codex", "reasoning_effort": "medium",
                              "quota_mode": gate.quota_mode,
                              "account_no_overage_user_confirmed": gate.quota_mode == "account_enforced",
                              "source_fingerprint": source_fingerprint(),
                              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    try:
        if not gate.check(force=True, starting=True):
            raise RuntimeError(gate.reason)
        models = rpc.call("model/list")
        if not any(m.get("model") == args.model for m in models.get("data", [])):
            raise RuntimeError("Requested model is unavailable; no fallback permitted")
        # Validate the actual kernel boundary before any model generation.
        secret = out / "isolation-secret.txt"
        secret.write_text("hidden")
        helper_probe = home / "tmp/helper-probe.txt"
        helper_probe.write_text("helper-visible")
        probe = subprocess.run(
            ["codex", "sandbox", "-P", "pilot", "-C", str(work), "--", "python3", "-c",
             "from pathlib import Path; Path('probe.txt').write_text('ok'); "
             f"assert Path({str(helper_probe)!r}).read_text() == 'helper-visible'; "
             f"blocked = {[str(secret), str(home / 'auth.json'), str(home / 'config.toml')]!r}; "
             "\nfor p in blocked:\n try: Path(p).read_text()\n except (PermissionError,FileNotFoundError): pass\n else: raise RuntimeError('private file readable')"
             "\nprint('ISOLATION_OK')"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        (out / "isolation.log").write_text(probe.stdout + probe.stderr)
        if probe.returncode or "ISOLATION_OK" not in probe.stdout:
            raise RuntimeError("worker_isolation_preflight_failed")
        if is_rlm:
            from evals.pilot.rlm_sandbox import install_rlm_sandbox

            if args.rlm_python is None:
                raise RuntimeError("RLM needs --rlm-python or RELAY_PILOT_RLM_PYTHON outside the hidden scratch root")
            cleanup_rlm = install_rlm_sandbox(python=args.rlm_python, work=work,
                                              env=env, timeout=args.seconds)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        settings = {"strategy": args.strategy, "context_window": args.window,
                    "compact_threshold": int(args.window * 0.75),
                    "clm_budget": args.window, "clm_dir": str(work / "live_ctx"),
                    "prolong_log": str(work / ".prolong/log.jsonl"),
                    "acm_dir": str(relay / "acm"),
                    "ace_playbook": str(relay / "ace_playbook.md"),
                    "event_log": str(relay / "events.jsonl"),
                    "trace": str(relay / "trace.jsonl"), "cache_path": str(relay / "store.json")}
        if gate.all_call_budget:
            settings.update(input_budget_counter=input_budget_counter or InputBudget.name,
                            input_budget_limit=args.window)
            settings["budget_dispatch_slack"] = int(os.environ.get("RELAY_PILOT_INPUT_SLACK", "0"))
            settings["overflow_fallback"] = os.environ.get("RELAY_PILOT_OVERFLOW_FALLBACK", "error")
        if getattr(args, "harness_profile", None):
            settings["harness_profile"] = args.harness_profile
        write_json(relay / "settings.json", settings)
        # Relay reads its configuration from this process's environment.
        os.environ["RELAY_HOME"] = str(relay)
        thread = threading.Thread(target=lambda: asyncio.run(serve_relay(gate, port, args.model)), daemon=True)
        thread.start()
        for _ in range(100):
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/relay/v1/status", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Relay failed to start")
        command = ["codex", "exec", "--skip-git-repo-check", "--json", "-m", args.model,
                   "-c", f'openai_base_url="http://127.0.0.1:{port}/backend-api/codex"',
                   "-C", str(work), "-o", str(out / "answer.txt"), "-"]
        write_json(out / "command.json", command)
        write_json(out / "run.json", result | {"status": "running"})
        with (out / "codex.jsonl").open("w") as stdout, (out / "codex.stderr").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=stdout,
                                       stderr=stderr, start_new_session=True, text=True)
            assert process.stdin is not None
            process.stdin.write(prompt)
            process.stdin.close()
            while process.poll() is None:
                if time.monotonic() - start > args.seconds and gate.reason is None:
                    gate.reason = "wall_time_limit"
                gate.check()
                if gate.reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1)
        result.update(status="completed" if process.returncode == 0 and not gate.reason else "stopped",
                      exit_code=process.returncode, stop_reason=gate.reason,
                      harness_requests=gate.requests, quota_used_percent=gate.used)
        if not gate.check(force=True):
            result.update(status="stopped", stop_reason=gate.reason)
        dispatch_limit = gate.dispatch_input_limit or args.window
        result.update(metrics(out, home, dispatch_limit))
        result.update(nominal_input_budget=args.window, dispatch_input_limit=dispatch_limit,
                      main_requests_above_nominal_budget=sum(
                          n > args.window for n in result["provider_input_tokens_per_request"]),
                      auxiliary_requests_above_nominal_budget=sum(n > args.window for n in gate.auxiliary_inputs))
        result["main_context_valid"] = result["protocol_valid"]
        result.update(
            task_request_attempts=gate.raw_request_attempts,
            task_request_charged=gate.requests,
            provider_main_429_responses=gate.provider_main_429_responses,
            provider_auxiliary_429_responses=gate.provider_auxiliary_429_responses,
            request_count_refunds=gate.request_count_refunds,
            context_budget_scope="all_model_requests" if gate.all_call_budget else "main_agent_requests",
            auxiliary_input_tokens_per_request=gate.auxiliary_inputs,
            quota_mode=gate.quota_mode,
            account_no_overage_user_confirmed=gate.quota_mode == "account_enforced",
            quota_start_percent=gate.start_percent if gate.quota_mode == "poll" else None,
            quota_stop_percent=gate.stop_percent if gate.quota_mode == "poll" else None,
            quota_weekly_reset=gate.weekly_reset if gate.quota_mode == "poll" else None,
            input_budget_policy=gate.input_budget.fingerprint() if gate.input_budget else None,
            harness_profile=getattr(args, "harness_profile", None),
            overflow_fallback=gate.input_budget.overflow_fallback if gate.input_budget else "error",
        )
        if gate.input_budget:
            result["usage_note"] = (
                "Provider usage is measured separately from local budget estimates. All model calls share "
                "the declared dispatch limit. Nominal-budget exceedances within tolerance are recorded "
                "and remain score-eligible. Codex totals exclude auxiliary usage; Relay traces preserve it.")
            budget_trace = relay / "trace.jsonl"
            budget_rows = [json.loads(line) for line in budget_trace.read_text().splitlines()] if budget_trace.exists() else []
            result.update(
                budget_input_tokens_per_request=[row["budget_tokens"] for row in budget_rows
                    if "body" in row and "budget_tokens" in row],
                auxiliary_budget_tokens_per_request=[row["budget_tokens"] for row in budget_rows
                    if "summary_request" in row and "budget_tokens" in row],
                budget_rejections=[row["budget_rejection"] for row in budget_rows if "budget_rejection" in row],
                overflow_fallback_events=[row["overflow_fallback"] for row in budget_rows if "overflow_fallback" in row],
            )
            if gate.reason in {"context_budget_exhausted", "context_management_failed",
                               "upstream_request_timeout", "input_accounting_failed"}:
                result.update(protocol_valid=False, failure_kind=gate.reason)
        if gate.all_call_budget and any(n > dispatch_limit for n in gate.auxiliary_inputs):
            result.update(protocol_valid=False, failure_kind="auxiliary_context_budget_exceeded")
        if gate.reason == "provider_input_budget_mismatch":
            result.update(protocol_valid=False, failure_kind=gate.reason)
        if any(term in str(gate.reason or "") for term in ("quota", "credit", "ordinary_plan", "account_limit")):
            result.update(protocol_valid=False, failure_kind="quota_stopped")
        if result.get("sandbox_execution_errors"):
            result.update(status="infrastructure_failed", error="worker_sandbox_execution_failed")
        elif process.returncode != 0 and gate.reason is None:
            result.update(status="infrastructure_failed", protocol_valid=False,
                          error=harness_failure(out) or f"harness_exit_{process.returncode}")
        elif (gate.reason == "context_budget_exceeded" and result.get("requests_over_context_budget", 0)
              and not result.get("strategy_errors") and not result.get("sandbox_execution_errors")):
            result["failure_kind"] = "context_budget_exceeded"
    except Exception as exc:
        result.update(status="preflight_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        if cleanup_rlm is not None:
            cleanup_rlm()
        logging.getLogger("relay").removeHandler(handler)
        handler.close()
        rpc.close()
        result["elapsed_seconds"] = time.monotonic() - start
        write_json(out / "run.json", result)
        # Never retain a copied login in experimental artifacts after a run.
        (home / "auth.json").unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


def main() -> int:
    from relay.strategies import STRATEGIES
    from relay.core.budget import (MEDIA_INPUT_BUDGET_COUNTER, InputBudget,
                                   PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="External scratch root hidden from worker")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--strategy", choices=tuple(STRATEGIES), required=True)
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--rlm-python", type=Path,
                        default=os.environ.get("RELAY_PILOT_RLM_PYTHON"),
                        help="Public RLM-only runtime outside hidden benchmark data")
    parser.add_argument("--window", type=int, default=32_000)
    parser.add_argument("--harness-profile", choices=("browse_minimal_v1",),
                        help="Versioned evaluation-only reduction of known Codex boilerplate")
    parser.add_argument("--input-budget-counter", choices=(InputBudget.name, MEDIA_INPUT_BUDGET_COUNTER,
                                                            PNG_JPEG_MEDIA_INPUT_BUDGET_COUNTER),
                        help="Explicit input accounting policy; requires all-call budget mode")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--public", type=Path, help="Only task-public files, copied into worker cwd")
    parser.add_argument("--read-path", type=Path, action="append", default=[],
                        help="Explicit non-secret runtime dependencies to reopen in sandbox")
    parser.add_argument("--deny-path", type=Path, action="append", default=[],
                        help="Additional private data roots hidden from the worker")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
