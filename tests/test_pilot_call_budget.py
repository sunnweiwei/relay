"""The fresh batch must enforce its quota and settings on auxiliary calls too."""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from evals.pilot.run import (QuotaGate, RequestAccounting, _REQUEST_ACCOUNTING,
                             check_upstream_request, guarded_relay_app, observe_proxy_error)
from relay.core.budget import BudgetExceeded, InputBudget
from relay.protocols import OpenAIResponses
from relay.transport.proxy import _UsageTap
from tests.test_pilot import Account


def test_auxiliary_overflow_stops_next_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_PILOT_ALL_CALL_BUDGET", "1")
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, time.monotonic(), input_limit=23560)
    tap = _UsageTap(OpenAIResponses())
    data = b'data: ' + json.dumps({"type": "response.completed", "response": {
        "usage": {"input_tokens": 23561}}}).encode() + b'\r\n\r\n'
    for offset in range(0, len(data), 7):
        tap.feed(data[offset:offset + 7])
    gate.observe_auxiliary_tokens(tap.tokens)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
                            json={"model": "gpt-6-luna"})
    with pytest.raises(RuntimeError, match="auxiliary_context_budget_exceeded"):
        check_upstream_request(request, gate, "gpt-6-luna")
    assert gate.auxiliary_inputs == [23561]


def test_legacy_budget_scope_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELAY_PILOT_ALL_CALL_BUDGET", raising=False)
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, 0, input_limit=23560)
    gate.observe_auxiliary_tokens(25000)
    assert gate.reason is None


def test_worker_can_finish_task_after_soft_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_PILOT_QUOTA_START_PERCENT", "88")
    monkeypatch.setenv("RELAY_PILOT_QUOTA_STOP_PERCENT", "88")
    account = Account(used=85)
    gate = QuotaGate(account, tmp_path / "quota.jsonl", 100, 0)
    assert gate.check(force=True, starting=True)
    account.used = 88
    assert not gate.check(force=True)


def test_weekly_reset_mismatch_cannot_restart_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_PILOT_QUOTA_WEEKLY_RESET", "1800000000")
    gate = QuotaGate(Account(used=1), tmp_path / "quota.jsonl", 100, 0)
    assert not gate.check(force=True)
    assert gate.reason == "quota_window_changed"


@pytest.mark.parametrize("offset,allowed", [(0, True), (1, True), (-1, True), (5, True),
                                           (-5, True), (6, False), (-6, False), (604800, False)])
def test_weekly_reset_tolerance_is_anchored_without_resetting_budget(tmp_path, monkeypatch, offset, allowed):
    from evals.pilot.run import same_weekly_reset
    anchor = 1800000000
    monkeypatch.setenv("RELAY_PILOT_QUOTA_WEEKLY_RESET", str(anchor))
    monkeypatch.setenv("RELAY_PILOT_QUOTA_START_PERCENT", "88")
    monkeypatch.setenv("RELAY_PILOT_QUOTA_STOP_PERCENT", "88")

    class WeeklyAccount:
        used = 77
        reset = anchor + offset
        def call(self, method):
            return {"ordinaryUsageAllowed": True, "rateLimits": {
                "primary": {"usedPercent": self.used, "windowDurationMins": 10080, "resetsAt": self.reset},
                "credits": {"balance": "10"}}}

    account = WeeklyAccount()
    gate = QuotaGate(account, tmp_path / "quota.jsonl", 100, 0)
    assert gate.check(force=True, starting=True) is allowed
    assert gate.weekly_reset == str(anchor)
    assert gate.start_percent == gate.stop_percent == 88
    if allowed:
        # The fixed anchor cannot drift one accepted five-second step at a time.
        assert not same_weekly_reset(anchor + 6, gate.weekly_reset)
        account.used = 88
        assert not gate.check(force=True)
        assert gate.reason == "plan_quota_reserve_reached"
    else:
        assert gate.reason == "quota_window_changed"


def test_auxiliary_reasoning_must_match_main_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_PILOT_EXPECTED_EFFORT", "medium")
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, 0)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
                            json={"model": "gpt-6-luna", "reasoning": {"effort": "low"}})
    with pytest.raises(RuntimeError, match="model_reasoning_effort_mismatch"):
        check_upstream_request(request, gate, "gpt-6-luna")


def test_local_budget_rejection_keeps_recovery_available(tmp_path: Path) -> None:
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, time.monotonic(), input_limit=100)
    gate.input_budget = InputBudget(100)
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-6-luna", "input": [{"role": "user", "content": "large " * 500}]})
    with pytest.raises(BudgetExceeded):
        check_upstream_request(request, gate, "gpt-6-luna")
    assert gate.reason is None
    observe_proxy_error(gate, json.dumps(BudgetExceeded(500, 100, "main").payload()).encode())
    assert gate.reason == "context_budget_exhausted"
    assert not gate.check()


def test_provider_count_disagreement_is_a_harness_failure(tmp_path: Path) -> None:
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, time.monotonic(), input_limit=100)
    gate.input_budget = InputBudget(100)
    gate.observe_input_tokens(101)
    assert gate.reason == "provider_input_budget_mismatch"


@pytest.mark.parametrize("auxiliary", [False, True])
def test_authorized_tolerance_applies_to_provider_usage_too(tmp_path, monkeypatch, auxiliary):
    monkeypatch.setenv("RELAY_PILOT_ALL_CALL_BUDGET", "1")
    gate = QuotaGate(Account(), tmp_path / "quota.jsonl", 100, 0, input_limit=23560)
    gate.input_budget = InputBudget(23560, dispatch_slack=2356)
    observe = gate.observe_auxiliary_tokens if auxiliary else gate.observe_input_tokens
    for tokens in (23561, 25000, 25916):
        observe(tokens)
        assert gate.reason is None
    observe(25917)
    assert gate.reason == "provider_input_budget_mismatch"


def _account_enforced_gate(tmp_path, monkeypatch, max_requests=10):
    monkeypatch.setenv("RELAY_PILOT_QUOTA_MODE", "account_enforced")
    monkeypatch.setenv("RELAY_PILOT_ACCOUNT_NO_OVERAGE_CONFIRMED", "1")
    return QuotaGate(Account(), tmp_path / "quota.jsonl", max_requests, 0)


def test_main_provider_429_releases_slot_and_records_raw_attempt(tmp_path, monkeypatch):
    gate = _account_enforced_gate(tmp_path, monkeypatch)
    gate.record_raw_request_attempt()
    assert gate.admit_task_request()
    accounting = RequestAccounting()
    token = _REQUEST_ACCOUNTING.set(accounting)
    try:
        gate.upstream_request_started("main")
        gate.upstream_response_seen("main", 429)
        accounting.client_status = 429
        observe_proxy_error(gate, b'{"error":{"code":"rate_limit_exceeded"}}', 429)
        gate.finish_task_request(accounting)
    finally:
        _REQUEST_ACCOUNTING.reset(token)
    assert gate.raw_request_attempts == 1
    assert gate.requests == 0
    assert gate.request_count_refunds == 1
    assert gate.provider_main_429_responses == 1


def test_terminal_auxiliary_only_429_releases_slot(tmp_path, monkeypatch):
    gate = _account_enforced_gate(tmp_path, monkeypatch)
    gate.record_raw_request_attempt()
    assert gate.admit_task_request()
    accounting = RequestAccounting(client_status=500)
    token = _REQUEST_ACCOUNTING.set(accounting)
    try:
        gate.upstream_request_started("auxiliary")
        gate.upstream_response_seen("auxiliary", 429)
        observe_proxy_error(gate, b'{"error":{"code":"context_management_failed"}}', 500)
        gate.finish_task_request(accounting)
    finally:
        _REQUEST_ACCOUNTING.reset(token)
    assert gate.requests == 0
    assert gate.request_count_refunds == 1
    assert gate.provider_auxiliary_429_responses == 1


def test_successful_auxiliary_retry_then_main_generation_is_charged(tmp_path, monkeypatch):
    gate = _account_enforced_gate(tmp_path, monkeypatch)
    gate.record_raw_request_attempt()
    assert gate.admit_task_request()
    accounting = RequestAccounting(client_status=200)
    token = _REQUEST_ACCOUNTING.set(accounting)
    try:
        gate.upstream_request_started("auxiliary")
        gate.upstream_response_seen("auxiliary", 429)
        gate.upstream_request_started("auxiliary")
        gate.upstream_response_seen("auxiliary", 200)
        gate.upstream_request_started("main")
        gate.upstream_response_seen("main", 200)
        gate.finish_task_request(accounting)
    finally:
        _REQUEST_ACCOUNTING.reset(token)
    assert gate.requests == 1
    assert gate.request_count_refunds == 0
    assert gate.provider_auxiliary_429_responses == 1


def test_quota_429_is_not_refunded(tmp_path, monkeypatch):
    gate = _account_enforced_gate(tmp_path, monkeypatch)
    gate.record_raw_request_attempt()
    assert gate.admit_task_request()
    accounting = RequestAccounting(client_status=429)
    token = _REQUEST_ACCOUNTING.set(accounting)
    try:
        gate.upstream_request_started("main")
        gate.upstream_response_seen("main", 429)
        observe_proxy_error(gate, b'{"error":{"code":"insufficient_quota"}}', 429)
        gate.finish_task_request(accounting)
    finally:
        _REQUEST_ACCOUNTING.reset(token)
    assert gate.requests == 1
    assert gate.request_count_refunds == 0
    assert gate.reason == "account_limit_reached"


def test_request_limit_rejection_is_synthetic_and_not_refunded(tmp_path, monkeypatch):
    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)
    gate.record_raw_request_attempt()
    assert gate.admit_task_request()
    gate.finish_task_request(RequestAccounting(main_requests=1, main_statuses=[200]))
    gate.record_raw_request_attempt()
    assert not gate.admit_task_request()
    assert gate.reason == "model_request_limit"
    assert gate.raw_request_attempts == 2
    assert gate.requests == 1
    assert gate.request_count_refunds == 0


def test_request_reservations_are_atomic_under_concurrency(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)
    for _ in range(8):
        gate.record_raw_request_attempt()
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(lambda _: gate.admit_task_request(), range(8)))
    assert sum(admitted) == 1
    assert gate.requests == 1
    assert gate.raw_request_attempts == 8


def test_asgi_provider_429_refunds_then_successful_retry_is_admitted(tmp_path, monkeypatch):
    import httpx
    from starlette.requests import Request

    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)
    calls = 0

    async def relay(scope, receive, send):
        nonlocal calls
        await Request(scope, receive).body()
        calls += 1
        status = 429 if calls == 1 else 200
        await asyncio.to_thread(gate.upstream_request_started, "main")
        await asyncio.to_thread(gate.upstream_response_seen, "main", status)
        body = (b'{"error":{"code":"rate_limit_exceeded"}}' if status == 429 else b'{"ok":true}')
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded_relay_app(gate, relay)),
                                     base_url="http://relay") as client:
            first = await client.post("/responses", json={"try": 1})
            assert first.status_code == 429
            assert gate.requests == 0
            second = await client.post("/responses", json={"try": 2})
            assert second.status_code == 200

    asyncio.run(exercise())
    assert gate.raw_request_attempts == 2
    assert gate.requests == 1
    assert gate.provider_main_429_responses == 1
    assert gate.request_count_refunds == 1


def test_asgi_auxiliary_only_429_refunds_without_main_generation(tmp_path, monkeypatch):
    import httpx
    from starlette.requests import Request

    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)

    async def relay(scope, receive, send):
        await Request(scope, receive).body()
        await asyncio.to_thread(gate.upstream_request_started, "auxiliary")
        await asyncio.to_thread(gate.upstream_response_seen, "auxiliary", 429)
        observe_proxy_error(gate, b'{"error":{"code":"rate_limit_exceeded"}}', 429)
        await send({"type": "http.response.start", "status": 500,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"error":{"code":"context_management_failed"}}'})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded_relay_app(gate, relay)),
                                     base_url="http://relay") as client:
            response = await client.post("/responses", json={"try": 1})
            assert response.status_code == 500

    asyncio.run(exercise())
    assert gate.requests == 0
    assert gate.provider_auxiliary_429_responses == 1
    assert gate.request_count_refunds == 1


def test_asgi_account_quota_429_keeps_charge_and_blocks_next_request(tmp_path, monkeypatch):
    import httpx
    from starlette.requests import Request

    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)

    async def relay(scope, receive, send):
        await Request(scope, receive).body()
        await asyncio.to_thread(gate.upstream_request_started, "main")
        await asyncio.to_thread(gate.upstream_response_seen, "main", 429)
        await send({"type": "http.response.start", "status": 429,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"error":{"code":"insufficient_quota"}}'})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded_relay_app(gate, relay)),
                                     base_url="http://relay") as client:
            first = await client.post("/responses", json={"try": 1})
            assert first.status_code == 429
            assert gate.requests == 1
            assert gate.reason == "account_limit_reached"
            second = await client.post("/responses", json={"try": 2})
            assert second.status_code == 429
            assert b"account_limit_reached" in second.content

    asyncio.run(exercise())
    assert gate.raw_request_attempts == 2
    assert gate.requests == 1
    assert gate.request_count_refunds == 0


def test_asgi_inflight_cap_rejection_does_not_latch_before_429_refund(tmp_path, monkeypatch):
    import httpx
    from starlette.requests import Request

    gate = _account_enforced_gate(tmp_path, monkeypatch, max_requests=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def relay(scope, receive, send):
        nonlocal calls
        payload = json.loads(await Request(scope, receive).body())
        calls += 1
        if payload["try"] == "first":
            await asyncio.to_thread(gate.upstream_request_started, "main")
            first_started.set()
            await release_first.wait()
            status, body = 429, b'{"error":{"code":"rate_limit_exceeded"}}'
            await asyncio.to_thread(gate.upstream_response_seen, "main", status)
        else:
            status, body = 200, b'{"ok":true}'
            await asyncio.to_thread(gate.upstream_request_started, "main")
            await asyncio.to_thread(gate.upstream_response_seen, "main", status)
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=guarded_relay_app(gate, relay)),
                                     base_url="http://relay") as client:
            pending = asyncio.create_task(client.post("/responses", json={"try": "first"}))
            await first_started.wait()
            capped = await client.post("/responses", json={"try": "concurrent"})
            assert capped.status_code == 429
            assert b"request_slot_temporarily_reserved" in capped.content
            assert gate.reason is None
            release_first.set()
            assert (await pending).status_code == 429
            retry = await client.post("/responses", json={"try": "next"})
            assert retry.status_code == 200

    asyncio.run(exercise())
    assert calls == 2
    assert gate.raw_request_attempts == 3
    assert gate.requests == 1
    assert gate.request_count_refunds == 1
    assert gate.reason is None
