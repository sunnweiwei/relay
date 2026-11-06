"""Offline integration checks for opt-in recovery followed by shared truncation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from evals.pilot.run import QuotaGate, observe_proxy_error
from relay.core.budget import BudgetExceeded, InputAccountingError, InputBudget
from relay.core.engine import Engine, _Summarizer
from relay.core.ir import Context, Item, Kind
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from tests.test_engine import body, msg, step, summary
from tests.test_input_budget import ForcedManager, proxy_request

CODEC = OpenAIResponses()


def test_main_truncates_only_after_method_recovery_and_persists_history() -> None:
    policy = InputBudget(160, overflow_fallback="truncate")
    manager = ForcedManager("noop", [])
    engine = Engine(manager, input_budget=policy)
    request = body(msg("developer", "preserved rules"), msg("user", "preserved task"),
                   *step(1, 5000), *step(2, 600))
    reports = []
    with pytest.raises(BudgetExceeded):
        engine.prepare(CODEC, Harness(), request, tenant="t", post=lambda _: summary("unused"))
    recovered = engine.prepare(CODEC, Harness(), request, tenant="t", post=lambda _: summary("unused"),
                               force=True, on_fallback=reports.append)
    assert manager.forces == [False, True]
    assert recovered.body["input"][:2] == request["input"][:2]
    assert not CODEC.orphans(recovered.body["input"])
    policy.check(CODEC, recovered.body, "test")
    assert reports[0]["dropped_groups"] >= 1
    engine.record(recovered, policy.count(CODEC, recovered.body))
    again = engine.prepare(CODEC, Harness(), request, tenant="t", post=lambda _: summary("unused"))
    assert again.body == recovered.body
    assert not any(value.get("call_id") == "c1" for value in again.body["input"])


def test_proxy_noop_recovery_reaches_fake_transport_after_truncation() -> None:
    manager = ForcedManager("noop", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(120, overflow_fallback="truncate"),
        body(msg("user", "task"), *step(1, 5000))))
    assert response.status_code == 200 and len(sent) == 1
    assert manager.forces == [False, True]
    assert sent[0]["input"][0] == msg("user", "task")


def test_generated_summary_and_instructions_survive_persisted_fallback() -> None:
    class LargeSummary(ForcedManager):
        def plan(self, request, lm):
            if request.force:
                return Context((Item(Kind.SYSTEM, "method guidance"), Item(Kind.USER, "task"),
                                Item(Kind.SUMMARY, "preserved summary evidence " * 1000)))
            return None

    policy = InputBudget(200, overflow_fallback="truncate")
    engine = Engine(LargeSummary("large", []), input_budget=policy)
    request = body(msg("developer", "fixed rules"), msg("user", "task"), *step(1, 5000))
    first = engine.prepare(CODEC, Harness(), request, tenant="t", post=lambda _: summary("unused"), force=True)
    engine.record(first, policy.count(CODEC, first.body))
    second = engine.prepare(CODEC, Harness(), request, tenant="t", post=lambda _: summary("unused"))
    assert first.body == second.body
    text = json.dumps(second.body)
    assert "harness truncated" in text and "summary evidence" in text
    assert text.count("method guidance") == 1


def test_provider_rejection_gets_one_smaller_request_even_when_method_is_noop() -> None:
    manager = ForcedManager("noop", [])
    policy = InputBudget(400, overflow_fallback="truncate")
    response, sent = asyncio.run(proxy_request(manager, policy,
        body(msg("user", "task"), *step(1, 700)), [400, 200]))
    assert response.status_code == 200 and len(sent) == 2
    assert policy.count(CODEC, sent[1]) < policy.count(CODEC, sent[0])
    assert manager.forces == [False, True]


def test_completion_trims_iteration_history_and_preserves_task_and_rules() -> None:
    policy, sent, reports = InputBudget(130, overflow_fallback="truncate"), [], []
    lm = _Summarizer(CODEC, body(), lambda _: [], 0,
                    lambda b: sent.append(b) or summary("done"), 130, 4,
                    input_budget=policy, on_fallback=reports.append)
    result = lm.complete([Item(Kind.SYSTEM, "RLM rules"), Item(Kind.USER, "original task"),
                          Item(Kind.ASSISTANT, "old iteration " * 2000),
                          Item(Kind.USER, "recent output " * 1000)])
    assert result == "done" and len(sent) == 1
    assert sent[0]["instructions"] == "RLM rules"
    assert CODEC.classify(sent[0]["input"][0]).text == "original task"
    policy.check(CODEC, sent[0], "test")
    assert reports[0]["scope"] == "completion"


def test_summary_trims_unsplittable_output_but_preserves_task_prompt_and_pairs() -> None:
    policy, sent, reports = InputBudget(200, overflow_fallback="truncate"), [], []
    history = [msg("developer", "rules"), msg("user", "original task"), *step(1, 10000)]
    lm = _Summarizer(CODEC, body(*history), lambda n: history[:n], 1,
                    lambda b: sent.append(b) or summary("done"), 200, 4,
                    input_budget=policy, on_fallback=reports.append)
    assert lm.summarize(len(history), "summarize faithfully") == "done"
    assert len(sent) == 1 and not CODEC.orphans(sent[0]["input"])
    assert sent[0]["input"][:2] == history[:2]
    assert CODEC.classify(sent[0]["input"][-1]).text == "summarize faithfully"
    assert any(value.get("call_id") == "c1" for value in sent[0]["input"])
    assert reports[0]["truncated_text_items"] == 1
    policy.check(CODEC, sent[0], "test")


def test_auxiliary_failure_does_not_disable_main_truncation() -> None:
    manager = ForcedManager("auxiliary", [])
    response, sent = asyncio.run(proxy_request(manager, InputBudget(120, overflow_fallback="truncate"),
        body(msg("user", "task"), *step(1, 5000))))
    assert response.status_code == 200 and len(sent) == 1
    assert manager.forces == [False, True]


@pytest.mark.parametrize("exception, code", [
    (httpx.ReadTimeout("timed out"), "upstream_request_timeout"),
    (InputAccountingError("unknown reasoning usage"), "input_accounting_failed"),
])
def test_failures_keep_specific_codes_and_stop_outer_task_retries(tmp_path: Path, exception, code: str) -> None:
    class FailingManager(ForcedManager):
        def plan(self, request, lm):
            raise exception

    response, sent = asyncio.run(proxy_request(FailingManager("error", []),
        InputBudget(120, overflow_fallback="truncate"), body(msg("user", "task"))))
    assert response.json()["error"]["code"] == code and not sent
    gate = QuotaGate(None, tmp_path / "quota.jsonl", 100, 0)  # no metadata call needed
    observe_proxy_error(gate, json.dumps(response.json()).encode())
    assert gate.reason == code
