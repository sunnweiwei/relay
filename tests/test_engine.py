from __future__ import annotations

import json
import unittest
import unittest.mock
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Rewrite
from relay.harnesses import ClaudeCode, Codex, Harness
from relay.prompts import SUMMARY_PREFIX
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import Compaction

CODEC, HARNESS = OpenAIResponses(), Harness()
SUMMARY_OK = (200, {"status": "completed", "output": [{"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": "S"}]}]})


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def step(i: int, size: int = 800) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "x" * size}]


def body(*items: dict[str, Any]) -> dict[str, Any]:
    return {"model": "m", "input": list(items)}


class Upstream:
    """Answers summary requests from a queue (default: success) and records them."""

    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> tuple[int, Any]:
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else SUMMARY_OK


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine(Compaction(threshold=300, min_gain=0))

    def prepare(self, request: dict[str, Any], upstream: Upstream, **kwargs: Any):
        return self.engine.prepare(CODEC, HARNESS, request, tenant="t", post=upstream, **kwargs)

    def test_small_requests_are_forwarded_untouched(self) -> None:
        request = body(msg("developer", "rules"), msg("user", "task"))
        upstream = Upstream()
        self.assertIs(self.prepare(request, upstream).body, request)
        self.assertEqual(upstream.requests, [])

    def test_compaction_is_stored_and_reused_by_later_requests(self) -> None:
        upstream = Upstream()
        history = [msg("developer", "rules"), msg("user", "task"), *step(1), *step(2)]
        first = self.prepare(body(*history), upstream)
        self.assertTrue(first.compacted)
        self.assertEqual(len(upstream.requests), 1)
        self.assertEqual(upstream.requests[0]["input"][:-1], history)
        summary = msg("user", f"{SUMMARY_PREFIX}\nS")
        self.assertEqual(first.body["input"][:2], history[:2])
        self.assertEqual(first.body["input"][2]["content"][0]["text"], summary["content"])

        follow_up = [*history, *step(3, size=10)]
        second = self.prepare(body(*follow_up), upstream)
        self.assertFalse(second.compacted)
        self.assertEqual(len(upstream.requests), 1)  # no new summary
        self.assertEqual(second.body["input"][:3], first.body["input"])
        self.assertEqual(second.body["input"][3:], step(3, size=10))

    def test_reported_usage_anchors_the_next_estimate(self) -> None:
        upstream = Upstream()
        history = [msg("user", "task"), *step(1, size=10)]
        exchange = self.prepare(body(*history), upstream)
        self.assertFalse(exchange.compacted)
        self.engine.record(exchange, 299)  # the upstream counted far more than the text
        self.assertTrue(self.prepare(body(*history, *step(2, size=10)), upstream).compacted)

    def test_failures_forward_unchanged_and_back_off(self) -> None:
        upstream = Upstream((400, {"error": {"message": "invalid request"}}))
        request = body(msg("user", "task"), *step(1), *step(2))
        with self.assertLogs("relay", "WARNING"):
            self.assertIs(self.prepare(request, upstream).body, request)
        self.assertIs(self.prepare(request, upstream).body, request)
        self.assertEqual(len(upstream.requests), 1)  # backed off
        self.assertTrue(self.prepare(request, upstream, force=True).compacted)

    def test_summary_overflow_drops_the_oldest_history_but_keeps_the_initial_context(self) -> None:
        overflow = (400, {"error": {"code": "context_length_exceeded", "message": "too long"}})
        upstream = Upstream(overflow)
        history = [msg("developer", "rules"), msg("user", "task"), *step(1), *step(2), *step(3)]
        self.assertTrue(self.prepare(body(*history), upstream).compacted)
        retried = upstream.requests[1]["input"][:-1]
        self.assertEqual(retried[0], history[0])
        self.assertLess(len(retried), len(history))
        self.assertEqual(retried[-2:], step(3))

    def test_tenants_do_not_share_state(self) -> None:
        upstream = Upstream()
        request = body(msg("user", "task"), *step(1), *step(2))
        self.prepare(request, upstream)
        self.engine.prepare(CODEC, HARNESS, request, tenant="other", post=upstream)
        self.assertEqual(len(upstream.requests), 2)

    def test_invalid_rewrites_are_rejected(self) -> None:
        class Broken:
            name = "broken"

            def fingerprint(self) -> dict[str, Any]:
                return {}

            def plan(self, view, summarizer):  # noqa: ANN001
                return Rewrite(1, ())  # splits a call from its output

        engine = Engine(Broken())
        request = body(*step(1))
        with self.assertLogs("relay", "WARNING"):
            exchange = engine.prepare(CODEC, HARNESS, request, tenant="t", post=Upstream())
        self.assertIs(exchange.body, request)


if __name__ == "__main__":
    unittest.main()


class InitialContextTests(unittest.TestCase):
    def test_initial_context_survives_repeated_compaction_in_codex_layout(self) -> None:
        engine, upstream = Engine(Compaction(threshold=300, min_gain=0)), Upstream()
        context = msg("user", "<environment_context>cwd</environment_context>")
        history = [msg("developer", "rules"), context, msg("user", "task"), *step(1),
                   msg("assistant", "done"), msg("user", "more"), *step(2)]
        for n in range(3, 6):  # each request compacts again
            history += step(n)
            exchange = engine.prepare(CODEC, Codex(), body(*history), tenant="t", post=upstream)
            self.assertTrue(exchange.compacted)
            engine.record(exchange, 500)
            sent = exchange.body["input"]
            # Codex re-renders its developer context with the rest, just above the last user message.
            self.assertEqual(sent[:4], [msg("user", "task"), msg("developer", "rules"), context, msg("user", "more")])
            self.assertEqual(CODEC.classify(sent[4]).text.split("\n")[0], SUMMARY_PREFIX.split("\n")[0])
            self.assertEqual(len(sent), 5)


    def test_context_injected_after_the_first_prompt_is_initial_context(self) -> None:
        engine, upstream = Engine(Compaction(threshold=300, min_gain=0)), Upstream()
        reminder = msg("user", "<system-reminder>Auto permission mode is active.</system-reminder>")
        history = [msg("user", "task"), reminder, *step(1), *step(2)]
        for n in range(3, 5):
            history += step(n)
            exchange = engine.prepare(CODEC, HARNESS, body(*history), tenant="t", post=upstream)
            engine.record(exchange, 500)
            self.assertEqual(exchange.body["input"][:2], [reminder, msg("user", "task")])
            self.assertEqual(len(exchange.body["input"]), 3)  # ... and the summary


    def test_anthropic_system_context_is_placed_legally_and_comes_back(self) -> None:
        codec, engine, upstream = AnthropicMessages(), Engine(Compaction(threshold=300, min_gain=0)), Upstream()
        upstream.responses = [(200, {"content": [{"type": "text", "text": "S"}], "stop_reason": "end_turn"})] * 3

        def tool(i: int) -> list[dict[str, Any]]:
            return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "sh", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 800}]}]

        env = {"role": "system", "content": "# Environment"}
        history = [{"role": "user", "content": "task"}, env, *tool(1), *tool(2)]
        roles = []
        for extra in ([], [{"role": "assistant", "content": "done"}, {"role": "user", "content": "more"}], tool(3)):
            history += extra
            exchange = engine.prepare(codec, ClaudeCode(), {"model": "m", "messages": history}, tenant="t", post=upstream)
            self.assertTrue(exchange.compacted)
            engine.record(exchange, 500)
            roles.append([m["role"] for m in exchange.body["messages"]])
        self.assertEqual(roles[0], ["user", "user", "system"])  # mid-turn: after the summary, before the model
        self.assertEqual(roles[1], ["user", "user", "user"])  # turn start: no legal place, so left out
        self.assertEqual(roles[2], ["user", "user", "user", "system"])  # back at the next mid-turn compaction


class SummaryRetryTests(unittest.TestCase):
    def test_transient_errors_are_retried(self) -> None:
        engine = Engine(Compaction(threshold=300, min_gain=0))
        upstream = Upstream((429, {"error": {"message": "slow down"}}), (503, None))
        with unittest.mock.patch("relay.core.engine.time.sleep"):
            exchange = engine.prepare(CODEC, HARNESS, body(msg("user", "task"), *step(1), *step(2)),
                                      tenant="t", post=upstream)
        self.assertTrue(exchange.compacted)
        self.assertEqual(len(upstream.requests), 3)

    def test_incomplete_summaries_are_retried_and_never_used(self) -> None:
        cut_short = (200, {"output": [{"type": "message", "role": "assistant",
                                       "content": [{"type": "output_text", "text": "half a summ"}]}]})
        request = body(msg("user", "task"), *step(1), *step(2))
        engine = Engine(Compaction(threshold=300, min_gain=0))
        with unittest.mock.patch("relay.core.engine.time.sleep"):
            exchange = engine.prepare(CODEC, HARNESS, request, tenant="t", post=Upstream(cut_short, cut_short))
        self.assertTrue(exchange.compacted)
        self.assertNotIn("half a summ", json.dumps(exchange.body))

        engine = Engine(Compaction(threshold=300, min_gain=0))
        with unittest.mock.patch("relay.core.engine.time.sleep"), self.assertLogs("relay", "WARNING"):
            exchange = engine.prepare(CODEC, HARNESS, request, tenant="t", post=Upstream(*[cut_short] * 6))
        self.assertIs(exchange.body, request)  # gave up: forwarded unchanged


class UnreportedUsageTests(unittest.TestCase):
    def test_growth_falls_back_to_estimates_when_usage_is_never_reported(self) -> None:
        engine = Engine(Compaction(growth=150, min_gain=0))
        upstream = Upstream()
        history = [msg("user", "task"), *step(1, size=40)]
        engine.record(engine.prepare(CODEC, HARNESS, body(*history), tenant="t", post=upstream), None)
        exchange = engine.prepare(CODEC, HARNESS, body(*history, *step(2, size=800)), tenant="t", post=upstream)
        self.assertTrue(exchange.compacted)
