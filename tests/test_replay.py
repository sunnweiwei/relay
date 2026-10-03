"""Real harness traffic, replayed: every request of every recorded session through the engine.

The fixtures in tests/replay are Docker check runs (tests/docker/check.py) of each harness:
two turns of one session through its own resume, instruction files edited between them, a
sub-agent asked for; plus Claude Code compacting itself and Codex spawning sub-agents.
Compacting every 3k new tokens, every request must find its conversation's stored compaction,
carry the harness's latest instructions, and stay well-formed for its protocol.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Kind
from relay.core.tokens import approx_tokens
from relay.harnesses import detect
from relay.protocols import codec_for
from relay.strategies import Compaction
from tests.replay.build import load

FIXTURES = sorted((Path(__file__).parent / "replay").glob("*.json.gz"))
SUMMARIES = {  # a completed summary response in each protocol
    "openai_responses": {"status": "completed", "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "SUMMARY"}]}]},
    "openai_chat": {"choices": [{"message": {"role": "assistant", "content": "SUMMARY"}, "finish_reason": "stop"}]},
    "anthropic_messages": {"content": [{"type": "text", "text": "SUMMARY"}], "stop_reason": "end_turn"},
    "gemini": {"candidates": [{"content": {"role": "model", "parts": [{"text": "SUMMARY"}]}, "finishReason": "STOP"}]},
}
LATEST = "codename is BETA"  # the instruction files' second version


class ReplayTests(unittest.TestCase):
    def test_recorded_sessions(self) -> None:
        self.assertGreaterEqual(len(FIXTURES), 15)
        for path in FIXTURES:
            name, requests = load(path)
            with self.subTest(name):
                compactions, hits = self.replay(requests)
                self.assertGreater(compactions, 0, "the session never compacted")
                self.assertGreater(hits, 0, "no request found a stored compaction")

    def replay(self, requests: list[tuple[str, dict[str, Any]]]) -> tuple[int, int]:
        engine, compacted, compactions, hits = Engine(Compaction(growth=3_000, min_gain=0)), set(), 0, 0
        for n, (path, body) in enumerate(requests):
            codec, harness = codec_for(path), detect({}, path=path)
            post = lambda request: (200, SUMMARIES[codec.name])  # noqa: E731
            exchange = engine.prepare(codec, harness, body, tenant="t", post=post)
            engine.record(exchange, approx_tokens(json.dumps(exchange.body)))
            items = codec.items(body)
            if harness.compacting(codec, items):  # the harness summarizing itself: no new compaction,
                self.assertFalse(exchange.compacted)  # and its request (prompt or trigger last) stays intact
                self.assertTrue(harness.compacting(codec, codec.items(exchange.body)), f"request {n} lost its prompt")
                continue
            thread = tuple(k for k in harness.identity(codec, items) if k not in (b"\0system", b"\0context"))[:3]
            # The prefix store: a conversation that compacted finds its compaction again.
            self.assertIsNone(exchange.diverged, f"request {n} left its stored compaction at item {exchange.diverged}")
            if thread in compacted and not exchange.compacted:
                self.assertGreater(exchange.state.get("covered", 0), 0, f"request {n} missed its compaction")
                hits += 1
            if exchange.compacted:
                compacted.add(thread)
                compactions += 1
            sent = codec.items(exchange.body)
            # The harness's latest instructions reach the model.
            told = [harness.refine(codec.classify(item)) for item in items]
            if any(LATEST in i.text for i in told if i.kind in {Kind.SYSTEM, Kind.CONTEXT, Kind.USER}):
                self.assertIn(LATEST, json.dumps(exchange.body), f"request {n} dropped the latest instructions")
            # Well-formed for the protocol: no tool result without its call, legal system messages.
            self.assertEqual(orphans(codec.name, sent), [], f"request {n} has tool results without their calls")
            if codec.name == "anthropic_messages":
                self.assertTrue(legal_system_messages(sent), f"request {n} places a system message illegally")
        return compactions, hits


def orphans(protocol: str, items: list[dict[str, Any]]) -> list[str]:
    """Tool results whose call is not in the request before them."""

    calls: set[str] = set()
    missing = []
    for item in items:
        if protocol == "openai_responses":
            kind, call = item.get("type", "message"), item.get("call_id")
            if call and kind.endswith("_call"):
                calls.add(call)
            elif call and kind.endswith("_output") and call not in calls:
                missing.append(call)
        elif protocol == "openai_chat":
            calls |= {c.get("id") for c in item.get("tool_calls") or []}
            if item.get("role") == "tool" and item.get("tool_call_id") not in calls:
                missing.append(item.get("tool_call_id"))
        elif protocol == "anthropic_messages" and isinstance(item.get("content"), list):
            for block in item["content"]:
                if block.get("type") == "tool_use":
                    calls.add(block.get("id"))
                elif block.get("type") == "tool_result" and block.get("tool_use_id") not in calls:
                    missing.append(block.get("tool_use_id"))
    return missing


def legal_system_messages(messages: list[dict[str, Any]]) -> bool:
    """Anthropic: a system message never comes first, and precedes the model's turn or ends."""

    return all(
        n > 0 and (n + 1 == len(messages) or messages[n + 1].get("role") in {"assistant", "system"})
        for n, message in enumerate(messages) if message.get("role") == "system"
    )


if __name__ == "__main__":
    unittest.main()
