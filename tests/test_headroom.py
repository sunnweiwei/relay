"""Headroom as the `headroom` strategy: the conversation it is asked about, its compressed results
forwarded in place, and what was forwarded never compressed again."""

from __future__ import annotations

import json
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Request
from relay.harnesses import Harness
from relay.protocols import AnthropicMessages, OpenAIResponses
from relay.strategies import Headroom

CODEC, HARNESS = OpenAIResponses(), Harness()
TOOLS = [{"type": "function", "name": "sh", "parameters": {}}]


def msg(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def run(i: int, command: str, output: str) -> list[dict[str, Any]]:
    return [{"type": "function_call", "call_id": f"c{i}", "name": "sh", "arguments": json.dumps({"cmd": command})},
            {"type": "function_call_output", "call_id": f"c{i}", "output": output}]


class FakeHeadroom:
    """Headroom's /v1/compress, compressing each result past the frozen messages to its size."""

    def __init__(self, failures: int = 0) -> None:
        self.bodies, self.failures = [], failures

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if self.failures:
            self.failures -= 1
            raise ValueError("Headroom answered 503: compression_error")
        frozen = body["config"]["frozen_message_count"]
        return {"messages": [m if n < frozen or m["role"] != "tool" else {**m, "content": f"[{len(m['content'])} chars]"}
                             for n, m in enumerate(body["messages"])],
                "tokens_before": 300, "tokens_saved": 200, "transforms_applied": ["router:log:0.01"]}


class HeadroomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.headroom = FakeHeadroom()
        self.engine = Engine(Headroom(post=self.headroom))

    def send(self, items: list[dict[str, Any]], codec: Any = CODEC) -> list[dict[str, Any]]:
        field = "input" if codec is CODEC else "messages"
        body = {"model": "m", "tools": TOOLS, field: items}
        return self.engine.prepare(codec, HARNESS, body, tenant="t", post=lambda r: (500, {})).body[field]

    def test_a_result_goes_compressed_and_headroom_sees_the_conversation_as_chat_messages(self) -> None:
        sent = self.send([msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 500)])
        (body,) = self.headroom.bodies
        self.assertEqual(body["messages"], [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1_0", "type": "function", "function": {"name": "sh", "arguments": '{"cmd": "cat app.log"}'}}]},
            {"role": "tool", "tool_call_id": "call_1_0", "content": "x" * 500}])
        self.assertEqual((body["model"], body["config"]), ("claude-sonnet-4-5-20250929", {"frozen_message_count": 0}))
        self.assertEqual(sent[-1], {"type": "function_call_output", "call_id": "c1", "output": "[500 chars]"})

    def test_what_was_forwarded_stays_as_it_was_and_only_the_new_results_are_compressed(self) -> None:
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 500)]
        self.send(first)
        sent = self.send([*first, *run(2, "cat test.log", "y" * 800)])  # (the harness sends its own history)
        body = self.headroom.bodies[-1]
        self.assertEqual(body["config"]["frozen_message_count"], 3)
        self.assertEqual(body["messages"][2]["content"], "[500 chars]")  # what the provider has, not the original
        self.assertEqual([item.get("output") for item in sent[1:]], [None, "[500 chars]", None, "[800 chars]"])

    def test_a_request_with_no_new_result_asks_nothing_and_what_came_since_is_frozen_later(self) -> None:
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 500)]
        self.send(first)
        self.send([*first, msg("assistant", "Done.")])
        self.assertEqual(len(self.headroom.bodies), 1)
        self.send([*first, msg("assistant", "Done."), msg("user", "and the tests"), *run(2, "pytest", "y" * 800)])
        self.assertEqual(self.headroom.bodies[-1]["config"]["frozen_message_count"], 3)

    def test_where_headroom_does_not_answer_the_results_go_as_they_are_and_stay_so(self) -> None:
        self.headroom.failures = 1
        first = [msg("user", "fix the bug"), *run(1, "cat app.log", "x" * 500)]
        self.assertEqual(self.send(first), first)
        sent = self.send([*first, *run(2, "pytest", "y" * 800)])
        self.assertEqual(self.headroom.bodies[-1]["config"]["frozen_message_count"], 3)
        self.assertEqual([item.get("output") for item in sent[1:]], [None, "x" * 500, None, "[800 chars]"])

    def test_on_anthropic_a_messages_results_go_as_one_and_of_several_tools_as_they_are(self) -> None:
        def turn(*tools: str) -> list[dict[str, Any]]:
            uses = [{"type": "tool_use", "id": f"t{k}", "name": tool, "input": {"path": f"f{k}"}} for k, tool in enumerate(tools)]
            results = [{"type": "tool_result", "tool_use_id": f"t{k}", "content": f"{k}" * 500} for k in range(len(tools))]
            return [{"role": "user", "content": "fix the bug"}, {"role": "assistant", "content": uses},
                    {"role": "user", "content": results}]

        sent = self.send(turn("Bash", "Bash"), AnthropicMessages())
        self.assertEqual([m.get("tool_call_id") for m in self.headroom.bodies[-1]["messages"] if m["role"] == "tool"],
                         ["call_1_0"])
        self.assertEqual([part["content"] for part in sent[-1]["content"]], [[{"type": "text", "text": "[1001 chars]"}], ""])
        sent = self.send(turn("Read", "Bash"), AnthropicMessages())  # (Headroom keeps a Read's output: Relay can't split it)
        self.assertEqual(sent[-1], turn("Read", "Bash")[-1])

    def test_a_history_not_as_forwarded_is_compressed_whole_and_the_settings_go_along(self) -> None:
        items = (Item(Kind.USER, "fix the bug"), Item(Kind.TOOL_CALL, 'sh({"cmd": "cat app.log"})'),
                 Item(Kind.TOOL_RESULT, "x" * 500))
        request = Request(items, items, frozenset({0, 1, 3}), 1000, 200_000, state={"sent": 3, "last": "another"})
        context = Headroom(mode="ccr", post=self.headroom).plan(request, None)
        body = self.headroom.bodies[-1]
        self.assertEqual((body["config"], body["token_budget"]), ({"frozen_message_count": 0, "mode": "ccr"}, 200_000))
        self.assertEqual(context.items[-1].text, "[500 chars]")


if __name__ == "__main__":
    unittest.main()
