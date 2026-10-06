"""The RLM strategies, with a stand-in for the authors' package (its own code runs in Docker)."""

from __future__ import annotations

import json
import types
import unittest
from typing import Any
from unittest import mock

from relay.core.engine import Engine
from relay.core.ir import Item, Kind, Request
from relay.harnesses import Harness
from relay.protocols import OpenAIResponses
from relay.strategies import RLM, PersistentRLM
from relay.strategies import rlm as module


class Runtime:
    """The authors' RLM as the strategy drives it: one model call per completion, then an answer."""

    made: list[Runtime] = []

    def __init__(self, **options: Any) -> None:
        self.options, self.contexts, self.closed = options, [], False
        self.backend_kwargs: dict[str, Any] = {}
        Runtime.made.append(self)

    def completion(self, context: Any, root_prompt: str) -> Any:
        self.contexts.append(context)
        self.prompt = root_prompt
        answer = self.backend_kwargs["client"].completion(
            [{"role": "system", "content": "RLM SYSTEM"}, {"role": "user", "content": "go"}])
        return types.SimpleNamespace(response=answer)

    def close(self) -> None:
        self.closed = True


class Model:
    def __init__(self) -> None:
        self.asked: list[list[Item]] = []

    def complete(self, items: list[Item]) -> str:
        self.asked.append(items)
        return "Run `cat file_2.txt`."


def request(*items: Item, conversation: str = "c") -> Request:
    return Request(items, items, frozenset(range(len(items) + 1)), 1_000, None, conversation=conversation)


TASK = Item(Kind.USER, "read the files", 1)
CALL, RESULT = Item(Kind.TOOL_CALL, 'sh({"cmd": "cat file_1.txt"})', 2), Item(Kind.TOOL_RESULT, "CODE: c1", 3)


class RLMTests(unittest.TestCase):
    def setUp(self) -> None:
        Runtime.made = []
        patcher = mock.patch.object(module, "_official", return_value=types.SimpleNamespace(RLM=Runtime))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_conversation_is_the_context_and_the_answer_the_handoff(self) -> None:
        model = Model()
        context = RLM().plan(request(TASK, CALL, RESULT), model)
        runtime, = Runtime.made
        self.assertEqual(runtime.contexts, [[{"role": "user", "content": "read the files"},
                                             {"role": "tool_call", "content": CALL.text},
                                             {"role": "tool", "content": "CODE: c1"}]])
        self.assertEqual(runtime.options["persistent"], False)
        self.assertEqual(model.asked, [[Item(Kind.SYSTEM, "RLM SYSTEM"), Item(Kind.USER, "go")]])  # the task's model
        handoff, = context.items
        self.assertIs(handoff.kind, Kind.SUMMARY)
        self.assertIn("Run `cat file_2.txt`.", handoff.text)
        RLM().plan(request(TASK, CALL, RESULT), model)
        self.assertEqual(len(Runtime.made), 2)  # a fresh query every request

    def test_persistent_adds_only_what_is_new_and_starts_over_when_the_history_changes(self) -> None:
        strategy, model = PersistentRLM(), Model()
        strategy.plan(request(TASK), model)
        strategy.plan(request(TASK, CALL, RESULT), model)
        runtime, = Runtime.made
        self.assertEqual(runtime.options["persistent"], True)
        self.assertEqual([len(c) for c in runtime.contexts], [1, 2])  # context_0, then context_1: the call and result
        self.assertIn("context_k", runtime.prompt)
        strategy.plan(request(Item(Kind.USER, "something else", 1)), model)  # not a continuation
        self.assertTrue(runtime.closed)
        self.assertEqual(len(Runtime.made), 2)
        strategy.plan(request(TASK, conversation="other"), model)
        self.assertEqual(len(strategy.sessions), 2)

    def test_through_the_engine_the_model_gets_the_rlms_own_request(self) -> None:
        codec, sent = OpenAIResponses(), []

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            sent.append(body)
            return 200, {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Run `cat file_2.txt`."}]}]}

        history = [{"type": "message", "role": "developer", "content": "harness rules"},
                   {"type": "message", "role": "user", "content": "read the files"}]
        body = {"model": "m", "instructions": "harness", "input": history, "tools": [{"type": "function", "name": "sh"}]}
        exchange = Engine(RLM()).prepare(codec, Harness(), body, tenant="t", post=upstream)
        rlm_call, = sent
        self.assertEqual(rlm_call["instructions"], "RLM SYSTEM")
        self.assertNotIn("tools", rlm_call)
        self.assertEqual([codec.classify(i).text for i in rlm_call["input"]], ["go"])
        forwarded = exchange.body
        self.assertEqual(forwarded["tools"], body["tools"])  # the step itself is the harness's model's, with its tools
        self.assertIn("Run `cat file_2.txt`.", json.dumps(forwarded["input"]))
        self.assertNotIn("read the files", json.dumps(forwarded["input"]))


if __name__ == "__main__":
    unittest.main()
