"""The RLM strategies, with a stand-in for the authors' package (its own code runs in Docker)."""

from __future__ import annotations

import json
import types
import unittest
from typing import Any
from unittest import mock

from relay.core.engine import Engine, Exchange
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

    def test_native_transport_retry_reuses_successful_handoff_without_advancing_runtime(self) -> None:
        for strategy_type in (RLM, PersistentRLM):
            with self.subTest(strategy=strategy_type.__name__):
                Runtime.made = []
                sent = []

                def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
                    sent.append(body)
                    return 200, {"status": "completed", "output": [{
                        "type": "message", "role": "assistant", "content": [{
                            "type": "output_text", "text": f"Run command {len(sent)}."}]}]}

                engine = Engine(strategy_type())
                body = {"model": "m", "tools": [{"type": "function", "name": "sh"}],
                        "input": [{"type": "message", "role": "user", "content": "read the files"}]}

                def prepare(value: dict[str, Any], **kwargs: Any) -> Exchange:
                    return engine.prepare(OpenAIResponses(), Harness(), value, tenant="t", post=upstream, **kwargs)

                first = prepare(body)
                # The native send receives HTTP 429, so no response usage is recorded.
                retried = prepare(body)
                self.assertEqual(len(sent), 1)
                self.assertEqual(retried.body, first.body)
                self.assertEqual(len(Runtime.made), 1)
                self.assertEqual(Runtime.made[0].contexts, [[{"role": "user", "content": "read the files"}]])

                changed = {**body, "input": [*body["input"],
                           {"type": "message", "role": "user", "content": "also inspect the second file"}]}
                prepare(changed)
                self.assertEqual(len(sent), 2)
                prepare(changed, force=True)
                self.assertEqual(len(sent), 3)  # Forced replanning keeps the existing behavior.

    def test_failed_or_empty_completion_does_not_cache_a_handoff(self) -> None:
        for strategy_type in (RLM, PersistentRLM):
            for failure in (RuntimeError("synthetic failure"), types.SimpleNamespace(response="  ")):
                with self.subTest(strategy=strategy_type.__name__, failure=str(failure)):
                    engine = Engine(strategy_type(), retry_after=0)
                    body = {"model": "m", "tools": [{"type": "function", "name": "sh"}],
                            "input": [{"type": "message", "role": "user", "content": "read the files"}]}

                    def prepare() -> Exchange:
                        return engine.prepare(OpenAIResponses(), Harness(), body, tenant="t",
                                              post=lambda _: self.fail("The stand-in runtime needs no upstream"))

                    with mock.patch.object(Runtime, "completion", side_effect=[
                        failure, types.SimpleNamespace(response="Run `cat file_2.txt`.")
                    ]) as completion:
                        with self.assertLogs("relay", level="WARNING"):
                            failed = prepare()
                        self.assertEqual(failed.body, body)
                        successful = prepare()
                        retried = prepare()
                    self.assertEqual(completion.call_count, 2)
                    self.assertEqual(retried.body, successful.body)
                    self.assertNotEqual(successful.body, body)

    def test_invalid_saved_handoff_is_recomputed(self) -> None:
        from dataclasses import replace

        for invalid in (None, 42, "", "  "):
            with self.subTest(invalid=invalid):
                strategy, model = PersistentRLM(), Model()
                initial = request(TASK)
                successful = strategy.plan(initial, model)
                saved = {**successful.state, "rlm_handoff": invalid}
                strategy.plan(replace(initial, state=saved), model)
                self.assertEqual(len(model.asked), 2)

    def test_direct_plan_only_reuses_a_handoff_when_the_caller_passes_state(self) -> None:
        from dataclasses import replace

        for strategy_type in (RLM, PersistentRLM):
            with self.subTest(strategy=strategy_type.__name__):
                strategy, model = strategy_type(), Model()
                initial = request(TASK)
                first = strategy.plan(initial, model)
                strategy.plan(initial, model)
                self.assertEqual(len(model.asked), 2)
                cached = strategy.plan(replace(initial, state=first.state), model)
                self.assertEqual(cached, first)
                self.assertEqual(len(model.asked), 2)

    def test_engine_keeps_persistent_session_across_first_tool_step_and_divergence(self) -> None:
        engine = Engine(PersistentRLM())
        sent: list[dict[str, Any]] = []

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            sent.append(body)
            return 200, {"status": "completed", "output": [{
                "type": "message", "role": "assistant", "content": [{
                    "type": "output_text", "text": "Run the next command."}]}]}

        def step(index: int, output: str) -> list[dict[str, Any]]:
            return [{"type": "function_call", "call_id": f"call-{index}", "name": "sh", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": f"call-{index}", "output": output}]

        def prepare(items: list[dict[str, Any]], **kwargs: Any) -> Exchange:
            body = {"model": "m", "tools": [{"type": "function", "name": "sh"}], "input": items}
            return engine.prepare(OpenAIResponses(), Harness(), body, tenant="t", post=upstream, **kwargs)

        initial = [{"type": "message", "role": "user", "content": "read the files"}]
        first = prepare(initial)
        after_tool = [*initial, *step(1, "first result")]
        second = prepare(after_tool)
        self.assertNotEqual(first.conversation, second.conversation)
        runtime, = Runtime.made
        self.assertEqual([len(context) for context in runtime.contexts], [1, 2])
        self.assertEqual(second.state["strategy"]["rlm_session_key"], first.conversation)
        self.assertEqual([item["role"] for item in runtime.contexts[1]], ["tool_call", "tool"])

        prepare(after_tool)  # Transport retry must also keep this session untouched.
        self.assertEqual(len(sent), 2)
        after_second_tool = [*after_tool, *step(2, "second result")]
        prepare(after_second_tool)
        self.assertEqual([len(context) for context in runtime.contexts], [1, 2, 2])
        prepare(after_second_tool, force=True)
        self.assertEqual(runtime.contexts[-1], [])

        prepare([*initial, *step(1, "edited result")])
        self.assertTrue(runtime.closed)
        self.assertEqual(len(Runtime.made), 2)
        self.assertEqual([len(context) for context in Runtime.made[-1].contexts], [3])

    def test_stable_persistent_keys_keep_the_existing_session_limit(self) -> None:
        strategy = PersistentRLM()
        engine = Engine(strategy)

        def upstream(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return 200, {"status": "completed", "output": [{
                "type": "message", "role": "assistant", "content": [{
                    "type": "output_text", "text": "Run the next command."}]}]}

        for number in range(module.SESSIONS + 1):
            body = {"model": "m", "tools": [{"type": "function", "name": "sh"}],
                    "input": [{"type": "message", "role": "user", "content": f"Task {number}"}]}
            engine.prepare(OpenAIResponses(), Harness(), body, tenant="t", post=upstream)
        self.assertEqual(len(strategy.sessions), module.SESSIONS)
        self.assertTrue(Runtime.made[0].closed)
        self.assertFalse(Runtime.made[-1].closed)


if __name__ == "__main__":
    unittest.main()
