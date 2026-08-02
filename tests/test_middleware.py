from __future__ import annotations

import json
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

from contextlab import (
    AgentFold,
    CODEX_COMPACTION_PROMPT,
    CodexPromptCompaction,
    ContextManagingOpenAI,
    NativeCompaction,
    OfficialRLMAdapter,
    RollbackFolding,
    RollingMemory,
    SlidingWindow,
    StandaloneCompaction,
    ThresholdCompaction,
)


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class FakeInputTokens:
    def __init__(self, owner: "FakeResponses") -> None:
        self.owner = owner

    def count(self, **request):
        self.owner.count_calls.append(request)
        return SimpleNamespace(input_tokens=self.owner.token_count)


class FakeResponses:
    def __init__(self, *, token_count: int = 1, task_outputs=None, fold_start: int = 1) -> None:
        self.token_count = token_count
        self.task_outputs = list(task_outputs or [[message("assistant", "task result")]])
        self.fold_start = fold_start
        self.create_calls: list[dict] = []
        self.compact_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.input_tokens = FakeInputTokens(self)

    def create(self, **request):
        self.create_calls.append(request)
        if request.get("text", {}).get("format", {}).get("name") == "context_fold":
            return SimpleNamespace(
                output=[],
                output_text=json.dumps(
                    {
                        "start_index": self.fold_start,
                        "summary": "folded coding state",
                        "reason": "completed branch",
                    }
                ),
            )
        last = request.get("input", [{}])[-1]
        if last.get("content") == CODEX_COMPACTION_PROMPT:
            return SimpleNamespace(output=[], output_text="durable coding handoff")
        output = self.task_outputs.pop(0)
        response = SimpleNamespace(output=output, output_text="")
        if request.get("stream") is True:
            return iter(
                [
                    SimpleNamespace(type="response.output_item.done", item=output[-1]),
                    SimpleNamespace(type="response.completed", response=response),
                ]
            )
        return response

    def compact(self, **request):
        self.compact_calls.append(request)
        return SimpleNamespace(
            output=[
                message("user", "retained user state"),
                {"type": "compaction", "encrypted_content": "opaque"},
            ]
        )

    def stream(self, **request):
        self.create_calls.append(request)
        output = self.task_outputs.pop(0)
        response = SimpleNamespace(output=output, output_text="")

        class Manager:
            def __enter__(self):
                return iter(
                    [
                        SimpleNamespace(type="response.output_item.done", item=output[-1]),
                        SimpleNamespace(type="response.completed", response=response),
                    ]
                )

            def __exit__(self, exc_type, exc, traceback):
                return False

        return Manager()


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


class MiddlewareTests(unittest.TestCase):
    def test_local_compaction_item_is_the_checkpoint_for_append_only_replay(self) -> None:
        api = FakeResponses(
            token_count=500,
            task_outputs=[
                [message("assistant", "first result")],
                [message("assistant", "second result")],
            ],
        )
        client = ContextManagingOpenAI(
            FakeClient(api), ThresholdCompaction(compact_threshold=100)
        )
        trajectory = [message("user", "old request")]

        first = client.responses.create(model="task", input=trajectory)
        self.assertEqual(first.output[0]["type"], "compaction")
        self.assertTrue(
            first.output[0]["encrypted_content"].startswith("contextlab:v1:")
        )
        trajectory.extend(first.output)
        trajectory.append(message("user", "continue"))

        api.token_count = 1
        client.responses.create(model="task", input=trajectory)
        sent = api.create_calls[-1]["input"]
        self.assertTrue(
            any("durable coding handoff" in item.get("content", "") for item in sent)
        )
        self.assertEqual(sent[-2]["content"], "first result")
        self.assertEqual(sent[-1]["content"], "continue")

    def test_official_context_management_shape_selects_local_threshold(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(
            FakeClient(api), ThresholdCompaction(compact_threshold=10_000)
        )
        response = client.responses.create(
            model="task",
            input=[message("user", "long")],
            context_management=[{"type": "compaction", "compact_threshold": 100}],
        )
        self.assertEqual(len(api.create_calls), 2)
        self.assertNotIn("context_management", api.create_calls[-1])
        self.assertEqual(response.output[0]["type"], "compaction")

    def test_local_compaction_item_detects_corruption(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(
            FakeClient(api), ThresholdCompaction(compact_threshold=100)
        )
        first = client.responses.create(
            model="task", input=[message("user", "long")]
        )
        marker = first.output[0].model_dump()
        marker["id"] = "cmp_local_corrupted"
        with self.assertRaisesRegex(ValueError, "integrity"):
            client.responses.create(
                model="task",
                input=[message("user", "long"), marker, *first.output[1:]],
            )

    def test_responses_compact_uses_selected_operator_and_returns_canonical_window(self) -> None:
        api = FakeResponses(token_count=1)
        client = ContextManagingOpenAI(
            FakeClient(api), ThresholdCompaction(compact_threshold=10_000)
        )
        compacted = client.responses.compact(
            model="task", input=[message("user", "full trajectory")]
        )
        self.assertEqual(compacted.object, "response.compaction")
        self.assertIn("durable coding handoff", compacted.output[-1]["content"])
        self.assertEqual(len(api.create_calls), 1)

    def test_sliding_window_is_available_through_responses_compact(self) -> None:
        api = FakeResponses()
        client = ContextManagingOpenAI(
            FakeClient(api), SlidingWindow(max_items=2)
        )
        compacted = client.responses.compact(
            model="task",
            input=[
                message("developer", "rules"),
                message("user", "one"),
                message("assistant", "two"),
                message("user", "three"),
                message("assistant", "four"),
            ],
        )
        self.assertEqual(
            [item["content"] for item in compacted.output],
            ["rules", "three", "four"],
        )

    def test_native_compaction_round_trips_the_official_output_item(self) -> None:
        first_output = [
            {"type": "compaction", "encrypted_content": "opaque"},
            message("assistant", "result one"),
        ]
        api = FakeResponses(task_outputs=[first_output, [message("assistant", "result two")]])
        client = ContextManagingOpenAI(FakeClient(api), NativeCompaction(1000))
        trajectory = [message("user", "start")]

        first = client.responses.create(model="test", input=trajectory, store=False)
        trajectory.extend(first.output)
        trajectory.append(message("user", "continue"))
        second = client.responses.create(
            model="test",
            input=trajectory,
            store=False,
        )

        sent = api.create_calls[-1]["input"]
        self.assertEqual(sent[0]["type"], "compaction")
        self.assertEqual(sent[-1]["content"], "continue")
        self.assertFalse(any(item.get("content") == "start" for item in sent))
        self.assertIn("context_management", api.create_calls[-1])
        self.assertEqual(second.output[-1]["content"], "result two")

    def test_codex_prompt_replaces_the_old_window_and_preserves_request_fields(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(
            FakeClient(api),
            ThresholdCompaction(compact_threshold=100, manager_model="manager"),
        )
        response = client.responses.create(
            model="task",
            instructions="coding instructions",
            tools=[{"type": "function", "name": "shell"}],
            input=[
                message("developer", "repo rules"),
                message("user", "old request"),
                message("assistant", "old work"),
                message("user", "current request"),
            ],
        )

        summary_call, task_call = api.create_calls
        self.assertEqual(summary_call["model"], "manager")
        self.assertEqual(summary_call["input"][-1]["content"], CODEX_COMPACTION_PROMPT)
        self.assertEqual(task_call["tools"][0]["name"], "shell")
        self.assertEqual(task_call["input"][0]["content"], "repo rules")
        self.assertIn("durable coding handoff", task_call["input"][-1]["content"])
        self.assertEqual(response.output[0]["type"], "compaction")

    def test_default_is_our_threshold_compaction_not_provider_compaction(self) -> None:
        api = FakeResponses(token_count=1)
        client = ContextManagingOpenAI(FakeClient(api))
        client.responses.create(model="task", input=[message("user", "short")])
        self.assertNotIn("context_management", api.create_calls[-1])
        self.assertEqual(client.responses.strategy.name, "threshold_compaction")

    def test_rollback_folding_is_a_real_model_selected_suffix_return(self) -> None:
        api = FakeResponses(token_count=500, fold_start=1)
        client = ContextManagingOpenAI(
            FakeClient(api), RollbackFolding(compact_threshold=100, manager_model="manager")
        )
        client.responses.create(
            model="task",
            input=[
                message("developer", "rules"),
                message("user", "branch start"),
                message("assistant", "branch work"),
            ],
        )
        task_input = api.create_calls[-1]["input"]
        self.assertEqual(len(task_input), 2)
        self.assertEqual(task_input[0]["content"], "rules")
        self.assertIn("folded coding state", task_input[1]["content"])

    def test_agentfold_preserves_the_latest_observation(self) -> None:
        api = FakeResponses(token_count=500, fold_start=1)
        client = ContextManagingOpenAI(
            FakeClient(api), AgentFold(compact_threshold=100, manager_model="manager")
        )
        client.responses.create(
            model="task",
            input=[
                message("developer", "rules"),
                message("assistant", "old step"),
                message("user", "new observation"),
            ],
        )
        task_input = api.create_calls[-1]["input"]
        self.assertEqual(task_input[-1]["content"], "new observation")
        self.assertIn("folded coding state", task_input[-2]["content"])

    def test_agentfold_keeps_a_tool_observation_with_its_function_call(self) -> None:
        api = FakeResponses(token_count=500, fold_start=1)
        client = ContextManagingOpenAI(
            FakeClient(api), AgentFold(compact_threshold=100, manager_model="manager")
        )
        client.responses.create(
            model="task",
            input=[
                message("developer", "rules"),
                message("user", "old task"),
                message("assistant", "old result"),
                {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
        )
        task_input = api.create_calls[-1]["input"]
        self.assertEqual(task_input[-2]["type"], "function_call")
        self.assertEqual(task_input[-1]["type"], "function_call_output")
        self.assertEqual(task_input[-2]["call_id"], task_input[-1]["call_id"])

    def test_rolling_memory_updates_after_the_task_response(self) -> None:
        api = FakeResponses(
            task_outputs=[
                [message("assistant", "first task")],
                [message("assistant", "second task")],
            ]
        )
        client = ContextManagingOpenAI(
            FakeClient(api), RollingMemory(manager_model="manager")
        )
        response = client.responses.create(
            model="task",
            input=[message("developer", "rules"), message("user", "do work")],
        )
        self.assertEqual(api.create_calls[0]["model"], "task")
        self.assertEqual(api.create_calls[1]["model"], "manager")
        self.assertEqual(response.output[-1]["type"], "compaction")
        trajectory = [
            message("developer", "rules"),
            message("user", "do work"),
            *response.output,
            message("user", "continue"),
        ]
        client.responses.create(model="task", input=trajectory)
        active = api.create_calls[2]["input"]
        self.assertEqual(active[0]["content"], "rules")
        self.assertIn("durable coding handoff", active[-2]["content"])
        self.assertEqual(active[-1]["content"], "continue")

    def test_standalone_compaction_uses_official_compact_output_as_is(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(
            FakeClient(api), StandaloneCompaction(compact_threshold=100)
        )
        client.responses.create(model="task", input=[message("user", "long history")])
        self.assertEqual(len(api.compact_calls), 1)
        self.assertEqual(api.create_calls[-1]["input"][1]["type"], "compaction")

    def test_stream_final_response_is_available_after_completion(self) -> None:
        api = FakeResponses(task_outputs=[[message("assistant", "streamed")]])
        client = ContextManagingOpenAI(FakeClient(api), NativeCompaction())
        stream = client.responses.create(
            model="test", input=[message("user", "stream")], stream=True
        )
        list(stream)
        self.assertIsNotNone(stream.final_response)
        self.assertEqual(stream.final_response.output[-1]["content"], "streamed")

    def test_stream_completed_event_contains_the_local_compaction_item(self) -> None:
        api = FakeResponses(
            token_count=500, task_outputs=[[message("assistant", "streamed")]]
        )
        client = ContextManagingOpenAI(
            FakeClient(api), ThresholdCompaction(compact_threshold=100)
        )
        events = list(
            client.responses.create(
                model="test", input=[message("user", "stream")], stream=True
            )
        )
        completed = events[-1]
        self.assertEqual(completed.response.output[0]["type"], "compaction")
        self.assertEqual(completed.response.output[-1]["content"], "streamed")

    def test_stream_manager_final_response_is_available_after_completion(self) -> None:
        api = FakeResponses(task_outputs=[[message("assistant", "streamed")]])
        client = ContextManagingOpenAI(FakeClient(api), NativeCompaction())
        with client.responses.stream(
            model="test", input=[message("user", "stream manager")]
        ) as stream:
            list(stream)
            self.assertIsNotNone(stream.final_response)
            self.assertEqual(stream.final_response.output[-1]["content"], "streamed")


class RLMTests(unittest.TestCase):
    def test_official_adapter_defaults_to_current_persistent_compacted_mode(self) -> None:
        captured = {}
        closed = []

        class FakeRLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def completion(self, prompt, root_prompt=None):
                return {"prompt": prompt, "root_prompt": root_prompt}

            def cleanup(self):
                closed.append(True)

        fake_module = types.ModuleType("rlm")
        fake_module.RLM = FakeRLM
        with patch.dict("sys.modules", {"rlm": fake_module}):
            adapter = OfficialRLMAdapter(model="test-model")
            result = adapter.completion(
                {"input": [message("user", "hello")]}, root_prompt="continue"
            )
            adapter.close()
        self.assertTrue(captured["persistent"])
        self.assertTrue(captured["compaction"])
        self.assertEqual(result["root_prompt"], "continue")
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
