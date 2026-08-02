from __future__ import annotations

from types import SimpleNamespace
import unittest

import relay
from relay import Compact, ContextManagingOpenAI
import relay.strategies
from relay.strategies.compact import (
    CODEX_COMPACTION_PROMPT,
    CODEX_SUMMARY_PREFIX,
    _retain_user_messages,
)


def message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": text}


class FakeContextWindowError(Exception):
    status_code = 400
    body = {
        "error": {
            "code": "context_length_exceeded",
            "message": "maximum context length exceeded",
        }
    }


class FakeInputTokens:
    def __init__(self, owner: "FakeResponses") -> None:
        self.owner = owner

    def count(self, **request):
        self.owner.count_calls.append(request)
        return SimpleNamespace(input_tokens=self.owner.token_count)


class FakeResponses:
    def __init__(
        self,
        *,
        token_count: int = 1,
        task_outputs=None,
        summary_failures: int = 0,
    ) -> None:
        self.token_count = token_count
        self.task_outputs = list(task_outputs or [[message("assistant", "task result")]])
        self.summary_failures = summary_failures
        self.create_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.input_tokens = FakeInputTokens(self)

    def create(self, **request):
        self.create_calls.append(request)
        last = request.get("input", [{}])[-1]
        if last.get("content") == CODEX_COMPACTION_PROMPT:
            if self.summary_failures:
                self.summary_failures -= 1
                raise FakeContextWindowError()
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


class CompactTests(unittest.TestCase):
    def test_public_strategy_surface_only_contains_compact(self) -> None:
        self.assertEqual(relay.strategies.__all__, ["Compact"])
        self.assertEqual(
            [name for name in relay.__all__ if name.endswith("Compaction")], []
        )

    def test_checkpoint_replays_in_an_append_only_loop(self) -> None:
        api = FakeResponses(
            token_count=500,
            task_outputs=[
                [message("assistant", "first result")],
                [message("assistant", "second result")],
            ],
        )
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=100))
        trajectory = [message("user", "old request")]

        first = client.responses.create(model="task", input=trajectory)
        self.assertEqual(first.output[0]["type"], "compaction")
        self.assertTrue(first.output[0]["encrypted_content"].startswith("relay:v1:"))
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

    def test_official_context_management_shape_sets_the_local_threshold(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=10_000))
        response = client.responses.create(
            model="task",
            input=[message("user", "long")],
            context_management=[{"type": "compaction", "compact_threshold": 100}],
        )
        self.assertEqual(len(api.create_calls), 2)
        self.assertNotIn("context_management", api.create_calls[-1])
        self.assertEqual(response.output[0]["type"], "compaction")

    def test_checkpoint_detects_corruption(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=100))
        first = client.responses.create(model="task", input=[message("user", "long")])
        marker = first.output[0].model_dump()
        marker["id"] = "cmp_local_corrupted"
        with self.assertRaisesRegex(ValueError, "integrity"):
            client.responses.create(
                model="task",
                input=[message("user", "long"), marker, *first.output[1:]],
            )

    def test_responses_compact_returns_the_canonical_compact_window(self) -> None:
        api = FakeResponses()
        client = ContextManagingOpenAI(FakeClient(api), Compact())
        compacted = client.responses.compact(
            model="task", input=[message("user", "full trajectory")]
        )
        self.assertEqual(compacted.object, "response.compaction")
        self.assertIn("durable coding handoff", compacted.output[-1]["content"])
        self.assertEqual(len(api.create_calls), 1)

    def test_compaction_uses_the_codex_prompt_and_task_model(self) -> None:
        api = FakeResponses(token_count=500)
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=100))
        response = client.responses.create(
            model="task",
            instructions="coding instructions",
            reasoning={"effort": "high"},
            tools=[{"type": "function", "name": "shell"}],
            input=[
                message("developer", "repo rules"),
                message("user", "old request"),
                message("assistant", "old work"),
                message("user", "current request"),
            ],
        )

        summary_call, task_call = api.create_calls
        self.assertEqual(summary_call["model"], "task")
        self.assertEqual(summary_call["reasoning"], {"effort": "high"})
        self.assertEqual(summary_call["input"][-1]["content"], CODEX_COMPACTION_PROMPT)
        self.assertEqual(task_call["tools"][0]["name"], "shell")
        self.assertEqual(task_call["input"][0]["content"], "repo rules")
        self.assertIn("durable coding handoff", task_call["input"][-1]["content"])
        self.assertEqual(response.output[0]["type"], "compaction")

    def test_retains_real_user_messages_with_the_codex_token_budget(self) -> None:
        previous_summary = message(
            "user", f"{CODEX_SUMMARY_PREFIX}\nold compacted state"
        )
        compacted = _retain_user_messages(
            [
                message("developer", "rules"),
                message("user", "one"),
                previous_summary,
                message("user", "two"),
                message("user", "abcdefghijklmnop"),
            ],
            max_tokens=5,
        )
        self.assertEqual(
            [item["content"] for item in compacted],
            ["two", "abcdefghijklmnop"],
        )
        self.assertNotIn("old compacted state", str(compacted))

    def test_truncates_one_large_retained_user_message_like_codex(self) -> None:
        compacted = _retain_user_messages(
            [message("user", "abcdefghijklmnopqrstuvwxyz")],
            max_tokens=2,
        )
        self.assertIn("tokens truncated", compacted[0]["content"])

    def test_retries_summary_after_trimming_oldest_history_on_context_overflow(self) -> None:
        api = FakeResponses(token_count=500, summary_failures=1)
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=100))
        client.responses.create(
            model="task",
            input=[
                message("developer", "rules"),
                message("user", "oldest"),
                message("assistant", "work"),
                message("user", "current"),
            ],
        )
        summary_calls = [
            call
            for call in api.create_calls
            if call["input"][-1].get("content") == CODEX_COMPACTION_PROMPT
        ]
        self.assertEqual(len(summary_calls), 2)
        self.assertFalse(
            any(item.get("content") == "oldest" for item in summary_calls[1]["input"])
        )

    def test_default_is_compact(self) -> None:
        api = FakeResponses()
        client = ContextManagingOpenAI(FakeClient(api))
        client.responses.create(model="task", input=[message("user", "short")])
        self.assertNotIn("context_management", api.create_calls[-1])
        self.assertEqual(client.responses.strategy.name, "compact")

    def test_stream_final_response_is_available_after_completion(self) -> None:
        api = FakeResponses(task_outputs=[[message("assistant", "streamed")]])
        client = ContextManagingOpenAI(FakeClient(api), Compact())
        stream = client.responses.create(
            model="test", input=[message("user", "stream")], stream=True
        )
        list(stream)
        self.assertIsNotNone(stream.final_response)
        self.assertEqual(stream.final_response.output[-1]["content"], "streamed")

    def test_stream_completed_event_contains_the_checkpoint(self) -> None:
        api = FakeResponses(
            token_count=500, task_outputs=[[message("assistant", "streamed")]]
        )
        client = ContextManagingOpenAI(FakeClient(api), Compact(compact_threshold=100))
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
        client = ContextManagingOpenAI(FakeClient(api), Compact())
        with client.responses.stream(
            model="test", input=[message("user", "stream manager")]
        ) as stream:
            list(stream)
            self.assertIsNotNone(stream.final_response)
            self.assertEqual(stream.final_response.output[-1]["content"], "streamed")


if __name__ == "__main__":
    unittest.main()
