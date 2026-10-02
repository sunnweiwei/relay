"""Codec behaviour, checked against traffic recorded from real harnesses."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from relay.core.ir import Item, Kind
from relay.harnesses import ClaudeCode, Codex, detect
from relay.protocols import AnthropicMessages, Gemini, OpenAIChat, OpenAIResponses

TRACES = Path(__file__).parent / "traces"


def load(name: str) -> dict:
    return json.loads((TRACES / name).read_text())


class RecordedTraceTests(unittest.TestCase):
    """The prefix store relies on each request extending the previous one."""

    cases = [
        ("codex-0.160.0.json", OpenAIResponses(), Codex(), "input"),
        ("claude-code-2.1.283.json", AnthropicMessages(), ClaudeCode(), "messages"),
    ]

    def test_requests_are_append_only_after_canonicalization(self) -> None:
        for name, codec, _, key in self.cases:
            requests = [[codec.canonical(item) for item in r[key]] for r in load(name)["requests"]]
            for previous, current in zip(requests, requests[1:]):
                with self.subTest(trace=name, length=len(current)):
                    self.assertEqual(current[: len(previous)], previous)

    def test_harness_is_detected_from_its_user_agent(self) -> None:
        for name, _, harness, _ in self.cases:
            self.assertIs(type(detect({"user-agent": load(name)["user_agent"]})), type(harness))

    def test_codex_context_and_user_messages(self) -> None:
        codec, harness = OpenAIResponses(), Codex()
        items = load("codex-0.160.0.json")["requests"][-1]["input"]
        kinds = [harness.refine(codec.classify(item)).kind for item in items]
        self.assertEqual(kinds[:3], [Kind.SYSTEM, Kind.CONTEXT, Kind.USER])
        self.assertEqual(kinds[3:6], [Kind.REASONING, Kind.TOOL_CALL, Kind.TOOL_RESULT])
        new_turn = kinds.index(Kind.ASSISTANT) + 1
        self.assertEqual(kinds[new_turn : new_turn + 3], [Kind.SYSTEM, Kind.CONTEXT, Kind.USER])

    def test_claude_code_context_and_user_messages(self) -> None:
        codec, harness = AnthropicMessages(), ClaudeCode()
        messages = load("claude-code-2.1.283.json")["requests"][-1]["messages"]
        kinds = [harness.refine(codec.classify(m)).kind for m in messages]
        self.assertEqual(kinds[:4], [Kind.USER, Kind.SYSTEM, Kind.TOOL_CALL, Kind.TOOL_RESULT])
        self.assertIn(Kind.ASSISTANT, kinds)
        self.assertEqual(kinds[kinds.index(Kind.ASSISTANT) + 1], Kind.USER)


class OpenAIResponsesTests(unittest.TestCase):
    codec = OpenAIResponses()

    def test_boundaries_keep_calls_with_outputs_and_reasoning_with_its_item(self) -> None:
        items = [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "x"},
            {"type": "custom_tool_call_output", "call_id": "b", "output": "y"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        ]
        self.assertEqual(self.codec.boundaries(items), {0, 1, 6, 7})

    def test_classify_marks_media_and_tool_items(self) -> None:
        image = {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "u"}]}
        self.assertTrue(self.codec.classify(image).media)
        self.assertEqual(self.codec.classify({"type": "local_shell_call", "call_id": "c"}).kind, Kind.TOOL_CALL)
        self.assertEqual(self.codec.classify({"type": "shell_call_output", "call_id": "c"}).kind, Kind.TOOL_RESULT)
        self.assertEqual(self.codec.classify({"type": "compaction", "encrypted_content": "x"}).kind, Kind.SUMMARY)

    def test_stateful_requests_are_not_managed(self) -> None:
        self.assertTrue(self.codec.managed({"input": []}))
        self.assertFalse(self.codec.managed({"input": "hi"}))
        self.assertFalse(self.codec.managed({"input": [], "previous_response_id": "r"}))

    def test_summary_request_continues_the_same_request_without_tools(self) -> None:
        body = {"model": "m", "instructions": "be", "tools": [{"type": "function"}], "stream": True,
                "prompt_cache_key": "k", "input": [{"type": "message", "role": "user", "content": "x"}]}
        request = self.codec.summary_request(body, body["input"], "SUMMARIZE")
        self.assertEqual(request["input"][-1]["content"][0]["text"], "SUMMARIZE")
        # streaming follows the harness (Codex's ChatGPT backend only streams)
        self.assertEqual((request["stream"], request["store"], request["tool_choice"]), (True, False, "none"))
        self.assertEqual((request["prompt_cache_key"], request["tools"]), ("k", body["tools"]))

    def test_codex_resume_writes_absent_reasoning_content_as_null(self) -> None:
        reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
        self.assertEqual(self.codec.canonical(reasoning), self.codec.canonical({**reasoning, "content": None}))
        self.assertNotEqual(self.codec.canonical(reasoning), self.codec.canonical({**reasoning, "encrypted_content": "x"}))
        visible = {**reasoning, "content": [{"type": "reasoning_text", "text": "new"}]}
        self.assertNotEqual(self.codec.canonical(reasoning), self.codec.canonical(visible))
        message = {"type": "message", "role": "assistant"}
        self.assertNotEqual(self.codec.canonical(message), self.codec.canonical({**message, "content": None}))
        summarized = {**reasoning, "summary": [{"type": "summary_text", "text": "thinking"}], "content": []}
        self.assertEqual(self.codec.canonical(reasoning), self.codec.canonical(summarized))  # OpenClaw resume

    def test_usage_and_overflow(self) -> None:
        completed = {"type": "response.completed", "response": {"usage": {"input_tokens": 7}}}
        self.assertEqual(self.codec.usage(completed), 7)
        self.assertEqual(self.codec.usage({"object": "response", "usage": {"input_tokens": 9}}), 9)
        self.assertIsNone(self.codec.usage({"type": "response.created", "response": {"usage": None}}))
        error = {"error": {"code": "context_length_exceeded", "message": "too long"}}
        self.assertTrue(self.codec.is_overflow(400, error))
        self.assertFalse(self.codec.is_overflow(400, {"error": {"message": "bad tool"}}))


class AnthropicMessagesTests(unittest.TestCase):
    codec = AnthropicMessages()

    def test_canonical_ignores_cache_control_and_string_content(self) -> None:
        listed = {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
        self.assertEqual(self.codec.canonical(listed), self.codec.canonical({"role": "user", "content": "hi"}))

    def test_boundaries_never_split_tool_use_from_its_result(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
        ]
        self.assertEqual(self.codec.boundaries(messages), {0, 1, 3})

    def test_summary_request_disables_tools_and_caps_output(self) -> None:
        body = {"model": "m", "max_tokens": 128_000, "stream": True, "tools": [{"name": "Bash"}], "messages": []}
        request = self.codec.summary_request(body, [], "SUMMARIZE")
        self.assertEqual(request["tool_choice"], {"type": "none"})
        self.assertEqual((request["stream"], request["max_tokens"]), (False, 32_000))
        self.assertEqual(request["messages"], [{"role": "user", "content": [{"type": "text", "text": "SUMMARIZE"}]}])

    def test_usage_counts_cached_prompt_tokens(self) -> None:
        start = {"type": "message_start", "message": {"usage": {
            "input_tokens": 5, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 100}}}
        self.assertEqual(self.codec.usage(start), 115)
        self.assertIsNone(self.codec.usage({"type": "message_delta", "usage": {"output_tokens": 3}}))
        zero = {"type": "message_start", "message": {"usage": {"input_tokens": 0, "output_tokens": 0}}}
        self.assertIsNone(self.codec.usage(zero))  # LiteLLM reports the count at the end instead
        self.assertEqual(self.codec.usage({"type": "message_delta", "usage": {"input_tokens": 8}}), 8)

    def test_system_messages_only_before_the_models_turn(self) -> None:
        system, user, summary = Item(Kind.SYSTEM, "env", 1), Item(Kind.USER, "task", 0), Item(Kind.SUMMARY, "s")
        self.assertEqual(self.codec.arrange((system, user, summary), mid_turn=True), (user, summary, system))
        self.assertEqual(self.codec.arrange((system, user, summary), mid_turn=False), (user, summary))

    def test_overflow(self) -> None:
        error = {"type": "error", "error": {"type": "invalid_request_error",
                                            "message": "prompt is too long: 210000 tokens > 200000 maximum"}}
        self.assertTrue(self.codec.is_overflow(400, error))


if __name__ == "__main__":
    unittest.main()


class OpenAIChatTests(unittest.TestCase):
    codec = OpenAIChat()

    def test_tool_calls_stay_with_their_results(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}},
                            {"id": "b", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "x"},
            {"role": "tool", "tool_call_id": "b", "content": "y"},
        ]
        self.assertEqual(self.codec.boundaries(messages), {0, 1, 2, 5})
        kinds = [self.codec.classify(m).kind for m in messages]
        self.assertEqual(kinds, [Kind.SYSTEM, Kind.USER, Kind.TOOL_CALL, Kind.TOOL_RESULT, Kind.TOOL_RESULT])

    def test_canonical_usage_and_summary_request(self) -> None:
        listed = {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
        self.assertEqual(self.codec.canonical(listed), self.codec.canonical({"role": "user", "content": "hi"}))
        self.assertEqual(self.codec.usage({"usage": {"prompt_tokens": 5}}), 5)
        request = self.codec.summary_request({"model": "m", "stream": True, "tools": [{}], "messages": []}, [], "S")
        self.assertEqual((request["stream"], request["tool_choice"]), (False, "none"))
        resumed = {"role": "tool", "tool_call_id": "c1", "content": "x", "name": "read"}
        self.assertEqual(self.codec.canonical(resumed), self.codec.canonical({**resumed, "name": None}))  # Hermes
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]}
        self.assertEqual(self.codec.canonical(call), self.codec.canonical({**call, "agent": "cli", "usage": {"total": 3}}))
        self.assertNotEqual(self.codec.canonical(call), self.codec.canonical({**call, "reasoning_content": "why"}))
        streamed = self.codec.stream_result([{"choices": [{"delta": {"content": "a"}}]},
                                             {"choices": [{"delta": {"content": "b"}}]}])
        self.assertEqual(self.codec.output_text(streamed), "ab")


class GeminiTests(unittest.TestCase):
    codec = Gemini()

    def test_function_calls_stay_with_their_responses(self) -> None:
        contents = [
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "s"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"output": "x"}}}]},
            {"role": "model", "parts": [{"text": "done"}]},
        ]
        self.assertEqual(self.codec.boundaries(contents), {0, 1, 3, 4})
        kinds = [self.codec.classify(c).kind for c in contents]
        self.assertEqual(kinds, [Kind.USER, Kind.TOOL_CALL, Kind.TOOL_RESULT, Kind.ASSISTANT])

    def test_snake_case_field_names_as_litellm_sends_them(self) -> None:
        contents = [
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"text": "reading"}, {"function_call": {"name": "bash", "args": {}}}]},
            {"role": "user", "parts": [{"function_response": {"name": "bash", "response": {"output": "x"}}}]},
            {"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": ""}}]},
        ]
        kinds = [self.codec.classify(c).kind for c in contents]
        self.assertEqual(kinds, [Kind.USER, Kind.TOOL_CALL, Kind.TOOL_RESULT, Kind.USER])
        self.assertTrue(self.codec.classify(contents[3]).media)
        self.assertEqual(self.codec.boundaries(contents), {0, 1, 3, 4})
        request = self.codec.summary_request({"contents": [], "generation_config": {"response_mime_type": "x"}}, [], "S")
        self.assertEqual(request["generation_config"], {})

    def test_gemini_cli_resume_rewrites_are_the_same_history(self) -> None:
        call = {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "real"}]}
        result = {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"output": "x"}}}]}
        resumed_call = {"role": "model", "parts": [{**call["parts"][0], "thoughtSignature": "skip_thought_signature_validator"}]}
        resumed_result = {"role": "user", "parts": result["parts"] * 2}
        self.assertEqual(self.codec.canonical(call), self.codec.canonical(resumed_call))
        self.assertEqual(self.codec.canonical(result), self.codec.canonical(resumed_result))
        other = {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"output": "y"}}}]}
        self.assertNotEqual(self.codec.canonical(result), self.codec.canonical(other))

    def test_summary_request_streams_and_usage(self) -> None:
        body = {"contents": [], "tools": [{"functionDeclarations": []}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 1}}
        request = self.codec.summary_request(body, [], "S")
        self.assertEqual(request["toolConfig"], {"functionCallingConfig": {"mode": "NONE"}})
        self.assertEqual(request["generationConfig"], {"temperature": 1})
        chunks = [{"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}]}}]},
                  {"candidates": [{"content": {"parts": [{"text": "sum"}]}}]},
                  {"candidates": [{"content": {"parts": [{"text": "mary"}]}}], "usageMetadata": {"promptTokenCount": 9}}]
        result = self.codec.stream_result(chunks)
        self.assertEqual((self.codec.output_text(result), self.codec.usage(result)), ("summary", 9))


class CompletionTests(unittest.TestCase):
    """A summary counts only if its response ended normally (Codex waits for response.completed)."""

    def check(self, codec, events: list[dict], cutoff: dict) -> None:
        self.assertTrue(codec.finished(codec.stream_result(events)))
        self.assertFalse(codec.finished(codec.stream_result(events[:-1])))  # the stream broke off
        self.assertFalse(codec.finished(codec.stream_result([*events[:-1], cutoff])))  # a token limit

    def test_openai_responses(self) -> None:
        text = {"type": "response.output_text.delta", "delta": "sum"}
        self.check(OpenAIResponses(), [text, {"type": "response.completed", "response": {"status": "completed"}}],
                   {"type": "response.incomplete", "response": {"status": "incomplete"}})

    def test_anthropic_messages(self) -> None:
        text = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "sum"}}
        self.check(AnthropicMessages(), [text, {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}],
                   {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}})

    def test_openai_chat(self) -> None:
        text = {"choices": [{"delta": {"content": "sum"}}]}
        self.check(OpenAIChat(), [text, {"choices": [{"delta": {}, "finish_reason": "stop"}]}],
                   {"choices": [{"delta": {}, "finish_reason": "length"}]})

    def test_gemini(self) -> None:
        text = {"candidates": [{"content": {"parts": [{"text": "sum"}]}}]}
        self.check(Gemini(), [text, {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}],
                   {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
