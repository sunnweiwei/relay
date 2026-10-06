"""Codec behaviour, checked against traffic recorded from real harnesses."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from relay.core.ir import Item, Kind, Media
from relay.core.tokens import bytes_per_token, item_tokens
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


    def test_canonical_ignores_metadata_and_argument_formatting(self) -> None:
        sent = {"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "c1", "name": "read",
                "arguments": '{"path":"a.txt","limit":20}'}
        resent = {"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"limit": 20, "path": "a.txt"}'}
        self.assertEqual(self.codec.canonical(sent), self.codec.canonical(resent))  # nanobot's compaction request
        self.assertNotEqual(self.codec.canonical(sent), self.codec.canonical({**resent, "arguments": '{"path": "b.txt"}'}))
        self.assertNotEqual(self.codec.canonical({"type": "item_reference", "id": "a"}),
                            self.codec.canonical({"type": "item_reference", "id": "b"}))


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
        spaced = {**call, "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{ }"}}]}
        self.assertEqual(self.codec.canonical(call), self.codec.canonical(spaced))
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


class OpaqueContentTests(unittest.TestCase):
    """Encrypted reasoning, server-side compactions and signatures cost prompt tokens though
    they carry no text: each codec reports their decoded size."""

    def test_each_protocol_reports_opaque_content(self) -> None:
        blob = "A" * 400  # base64: 300 bytes
        cases = [
            (OpenAIResponses(), {"type": "reasoning", "summary": [], "encrypted_content": blob}),
            (OpenAIResponses(), {"type": "compaction", "encrypted_content": blob}),
            (AnthropicMessages(), {"role": "assistant", "content": [{"type": "redacted_thinking", "data": blob}]}),
            (Gemini(), {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": blob}]}),
        ]
        for codec, item in cases:
            self.assertEqual(codec.classify(item).opaque, 300, item)
        signed = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": blob}]}
        self.assertEqual(AnthropicMessages().classify(signed).opaque, 0)  # a signature only verifies

    def test_token_estimates_follow_the_model_and_count_opaque_bytes(self) -> None:
        self.assertEqual((bytes_per_token("gpt-6-luna"), bytes_per_token("anthropic/claude-sonnet-5-5")), (4, 2.5))
        self.assertEqual(item_tokens(Item(Kind.REASONING, "abcd", opaque=300)), 76)
        self.assertEqual(item_tokens(Item(Kind.USER, "x" * 250), bytes_per_token("claude-x")), 100)


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


class StrategyAdditionsTests(unittest.TestCase):
    """What a strategy may put into a request: items of its own (with media), new content for a
    request item, a note at the end, a section of the system prompt; each written legally for its
    protocol, and what the API would reject found before it is sent."""

    IMAGE = Media("image", "image/png", "iVBORw0KGgo=")
    # A tool call and its result in each protocol.
    CALLS = {
        "openai_responses": ({"type": "function_call", "call_id": "c1", "name": "sh", "arguments": "{}"},
                             {"type": "function_call_output", "call_id": "c1", "output": "a long output"}),
        "anthropic_messages": ({"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                                                {"type": "tool_use", "id": "t1", "name": "sh", "input": {}}]},
                               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                                             "content": "a long output"}]}),
        "openai_chat": ({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "sh", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": "a long output"}),
        "gemini": ({"role": "model", "parts": [{"functionCall": {"name": "sh", "args": {}}}]},
                   {"role": "user", "parts": [{"functionResponse": {"name": "sh", "response": {"output": "a long output"}}}]}),
    }
    CODECS = (OpenAIResponses(), AnthropicMessages(), OpenAIChat(), Gemini())

    def test_written_items_read_back_with_their_media(self) -> None:
        for codec in self.CODECS:
            user = codec.classify(codec.write(Item(Kind.USER, "look", media=(self.IMAGE,))))
            self.assertEqual((user.kind, user.text, user.media), (Kind.USER, "look", (self.IMAGE,)), codec.name)
            assistant = codec.classify(codec.write(Item(Kind.ASSISTANT, "noted")))
            self.assertEqual((assistant.kind, assistant.text), (Kind.ASSISTANT, "noted"), codec.name)
            with self.assertRaises(ValueError):
                codec.write(Item(Kind.TOOL_CALL, "sh {}"))

    def test_an_edited_tool_result_still_answers_its_call(self) -> None:
        for codec in self.CODECS:
            call, result = self.CALLS[codec.name]
            edited = codec.edit(result, "short", ())
            self.assertEqual(codec.orphans([codec.user_message("go"), call, edited]), set(), codec.name)
            self.assertIn("short", codec.classify(edited).text)
            self.assertNotIn("a long output", json.dumps(edited))
            with self.assertRaises(ValueError):
                codec.edit(call, "another call", ())

    def test_orphans_are_a_call_without_its_result_and_a_result_without_its_call(self) -> None:
        for codec in self.CODECS:
            call, result = self.CALLS[codec.name]
            go = codec.user_message("go")
            self.assertEqual(codec.orphans([go, call]), {1}, codec.name)
            self.assertEqual(codec.orphans([go, result]), {1}, codec.name)
            if codec.name != "openai_responses":  # the Responses API pairs items by call id, not position
                self.assertEqual(codec.orphans([go, call, go, result]), {1, 3}, codec.name)

    def test_tool_results_carry_their_images(self) -> None:
        source = {"type": "base64", "media_type": "image/png", "data": self.IMAGE.data}
        result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                               "content": [{"type": "text", "text": "screen"},
                                                           {"type": "image", "source": source}]}]}
        self.assertEqual(AnthropicMessages().classify(result).media, (self.IMAGE,))
        output = {"type": "function_call_output", "call_id": "c1",
                  "output": [{"type": "input_image", "image_url": f"data:image/png;base64,{self.IMAGE.data}"}]}
        self.assertEqual(OpenAIResponses().classify(output).media, (self.IMAGE,))
        # Dropping a screenshot keeps the result and its text.
        kept = AnthropicMessages().edit(result, "screen", ())
        self.assertEqual(AnthropicMessages().classify(kept).media, ())
        self.assertEqual(kept["content"][0]["tool_use_id"], "t1")

    def test_a_repeated_gemini_result_reads_once(self) -> None:
        _, result = self.CALLS["gemini"]
        repeated = {**result, "parts": result["parts"] * 2}  # Gemini CLI repeats each result in later requests
        self.assertEqual(Gemini().classify(repeated), Gemini().classify(result))

    def test_a_repeated_gemini_result_stays_repeated_when_it_changes(self) -> None:
        call, result = self.CALLS["gemini"]
        repeated = {**result, "parts": result["parts"] * 2}  # Gemini CLI's resumed history
        edited = Gemini().edit(repeated, "short", ())
        self.assertEqual(edited["parts"][0], edited["parts"][1])
        self.assertEqual(Gemini().orphans([Gemini().user_message("go"), call, edited]), set())

    def test_an_assistant_turn_keeps_its_thinking_when_its_text_changes(self) -> None:
        reply = {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                                  {"type": "text", "text": "a long answer"}]}
        self.assertEqual(AnthropicMessages().edit(reply, "short", ())["content"],
                         [reply["content"][0], {"type": "text", "text": "short"}])

    def test_notes_join_a_trailing_user_turn_where_roles_must_alternate(self) -> None:
        result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
        self.assertEqual(AnthropicMessages().note([result], "size")[-1]["content"][-1], {"type": "text", "text": "size"})
        response = {"role": "user", "parts": [{"functionResponse": {"name": "sh", "response": {}}}]}
        self.assertEqual(Gemini().note([response], "size"), [{**response, "parts": [*response["parts"], {"text": "size"}]}])
        self.assertEqual(OpenAIResponses().note([], "size"), [OpenAIResponses().user_message("size")])
        self.assertEqual(OpenAIChat().note([{"role": "tool", "content": "ok"}], "size")[-1], {"role": "user", "content": "size"})

    def test_instructions_extend_each_protocols_system_prompt(self) -> None:
        self.assertEqual(OpenAIResponses().with_instructions({"instructions": "base"}, "more")["instructions"], "base\n\nmore")
        blocks = [{"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}]
        self.assertEqual(AnthropicMessages().with_instructions({"system": blocks}, "more")["system"],
                         [*blocks, {"type": "text", "text": "more"}])
        self.assertEqual(AnthropicMessages().with_instructions({}, "more")["system"], "more")
        chat = OpenAIChat().with_instructions({"messages": [{"role": "system", "content": "base"}]}, "more")
        self.assertEqual(chat["messages"], [{"role": "system", "content": "base\n\nmore"}])
        self.assertEqual(OpenAIChat().with_instructions({"messages": []}, "more")["messages"], [{"role": "system", "content": "more"}])
        gemini = Gemini().with_instructions({"system_instruction": {"parts": [{"text": "base"}]}}, "more")
        self.assertEqual(gemini, {"system_instruction": {"parts": [{"text": "base"}, {"text": "more"}]}})
