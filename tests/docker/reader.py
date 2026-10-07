"""A deterministic model for comparing a harness's own compaction with Relay's.

It speaks the OpenAI Responses and Chat Completions APIs, Anthropic Messages and the Gemini API
(chosen by the request path), records every request to a JSONL file, and plays one agent: the
latest message holding `TASK:` names files, and it reads them one per step with the harness's
shell tool, then answers. Asked for a summary (Relay's prompt or a harness's own), it writes
the task and its progress, so a compacted conversation goes on where it was in both runs and the
summary text is the same in both.

It reports prompt tokens as the request's size over four, times READER_SCALE (default 1), and
rejects a request above READER_LIMIT of them (but a summary request) with the protocol's
context-overflow error. With
READER_TOOL it reads with that tool (a `file_path` in /project) where the harness offers it.

python3 -m tests.docker.reader PORT LOG
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

TASK = re.compile(r"TASK: ((?:file_\d+\.txt ?)+)")
SUMMARY = "READER SUMMARY"
PROGRESS = re.compile(SUMMARY + r": task=((?:file_\d+\.txt ?)*); done=(\d+)")
# How Relay (Codex's prompt) and the harnesses ask for a summary.
ASKS = re.compile(r"CONTEXT CHECKPOINT COMPACTION|Your task is to create a detailed summary of|structured summary|"
                  r"<conversation>|summary of our conversation|run out of context|compact replacement checkpoint|"
                  r"state_snapshot|compaction engine|summarization agent|summarize the conversation|\[User\]:|"
                  r"summarizing a conversation|context summarization assistant", re.I)
SHELL = re.compile(r"(^|[_.])(bash|shell|exec_command|exec|terminal|run_shell_command|run_command|execute)$", re.I)
COMMAND_KEYS = ("command", "cmd", "script", "code")


class Turn:
    """One message, protocol-neutral."""

    def __init__(self, role: str, text: str, results: int = 0) -> None:
        self.role, self.text, self.results = role, text, results


def progress(turns: list[Turn]) -> tuple[list[str], int]:
    """The files of the current task and how many tool results it has had."""

    done = 0
    for turn in reversed(turns):
        if turn.role == "user" and (match := TASK.search(turn.text)) and SUMMARY not in turn.text:
            return match[1].split(), done
        if match := PROGRESS.search(turn.text):
            return match[1].split(), done + int(match[2])
        done += turn.results
    return [], done


def summarizing(system: str, turns: list[Turn]) -> bool:
    last = next((turn for turn in reversed(turns) if turn.role != "system"), None)
    return bool(last and ASKS.search(last.text)) or bool(ASKS.search(system) and not TASK.search(system))


def decide(system: str, turns: list[Turn], tools: list[tuple[str, dict]]) -> tuple[str, Any]:
    """("text", answer) or ("call", (name, arguments))."""

    if summarizing(system, turns):
        files, done = progress(turns)
        return "text", f"{SUMMARY}: task={' '.join(files)}; done={done}."
    files, done = progress(turns)
    shell = next(((name, schema) for name, schema in tools if SHELL.search(name)), None)
    if not tools or shell is None or done >= len(files):
        return "text", f"READER DONE: {done} of {len(files)} files read." if files else "READER OK"
    name, schema = shell
    reader = os.getenv("READER_TOOL")
    if reader and any(name == reader for name, _ in tools):
        return "call", (reader, {"file_path": f"/project/{files[done]}"})
    return "call", (name, arguments(schema, f"cat {files[done]}"))


def arguments(schema: dict, command: str) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    key = next((k for k in COMMAND_KEYS if k in properties), next(iter(properties), "command"))
    value: Any = ["bash", "-lc", command] if (properties.get(key) or {}).get("type") == "array" else command
    args = {key: value}
    for other in schema.get("required") or []:
        if other not in args and (properties.get(other) or {}).get("type") == "string":
            args[other] = "read a file"
    return args


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_text(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
                       for part in content)
    return ""


class Reader:
    def __init__(self, log: str) -> None:
        self.log, self.count = log, 0
        self.app = Starlette(routes=[Route("/{path:path}", self.handle, methods=["GET", "POST", "HEAD"])])

    async def handle(self, request: Request) -> Response:
        path = request.url.path
        body = await request.json() if request.method == "POST" else None
        self.count, self.answer = self.count + 1, None
        response = self.answer_to(path, body)
        headers = {k: v for k, v in request.headers.items() if k.startswith(("x-", "user-agent", "anthropic"))}
        with open(self.log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"n": self.count, "path": path, "answer": self.answer, "headers": headers,
                                     "body": body}, ensure_ascii=False) + "\n")
        return response

    def decide(self, system: str, turns: list[Turn], tools: list[tuple[str, dict]]) -> tuple[str, Any]:
        self.answer = decide(system, turns, tools)
        return self.answer

    def answer_to(self, path: str, body: Any) -> Response:
        tokens = len(json.dumps(body)) // 4 * int(os.getenv("READER_SCALE", "1"))
        if body is None:
            return JSONResponse({"data": [], "models": []})
        if path.endswith(":countTokens"):
            return JSONResponse({"totalTokens": tokens})
        if path.endswith("/count_tokens"):
            return JSONResponse({"input_tokens": tokens})
        if path.endswith("/responses"):
            response = self.responses(body, tokens)
        elif path.endswith("/chat/completions"):
            response = self.chat(body, tokens)
        elif path.endswith("/messages"):
            response = self.messages(body, tokens)
        elif ":generateContent" in path or ":streamGenerateContent" in path:
            response = self.gemini(body, tokens, ":streamGenerateContent" in path)
        else:
            return JSONResponse({"error": {"message": f"unknown path {path}"}}, 404)
        # A summary request always fits: a harness compacts before its model's own limit.
        summary = self.answer and self.answer[0] == "text" and str(self.answer[1]).startswith(SUMMARY)
        if tokens > int(os.getenv("READER_LIMIT") or 10**12) and not summary:
            self.answer = ("overflow", tokens)
            return _overflow(path, tokens)
        return response

    # OpenAI Responses
    def responses(self, body: dict, tokens: int) -> Response:
        turns = []
        for item in body.get("input") or []:
            kind = item.get("type", "message")
            if kind == "message":
                turns.append(Turn(item.get("role", ""), _text(item.get("content"))))
            elif kind.endswith("_output"):
                turns.append(Turn("tool", _text(item.get("output")), 1))
        tools = [(t.get("name", ""), t.get("parameters") or {}) for t in body.get("tools") or [] if t.get("name")]
        action, value = self.decide(_text(body.get("instructions")), turns, tools)
        n = self.count
        if action == "call":
            output = [{"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "status": "completed",
                       "name": value[0], "arguments": json.dumps(value[1])}]
        else:
            output = [{"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": value, "annotations": []}]}]
        response = {"id": f"resp_{n}", "object": "response", "created_at": 0, "status": "completed", "model": body.get("model"),
                    "output": output, "usage": {"input_tokens": tokens, "output_tokens": 10,
                                                "total_tokens": tokens + 10,
                                                "input_tokens_details": {"cached_tokens": 0},
                                                "output_tokens_details": {"reasoning_tokens": 0}}}
        if not body.get("stream"):
            return JSONResponse(response)
        events = [("response.created", {"type": "response.created",
                                        "response": {**response, "output": [], "status": "in_progress"}})]
        for index, item in enumerate(output):
            events.append(("response.output_item.added", {"type": "response.output_item.added", "output_index": index,
                                                          "item": {**item, "status": "in_progress",
                                                                   **({"arguments": ""} if "arguments" in item else {"content": []})}}))
            if item["type"] == "message":
                part = item["content"][0]
                events += [("response.content_part.added", {"type": "response.content_part.added", "item_id": item["id"],
                                                            "output_index": index, "content_index": 0,
                                                            "part": {**part, "text": ""}}),
                           ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item["id"],
                                                           "output_index": index, "content_index": 0, "delta": part["text"]}),
                           ("response.output_text.done", {"type": "response.output_text.done", "item_id": item["id"],
                                                          "output_index": index, "content_index": 0, "text": part["text"]}),
                           ("response.content_part.done", {"type": "response.content_part.done", "item_id": item["id"],
                                                           "output_index": index, "content_index": 0, "part": part})]
            else:
                events += [("response.function_call_arguments.delta",
                            {"type": "response.function_call_arguments.delta", "item_id": item["id"],
                             "output_index": index, "delta": item["arguments"]}),
                           ("response.function_call_arguments.done",
                            {"type": "response.function_call_arguments.done", "item_id": item["id"],
                             "output_index": index, "arguments": item["arguments"]})]
            events.append(("response.output_item.done", {"type": "response.output_item.done", "output_index": index,
                                                         "item": item}))
        events.append(("response.completed", {"type": "response.completed", "response": response}))
        return _sse(events)

    # OpenAI Chat Completions
    def chat(self, body: dict, tokens: int) -> Response:
        system, turns = "", []
        for message in body.get("messages") or []:
            role = message.get("role", "")
            if role in ("system", "developer"):
                system += _text(message.get("content"))
            turns.append(Turn("tool" if role == "tool" else role, _text(message.get("content")), int(role == "tool")))
        tools = [(t["function"].get("name", ""), t["function"].get("parameters") or {})
                 for t in body.get("tools") or [] if isinstance(t.get("function"), dict)]
        action, value = self.decide(system, turns, tools)
        n = self.count
        if action == "call":
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{n}", "type": "function", "function": {"name": value[0], "arguments": json.dumps(value[1])}}]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": value}, "stop"
        usage = {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10}
        if not body.get("stream"):
            return JSONResponse({"id": f"chatcmpl-{n}", "object": "chat.completion", "created": 0, "model": body.get("model"),
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
        delta = {"role": "assistant", **({"tool_calls": [{"index": 0, **message["tool_calls"][0]}]}
                                         if action == "call" else {"content": value})}
        chunk = lambda choices, **extra: {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk", "created": 0,
                                          "model": body.get("model"), "choices": choices, **extra}
        lines = [chunk([{"index": 0, "delta": delta, "finish_reason": None}]),
                 chunk([{"index": 0, "delta": {}, "finish_reason": finish}]), chunk([], usage=usage)]
        payload = b"".join(f"data: {json.dumps(line)}\n\n".encode() for line in lines) + b"data: [DONE]\n\n"
        return StreamingResponse(iter([payload]), media_type="text/event-stream")

    # Anthropic Messages
    def messages(self, body: dict, tokens: int) -> Response:
        turns = []
        for message in body.get("messages") or []:
            content = message.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
            results = sum(b.get("type") == "tool_result" for b in blocks)
            text = "".join(b.get("text", "") if b.get("type") == "text" else _text(b.get("content"))
                           for b in blocks if b.get("type") in ("text", "tool_result"))
            turns.append(Turn("tool" if results and message.get("role") == "user" and not TASK.search(text)
                              else message.get("role", ""), text, results))
        tools = [(t.get("name", ""), t.get("input_schema") or {}) for t in body.get("tools") or [] if t.get("name")]
        action, value = self.decide(_text(body.get("system")), turns, tools)
        n = self.count
        content = ([{"type": "tool_use", "id": f"toolu_{n:024d}", "name": value[0], "input": value[1]}]
                   if action == "call" else [{"type": "text", "text": value}])
        stop = "tool_use" if action == "call" else "end_turn"
        usage = {"input_tokens": tokens, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": f"msg_{n:024d}", "type": "message", "role": "assistant", "model": body.get("model"),
                   "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            return JSONResponse(message)
        events = [("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}})]
        for index, block in enumerate(content):
            if block["type"] == "text":
                start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
            else:
                start, delta = {**block, "input": {}}, {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            events += [("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}),
                       ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}),
                       ("content_block_stop", {"type": "content_block_stop", "index": index})]
        events += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                      "usage": {"output_tokens": 10}}),
                   ("message_stop", {"type": "message_stop"})]
        return _sse(events)

    # Gemini
    def gemini(self, body: dict, tokens: int, stream: bool) -> Response:
        turns = []
        for content in body.get("contents") or []:
            parts = content.get("parts") or []
            results = sum("functionResponse" in p for p in parts)
            text = "".join(p.get("text", "") for p in parts) + "".join(
                json.dumps(p["functionResponse"].get("response")) for p in parts if "functionResponse" in p)
            role = content.get("role", "user")
            turns.append(Turn("tool" if results else "assistant" if role == "model" else role, text, results))
        system = _text((body.get("systemInstruction") or body.get("system_instruction") or {}).get("parts"))
        tools = [(d.get("name", ""), d.get("parameters") or d.get("parametersJsonSchema") or {})
                 for t in body.get("tools") or [] for d in t.get("functionDeclarations") or []]
        action, value = self.decide(system, turns, tools)
        part = {"functionCall": {"name": value[0], "args": value[1]}} if action == "call" else {"text": value}
        response = {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
                    "usageMetadata": {"promptTokenCount": tokens, "candidatesTokenCount": 10, "totalTokenCount": tokens + 10},
                    "modelVersion": "reader"}
        if not stream:
            return JSONResponse(response)
        return StreamingResponse(iter([f"data: {json.dumps(response)}\n\n".encode()]), media_type="text/event-stream")


def _overflow(path: str, tokens: int) -> Response:
    limit = os.getenv("READER_LIMIT")
    if path.endswith("/messages"):
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                        "message": f"prompt is too long: {tokens} tokens > {limit} maximum"}}, 400)
    if ":generateContent" in path or ":streamGenerateContent" in path:
        return JSONResponse({"error": {"code": 400, "status": "INVALID_ARGUMENT", "message":
                                       f"The input token count ({tokens}) exceeds the maximum number of tokens allowed ({limit})."}}, 400)
    return JSONResponse({"error": {"code": "context_length_exceeded", "type": "invalid_request_error", "param": "input",
                                   "message": "Your input exceeds the context window of this model."}}, 400)


def _sse(events: list[tuple[str, dict]]) -> StreamingResponse:
    events = [(name, {**data, "sequence_number": n}) if name.startswith("response.") else (name, data)
              for n, (name, data) in enumerate(events)]
    payload = b"".join(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events)
    return StreamingResponse(iter([payload]), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(Reader(sys.argv[2]).app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
