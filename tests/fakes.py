"""Scripted fake upstreams for the OpenAI Responses and Anthropic Messages APIs.

The fake records every request and plays a tiny deterministic agent: it calls the
harness's shell tool until `tool_calls` tool results belong to the current user turn,
then answers with a final message. Answering a compaction prompt, it writes how many
tool results it has seen into the summary and later reads that count back, so a
compacted conversation keeps its progress like a real model would. Reported prompt
tokens are the request size divided by four; prompts above `max_prompt_tokens` are
rejected with the protocol's context-overflow error.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from relay.providers import Upstream

FINAL_TEXT = "FAKE FINAL ANSWER"
COMPACTION_MARKER = "CONTEXT CHECKPOINT COMPACTION"
_SUMMARY = re.compile(r"FAKE SUMMARY: (\d+) tool results")


class FakeUpstream:
    def __init__(
        self,
        *,
        tool_calls: int = 1,
        command: str = "echo relay",
        max_prompt_tokens: int | None = None,
        fail_summaries: bool = False,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.tool_calls = tool_calls
        self.command = command
        self.arguments = arguments or {}  # extra exec_command arguments, e.g. an escalation request
        self.max_prompt_tokens = max_prompt_tokens
        self.fail_summaries = fail_summaries
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.app = Starlette(
            routes=[
                Route("/v1/responses", self._responses, methods=["POST"]),
                Route("/v1/messages", self._messages, methods=["POST"]),
                Route("/{path:path}", self._other, methods=["GET", "POST", "HEAD"]),
            ]
        )

    def bodies(self, path: str) -> list[dict[str, Any]]:
        with self.lock:
            return [r["body"] for r in self.requests if r["path"] == path]

    def _record(self, request: Request, body: Any) -> None:
        with self.lock:
            self.requests.append({"path": request.url.path, "headers": dict(request.headers), "body": body})

    def _decide(self, turns: list[tuple[str, str, int]]) -> tuple[str, str | None]:
        """("summary", text) | ("tool", None) | ("final", text) for normalized turns."""

        summarizing = bool(turns) and COMPACTION_MARKER in turns[-1][1]
        if summarizing and self.fail_summaries:
            return "error", None
        steps = _steps(turns[:-1] if summarizing else turns)
        if summarizing:
            return "summary", f"FAKE SUMMARY: {steps} tool results so far."
        return ("tool", None) if steps < self.tool_calls else ("final", FINAL_TEXT)

    async def _other(self, request: Request) -> Response:
        self._record(request, None)
        return JSONResponse({"error": {"message": "not found"}}, status_code=404)

    async def _responses(self, request: Request) -> Response:
        body = await request.json()
        self._record(request, body)
        items = body.get("input") or []
        if self.max_prompt_tokens and _prompt_tokens(body) > self.max_prompt_tokens:
            return JSONResponse({"error": {"code": "context_length_exceeded",
                                           "message": "Your input exceeds the context window."}}, 400)
        action, text = self._decide([_responses_turn(item) for item in items])
        if action == "error":
            return JSONResponse({"error": {"message": "summary failed"}}, 400)
        if action == "tool":
            n = len(items)
            output = [
                {"type": "reasoning", "id": f"rs_{n}", "summary": [], "encrypted_content": "ZmFrZQ=="},
                {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "status": "completed",
                 **_shell_call(body.get("tools") or [], self.command, self.arguments)},
            ]
        else:
            output = [{"type": "message", "id": "msg_fake", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]}]
        tokens = _prompt_tokens(body)
        response = {"id": f"resp_{len(self.requests)}", "object": "response", "status": "completed",
                    "model": body.get("model"), "output": output,
                    "usage": {"input_tokens": tokens, "output_tokens": 10, "total_tokens": tokens + 10}}
        if not body.get("stream"):
            return JSONResponse(response)
        events = [("response.created", {"type": "response.created",
                                        "response": {**response, "output": [], "status": "in_progress"}})]
        for index, item in enumerate(output):
            for kind in ("response.output_item.added", "response.output_item.done"):
                events.append((kind, {"type": kind, "output_index": index, "item": item}))
        events.append(("response.completed", {"type": "response.completed", "response": response}))
        return _sse(events)

    async def _messages(self, request: Request) -> Response:
        body = await request.json()
        self._record(request, body)
        messages = body.get("messages") or []
        if self.max_prompt_tokens and _prompt_tokens(body) > self.max_prompt_tokens:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "prompt is too long"}}, 400)
        action, text = self._decide([_anthropic_turn(message) for message in messages])
        if action == "error":
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "failed"}}, 400)
        if action == "tool" and body.get("tools"):
            content = [{"type": "tool_use", "id": f"toolu_{len(messages)}", "name": "Bash",
                        "input": {"command": self.command, "description": "Run"}}]
        else:
            content = [{"type": "text", "text": text or FINAL_TEXT}]
        stop = "tool_use" if content[0]["type"] == "tool_use" else "end_turn"
        usage = {"input_tokens": _prompt_tokens(body), "output_tokens": 10,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": f"msg_{len(self.requests)}", "type": "message", "role": "assistant",
                   "model": body.get("model"), "content": content, "stop_reason": stop,
                   "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            return JSONResponse(message)
        events = [("message_start", {"type": "message_start",
                                     "message": {**message, "content": [], "stop_reason": None}})]
        for index, block in enumerate(content):
            if block["type"] == "text":
                start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
            else:
                start = {**block, "input": {}}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            events += [
                ("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}),
                ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}),
                ("content_block_stop", {"type": "content_block_stop", "index": index}),
            ]
        events += [
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": 10}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        return _sse(events)


def _steps(turns: list[tuple[str, str, int]]) -> int:
    """Tool results in the current user turn, including those recorded in a summary."""

    steps = 0
    for role, text, results in reversed(turns):
        if match := _SUMMARY.search(text):
            return steps + int(match[1])
        if role == "user" and not text.lstrip().startswith("<"):  # a real user message
            return steps
        steps += results
    return steps


def _responses_turn(item: dict[str, Any]) -> tuple[str, str, int]:
    if item.get("type", "message") == "message":
        content = item.get("content")
        text = content if isinstance(content, str) else "".join(
            part.get("text", "") for part in content or [] if isinstance(part, dict))
        return item.get("role", ""), text, 0
    return "tool", "", int(str(item.get("type", "")).endswith("_output"))


def _anthropic_turn(message: dict[str, Any]) -> tuple[str, str, int]:
    content = message.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    results = sum(block.get("type") == "tool_result" for block in blocks)
    return ("tool" if results else message.get("role", "")), text, results


def _shell_call(tools: list[dict[str, Any]], command: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if "exec_command" in {tool.get("name") for tool in tools}:
        return {"name": "exec_command", "arguments": json.dumps({"cmd": command, **arguments})}
    return {"name": "shell", "arguments": json.dumps({"command": ["bash", "-lc", command]})}


def _prompt_tokens(body: dict[str, Any]) -> int:
    return len(json.dumps(body)) // 4


def _sse(events: list[tuple[str, dict[str, Any]]]) -> StreamingResponse:
    payload = b"".join(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events)
    return StreamingResponse(iter([payload]), media_type="text/event-stream")


def local_upstreams(url: str, api_key: str | None = None) -> dict[str, Upstream]:
    """Every Relay upstream pointed at the fake served at `url`."""

    return {
        "openai": Upstream(f"{url}/v1", "/v1", api_key),
        "chatgpt": Upstream(f"{url}/v1", "/backend-api/codex", api_key),
        "gemini": Upstream(url, "", api_key),
        "anthropic": Upstream(url, "", api_key, "x-api-key"),
    }


@contextmanager
def serve(app: Starlette) -> Iterator[str]:
    """Run an ASGI app on a free local port for the duration of the block."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
