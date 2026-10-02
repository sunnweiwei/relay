"""Anthropic Messages task route backed by OpenAI Responses."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from experiments.compact_cache.anthropic_adapter import to_relay_request
from experiments.compact_cache.budget import TaskBudget


def _text(response: Any) -> str:
    direct = getattr(response, "output_text", "")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return "\n".join(
        getattr(part, "text", "") for item in getattr(response, "output", ())
        for part in getattr(item, "content", ()) or ()
        if isinstance(getattr(part, "text", None), str)
    ).strip()


def _blocks(response: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) == "function_call":
            arguments = json.loads(item.arguments)
            if item.name == "Read" and arguments.get("pages") == "":
                # Claude Code rejects an empty optional pages value.
                arguments.pop("pages")
            blocks.append({"type": "tool_use", "id": item.call_id,
                           "name": item.name, "input": arguments})
        elif getattr(item, "type", None) == "message":
            for content in getattr(item, "content", ()):
                if getattr(content, "type", None) == "output_text" and content.text:
                    blocks.append({"type": "text", "text": content.text})
    return blocks


def _tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": item["name"],
             "description": item.get("description", ""),
             "parameters": item.get("input_schema") or {"type": "object", "properties": {}}}
            for item in body.get("tools", [])]


def _sse(model: str, blocks: list[dict[str, Any]], message_id: str,
         input_tokens: int, output_tokens: int) -> bytes:
    envelope = {"id": message_id, "type": "message", "role": "assistant",
                "content": [], "model": model, "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": input_tokens,
                                                  "output_tokens": 0}}
    events = [("message_start", {"type": "message_start", "message": envelope})]
    for index, block in enumerate(blocks):
        if block["type"] == "tool_use":
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events.extend([
            ("content_block_start", {"type": "content_block_start", "index": index,
                                     "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": index,
                                     "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ])
    events.extend([
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": ("tool_use" if any(
                               block["type"] == "tool_use" for block in blocks)
                               else "end_turn"), "stop_sequence": None},
                           "usage": {"output_tokens": output_tokens}}),
        ("message_stop", {"type": "message_stop"}),
    ])
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n"
                   for name, data in events).encode()


def create_bridge(client: OpenAI, model: str,
                  seen: list[dict[str, Any]] | None = None,
                  budget: TaskBudget | None = None) -> Starlette:
    async def messages(request: Request):
        try:
            body = await request.json()
            normalized = to_relay_request(body, manager_model=model)
            if seen is not None:
                seen.append(body)
            task = {"model": model, "input": normalized["input"],
                    "instructions": normalized["instructions"],
                    "max_output_tokens": 512, "reasoning": {"effort": "none"},
                    "stream": False, "store": False}
            tools = _tools(body)
            if tools:
                task["tools"] = tools
            if budget is not None:
                task = await run_in_threadpool(budget.bounded, task)
            response = await run_in_threadpool(client.responses.create, **task)
            blocks = _blocks(response)
            if not blocks:
                content = _text(response)
                if content:
                    blocks = [{"type": "text", "text": content}]
            if not blocks:
                raise ValueError("task model returned no text or tool call")
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            message_id = f"msg_{getattr(response, 'id', 'relay')}"
            if body.get("stream"):
                return Response(_sse(model, blocks, message_id,
                                     input_tokens, output_tokens),
                                media_type="text/event-stream")
            return JSONResponse({"id": message_id, "type": "message",
                                 "role": "assistant", "content": blocks,
                                 "model": model, "stop_reason": (
                                     "tool_use" if any(block["type"] == "tool_use"
                                                       for block in blocks) else "end_turn"),
                                 "usage": {"input_tokens": input_tokens,
                                           "output_tokens": output_tokens}})
        except (TypeError, ValueError, KeyError) as exc:
            return JSONResponse({"type": "error", "error": {
                "type": "invalid_request_error", "message": str(exc)}},
                status_code=400)

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])
