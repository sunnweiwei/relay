"""Experimental Anthropic Messages ingress for Relay's Compact/cache engine.

Only text and paired tool_use/tool_result content are supported. This module
deliberately fails closed on unknown content instead of silently losing data.
The task upstream must speak Anthropic Messages; a Gemini task-model bridge is
separate work and is NOT implemented here.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from relay import Compact, PrefixCheckpointCache
from relay.middleware import ContextEngine


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(part, dict) and part.get("type") == "text"
        and isinstance(part.get("text"), str) for part in value
    ):
        return "\n".join(part["text"] for part in value)
    raise ValueError("unsupported Anthropic text/tool-result content")


def to_relay_request(body: dict[str, Any], *, manager_model: str) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Anthropic messages must be a list")
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError("unsupported Anthropic message role")
        content = message.get("content")
        blocks = ([{"type": "text", "text": content}]
                  if isinstance(content, str) else content)
        if not isinstance(blocks, list):
            raise ValueError("unsupported Anthropic message content")
        for block in blocks:
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                items.append({"type": "message", "role": role, "content": block["text"]})
            elif kind == "tool_use" and role == "assistant":
                items.append({"type": "function_call", "call_id": block["id"],
                              "name": block["name"],
                              "arguments": json.dumps(block.get("input", {}), sort_keys=True)})
            elif kind == "tool_result" and role == "user":
                items.append({"type": "function_call_output",
                              "call_id": block["tool_use_id"],
                              "output": _text(block.get("content", ""))})
            else:
                raise ValueError(f"unsupported Anthropic block: {kind}")
    system = body.get("system", "")
    instructions = _text(system) if system else ""
    if "tools" in body:
        instructions += "\nAnthropic tools: " + json.dumps(body["tools"], sort_keys=True)
    return {"model": manager_model, "input": items, "instructions": instructions}


def from_relay_input(original: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError("unsupported generated Anthropic message role")
            block = {"type": "text", "text": _text(item.get("content", ""))}
        elif kind == "function_call":
            role = "assistant"
            block = {"type": "tool_use", "id": item["call_id"], "name": item["name"],
                     "input": json.loads(item["arguments"])}
        elif kind == "function_call_output":
            role = "user"
            block = {"type": "tool_result", "tool_use_id": item["call_id"],
                     "content": _text(item["output"])}
        else:
            raise ValueError(f"unsupported generated Anthropic item: {kind}")
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})
    forwarded = deepcopy(original)
    forwarded["messages"] = messages
    return forwarded


def create_anthropic_adapter(
    *, task_base_url: str, management_responses: Any,
    manager_model: str, cache: PrefixCheckpointCache | None = None,
    compact_threshold: int = 100, task_model: str | None = None,
    task_auth_token: str | None = None,
) -> Starlette:
    engine = ContextEngine(Compact(compact_threshold=compact_threshold),
                           checkpoint_mode="cache", checkpoint_cache=cache)
    task_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None))

    async def messages(request: Request) -> Response:
        try:
            body = await request.json()
            key = request.headers.get("x-api-key")
            if not key:
                raise ValueError("cache mode requires x-api-key")
            normalized = to_relay_request(body, manager_model=manager_model)
            prepared = await run_in_threadpool(
                engine.prepare, management_responses, normalized,
                json.dumps({"protocol": "anthropic", "key": key,
                            "task_model": task_model or body.get("model")}, sort_keys=True),
            )
            if prepared.overrides:
                raise ValueError("strategy request overrides are unsupported on Anthropic")
            forwarded = from_relay_input(body, prepared.input)
            if task_model is not None:
                forwarded["model"] = task_model
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return JSONResponse({"type": "error", "error": {
                "type": "invalid_request_error", "message": str(exc),
            }}, status_code=400)
        task_headers = {"anthropic-version":
                        request.headers.get("anthropic-version", "2023-06-01")}
        if task_auth_token is None:
            task_headers["x-api-key"] = key
        else:
            task_headers["authorization"] = f"Bearer {task_auth_token}"
        if "anthropic-beta" in request.headers:
            task_headers["anthropic-beta"] = request.headers["anthropic-beta"]
        upstream = await task_client.send(
            task_client.build_request(
                "POST", task_base_url.rstrip("/") + "/v1/messages",
                headers=task_headers,
                json=forwarded,
            ), stream=True,
        )
        content_type = upstream.headers.get("content-type", "application/json")
        if body.get("stream"):
            return StreamingResponse(
                upstream.aiter_bytes(), status_code=upstream.status_code,
                media_type=content_type,
                background=BackgroundTask(upstream.aclose),
            )
        try:
            return Response(await upstream.aread(), status_code=upstream.status_code,
                            media_type=content_type)
        finally:
            await upstream.aclose()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        try:
            yield
        finally:
            await task_client.aclose()

    app = Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])],
                    lifespan=lifespan)
    app.state.engine = engine
    return app
