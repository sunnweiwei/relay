"""Gemini generateContent facade backed by an OpenAI Responses model."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from relay.gemini import GeminiInput
from experiments.compact_cache.budget import TaskBudget


def _output_text(response: Any) -> str:
    value = getattr(response, "output_text", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "\n".join(
        getattr(part, "text", "") for item in getattr(response, "output", ())
        for part in getattr(item, "content", ()) or ()
        if isinstance(getattr(part, "text", None), str)
    ).strip()


def _tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    def schema(value: Any) -> Any:
        if isinstance(value, list):
            return [schema(item) for item in value]
        if isinstance(value, dict):
            return {key: (item.lower() if key == "type" and isinstance(item, str)
                          else schema(item)) for key, item in value.items()}
        return value

    result = []
    for group in body.get("tools", []):
        for declaration in group.get("functionDeclarations", []):
            result.append({"type": "function", "name": declaration["name"],
                           "description": declaration.get("description", ""),
                           "parameters": schema(declaration.get("parametersJsonSchema")
                           or declaration.get("parameters")
                           or {"type": "object", "properties": {}})})
    return result


def _parts(response: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) == "function_call":
            parts.append({"functionCall": {
                "name": item.name, "args": json.loads(item.arguments),
                "id": item.call_id,
            }})
        elif getattr(item, "type", None) == "message":
            for content in getattr(item, "content", ()):
                if getattr(content, "type", None) == "output_text" and content.text:
                    parts.append({"text": content.text})
    return parts


def create_bridge(client: OpenAI, model: str,
                  seen: list[dict[str, Any]], budget: TaskBudget | None = None) -> Starlette:
    async def generate(request: Request):
        if len(seen) >= 8:
            return JSONResponse({"error": {"message": "live request budget exceeded"}},
                                status_code=429)
        try:
            body = await request.json()
            normalized = GeminiInput(body, model).request
            seen.append(body)
            task = {"model": model, "input": normalized["input"],
                    "instructions": normalized.get("instructions", ""),
                    "max_output_tokens": 512, "reasoning": {"effort": "none"},
                    "store": False, "stream": False}
            tools = _tools(body)
            if tools:
                task["tools"] = tools
            if budget is not None:
                task = await run_in_threadpool(budget.bounded, task)
            response = await run_in_threadpool(client.responses.create, **task)
            parts = _parts(response)
            if not parts:
                content = _output_text(response)
                if content:
                    parts = [{"text": content}]
            if not parts:
                raise ValueError("task model returned no text or tool call")
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            payload = {"candidates": [{"content": {"role": "model",
                        "parts": parts}, "finishReason": "STOP",
                        "index": 0}], "usageMetadata": {
                            "promptTokenCount": input_tokens,
                            "candidatesTokenCount": output_tokens,
                            "totalTokenCount": input_tokens + output_tokens}}
            if request.url.path.endswith(":streamGenerateContent"):
                return Response("data: " + json.dumps(payload) + "\n\n",
                                media_type="text/event-stream")
            return JSONResponse(payload)
        except (TypeError, ValueError, KeyError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

    return Starlette(routes=[Route("/{path:path}", generate, methods=["POST"])])
