"""Google Gemini API (`.../models/<model>:generateContent` and `:streamGenerateContent`).

Items are `contents` entries. A model turn with `functionCall` parts and the user turn
carrying the `functionResponse` parts form one group. Streaming is chosen by the URL,
so a summary request sent to the same URL may come back as a stream. Like the API, the
codec accepts snake_case field names (`function_call`, as litellm sends) as well.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.ir import Item, Kind
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, error_message


class Gemini:
    name = "gemini"
    paths = (":generateContent", ":streamGenerateContent")

    def managed(self, body: Body) -> bool:
        return isinstance(body.get("contents"), list)

    def items(self, body: Body) -> list[WireItem]:
        return list(body.get("contents") or [])

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return {**body, "contents": items}

    def classify(self, item: WireItem) -> Item:
        parts = [part for part in item.get("parts") or [] if isinstance(part, dict)]
        text = "\n".join(filter(None, (_part_text(part) for part in parts)))
        if item.get("role") == "model":
            if any(_has(part, "functionCall") for part in parts):
                return Item(Kind.TOOL_CALL, text)
            if parts and all(part.get("thought") for part in parts):
                return Item(Kind.REASONING, text)
            return Item(Kind.ASSISTANT, text)
        if item.get("role") == "function" or any(_has(part, "functionResponse") for part in parts):
            return Item(Kind.TOOL_RESULT, text)
        return Item(Kind.USER, text, media=any(_has(p, "inlineData") or _has(p, "fileData") for p in parts))

    def canonical(self, item: WireItem) -> bytes:
        """Gemini CLI rewrites a resumed history: each tool result is repeated and thought
        signatures become a placeholder. Neither changes the conversation."""

        parts: list[Any] = []
        for part in item.get("parts") or []:
            if isinstance(part, dict):
                part = {key: value for key, value in part.items() if key not in {"thoughtSignature", "thought_signature"}}
            if not parts or part != parts[-1]:
                parts.append(part)
        return canonical_json({**item, "parts": parts})

    def boundaries(self, items: list[WireItem]) -> frozenset[int]:
        legal = {0}
        for index, item in enumerate(items):
            if self.classify(item).kind is not Kind.TOOL_CALL:
                legal.add(index + 1)
        return frozenset(legal)

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        return head

    def user_message(self, text: str) -> WireItem:
        return {"role": "user", "parts": [{"text": text}]}

    def preamble(self, body: Body) -> str:
        system = body.get("systemInstruction") or body.get("system_instruction") or {}
        parts = system.get("parts") or [] if isinstance(system, dict) else []
        return "".join(_part_text(p) for p in parts if isinstance(p, dict)) + json.dumps(body.get("tools") or [])

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        request = {key: value for key, value in body.items() if key not in {"toolConfig", "tool_config"}}
        request["contents"] = [*items, self.user_message(prompt)]
        if request.get("tools"):
            request["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        for name in ("generationConfig", "generation_config"):
            if isinstance(config := request.get(name), dict):  # no structured-output schema for the summary
                drop = {f(key) for key in ("responseMimeType", "responseSchema", "responseJsonSchema") for f in (str, _snake)}
                request[name] = {k: v for k, v in config.items() if k not in drop}
        return request

    def stream_result(self, events: list[Body]) -> Body:
        parts = [part for event in events for part in _parts(event)]
        usage = next((e["usageMetadata"] for e in reversed(events) if "usageMetadata" in e), None)
        reason = next((c["finishReason"] for e in reversed(events) for c in e.get("candidates") or []
                       if c.get("finishReason")), None)
        candidate = {"content": {"role": "model", "parts": parts}, "finishReason": reason}
        return {"candidates": [candidate], "usageMetadata": usage}

    def output_text(self, payload: Body) -> str:
        texts = (p.get("text") for p in _parts(payload) if not p.get("thought"))
        return "".join(t for t in texts if isinstance(t, str)).strip()

    def finished(self, payload: Body) -> bool:
        return (payload.get("candidates") or [{}])[0].get("finishReason") == "STOP"

    def usage(self, payload: Body) -> int | None:
        usage = payload.get("usageMetadata")
        tokens = usage.get("promptTokenCount") if isinstance(usage, dict) else None
        return tokens if isinstance(tokens, int) and tokens > 0 else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        _, message = error_message(payload)
        return status == 400 and any(p in message for p in OVERFLOW_PHRASES)


def _parts(payload: Body) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") or [{}]
    content = candidates[0].get("content") or {}
    return [part for part in content.get("parts") or [] if isinstance(part, dict)]


def _part_text(part: dict[str, Any]) -> str:
    if isinstance(part.get("text"), str):
        return part["text"]
    for key in ("functionCall", "functionResponse", "executableCode", "codeExecutionResult"):
        if _has(part, key):
            return json.dumps(part.get(key, part.get(_snake(key))), ensure_ascii=False)
    return ""


def _has(part: dict[str, Any], name: str) -> bool:
    return name in part or _snake(name) in part


def _snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lower()
