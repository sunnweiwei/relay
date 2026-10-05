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

from ..core.ir import Item, Kind, Media
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, decoded, error_message


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
            opaque = sum(decoded(part.get("thoughtSignature") or part.get("thought_signature")) for part in parts)
            if any(_has(part, "functionCall") for part in parts):
                return Item(Kind.TOOL_CALL, text, opaque=opaque)
            if parts and all(part.get("thought") for part in parts):
                return Item(Kind.REASONING, text, opaque=opaque)
            return Item(Kind.ASSISTANT, text, opaque=opaque)
        media = tuple(m for part in parts if (m := _media(part)))
        if item.get("role") == "function" or any(_has(part, "functionResponse") for part in parts):
            return Item(Kind.TOOL_RESULT, text, media=media)
        return Item(Kind.USER, text, media=media)

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

    def write(self, item: Item) -> WireItem:
        if item.kind is Kind.ASSISTANT and not item.media:
            return {"role": "model", "parts": [{"text": item.text}]}
        if item.kind in (Kind.USER, Kind.SUMMARY, Kind.CONTEXT):
            return {"role": "user", "parts": _content(item.text, item.media)}
        raise ValueError(f"Relay cannot write a {item.kind.value} item")

    def edit(self, wire: WireItem, text: str, media: tuple[Media, ...]) -> WireItem:
        """Thoughts stay as they were; tool results share the new content: the first carries it."""

        parts = [part for part in wire.get("parts") or [] if isinstance(part, dict)]
        if wire.get("role") == "model":
            if media or any(_has(part, "functionCall") for part in parts):
                raise ValueError("the content of a tool call cannot change on its own")
            return {**wire, "parts": [*(part for part in parts if part.get("thought")), {"text": text}]}
        responses = [part for part in parts if _has(part, "functionResponse")]
        if not responses:
            return {**wire, "parts": _content(text, media)}
        edited = []
        for n, part in enumerate(responses):
            key = "functionResponse" if "functionResponse" in part else "function_response"
            edited.append({**part, key: {**part[key], "response": {"output": text if n == 0 else ""}}})
        return {**wire, "parts": [*edited, *(_content("", media) if media else [])]}

    def legal(self, items: list[WireItem]) -> str | None:
        """A model turn's function calls are answered by the turn right after it, and only them."""

        asked = 0
        for index, item in enumerate(items):
            parts = [part for part in item.get("parts") or [] if isinstance(part, dict)]
            responses = [part for part in parts if _has(part, "functionResponse")]
            answered = len({canonical_json(part) for part in responses})  # Gemini CLI repeats them on resume
            if answered != asked:
                return f"content {index} answers {answered} function calls where {asked} were made"
            asked = sum(_has(part, "functionCall") for part in parts) if item.get("role") == "model" else 0
        return f"{asked} function calls are not answered" if asked else None

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        if not items or items[-1].get("role") != "user":
            return [*items, self.user_message(text)]
        return [*items[:-1], {**items[-1], "parts": [*(items[-1].get("parts") or []), {"text": text}]}]

    def offers_tools(self, body: Body) -> bool:
        return bool(body.get("tools"))

    def with_instructions(self, body: Body, text: str) -> Body:
        key = "system_instruction" if "system_instruction" in body else "systemInstruction"
        system = body.get(key) if isinstance(body.get(key), dict) else {}
        return {**body, key: {**system, "parts": [*(system.get("parts") or []), {"text": text}]}}

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


def _media(part: dict[str, Any]) -> Media | None:
    inline = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline, dict):
        return Media(_kind(inline.get("mimeType") or inline.get("mime_type") or ""),
                     inline.get("mimeType") or inline.get("mime_type") or "", inline.get("data") or "")
    file = part.get("fileData") or part.get("file_data")
    if isinstance(file, dict):
        mime = file.get("mimeType") or file.get("mime_type") or ""
        return Media(_kind(mime), mime, url=file.get("fileUri") or file.get("file_uri") or "")
    return None


def _kind(mime: str) -> str:
    return "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else "file"


def _content(text: str, media: tuple[Media, ...]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": text}] if text or not media else []
    for m in media:
        parts.append({"inlineData": {"mimeType": m.mime, "data": m.data}} if m.data
                     else {"fileData": {"mimeType": m.mime, "fileUri": m.url}})
    return parts


def _has(part: dict[str, Any], name: str) -> bool:
    return name in part or _snake(name) in part


def _snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lower()
