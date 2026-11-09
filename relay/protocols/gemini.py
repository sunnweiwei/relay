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
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, decoded, error_message, extra, extras


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
        parts = _distinct(item.get("parts") or [])
        text = "\n".join(filter(None, (_part_text(part) for part in parts)))
        if item.get("role") == "model":
            opaque = sum(decoded(part.get("thoughtSignature") or part.get("thought_signature")) for part in parts)
            if any(_has(part, "functionCall") for part in parts):
                return Item(Kind.TOOL_CALL, text, opaque=opaque)
            if parts and all(part.get("thought") for part in parts):
                return Item(Kind.REASONING, text, opaque=opaque)
            return Item(Kind.ASSISTANT, text, opaque=opaque)
        media = tuple(m for part in parts for m in _media(part))
        if item.get("role") == "function" or any(_has(part, "functionResponse") for part in parts):
            return Item(Kind.TOOL_RESULT, text, media=media)
        return Item(Kind.USER, text, media=media)

    def canonical(self, item: WireItem) -> bytes:
        """Gemini CLI rewrites a resumed history: each tool result is repeated and thought
        signatures become a placeholder. Neither changes the conversation."""

        unsigned = [{k: v for k, v in part.items() if k not in {"thoughtSignature", "thought_signature"}}
                    for part in item.get("parts") or [] if isinstance(part, dict)]
        return canonical_json({**item, "parts": _distinct(unsigned)})

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
        """Thoughts stay as they were; tool results share the new content: the first carries it
        (and its repeats), with the media, inside it where the function response had them there."""

        parts = [part for part in wire.get("parts") or [] if isinstance(part, dict)]
        if wire.get("role") == "model":
            if media or any(_has(part, "functionCall") for part in parts):
                raise ValueError("the content of a tool call cannot change on its own")
            return {**wire, "parts": [*(part for part in parts if part.get("thought")), {"text": text}]}
        responses = [part for part in parts if _has(part, "functionResponse")]
        if not responses:
            return {**wire, "parts": _content(text, media)}
        keys = ("functionResponse", "function_response")
        nested = any((part.get(key) or {}).get("parts") for part in responses for key in keys)
        edited = []
        for part in responses:  # a result Gemini CLI repeats (it does on resume) stays repeated
            key = keys[0] if keys[0] in part else keys[1]
            same = canonical_json(part) == canonical_json(responses[0])
            response = {k: v for k, v in part[key].items() if k != "parts"}
            edited.append({**part, key: {**response, "response": {"output": text if same else ""},
                                         **({"parts": _content("", media)} if nested and same and media else {})}})
        return {**wire, "parts": [*edited, *(_content("", media) if media and not nested else [])]}

    def orphans(self, items: list[WireItem]) -> set[int]:
        """Model turns whose function calls the turn right after them does not answer, and turns
        answering calls the turn before them does not make."""

        def count(item: WireItem, kind: str) -> int:  # (Gemini CLI repeats responses on resume)
            return len({canonical_json(part) for part in _distinct(item.get("parts") or []) if _has(part, kind)})

        lonely = set()
        for n, item in enumerate(items):
            asked = count(item, "functionCall") if item.get("role") == "model" else 0
            if asked and not (n + 1 < len(items) and count(items[n + 1], "functionResponse") == asked):
                lonely.add(n)
            answered = count(item, "functionResponse")
            before = count(items[n - 1], "functionCall") if n and items[n - 1].get("role") == "model" else 0
            if answered and answered != before:
                lonely.add(n)
        return lonely

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        if not items or items[-1].get("role") != "user":
            return [*items, self.user_message(text)]
        return [*items[:-1], {**items[-1], "parts": [*(items[-1].get("parts") or []), {"text": text}]}]

    def offers_tools(self, body: Body) -> bool:
        return bool(body.get("tools"))

    def with_instructions(self, body: Body, text: str) -> Body:
        """`text` joins the last text part: gateways translating to another API (LiteLLM to OpenAI)
        keep only the first part, which is all the harness itself sends."""

        key = "system_instruction" if "system_instruction" in body else "systemInstruction"
        system = body.get(key) if isinstance(body.get(key), dict) else {}
        parts = list(system.get("parts") or [])
        last = next((n for n in range(len(parts) - 1, -1, -1) if isinstance(parts[n].get("text"), str)), None)
        if last is None:
            parts.append({"text": text})
        else:
            parts[last] = {**parts[last], "text": f"{parts[last]['text']}\n\n{text}"}
        return {**body, key: {**system, "parts": parts}}

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

    def request(self, body: Body, system: str, items: list[WireItem]) -> Body:
        drop = {"tools", "systemInstruction", "system_instruction"}
        request = self.summary_request({k: v for k, v in body.items() if k not in drop}, items, "")
        request["contents"] = items
        if system:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        return request

    def stream_result(self, events: list[Body]) -> Body:
        parts = [part for event in events for part in _parts(event)]
        usage = next((e["usageMetadata"] for e in reversed(events) if "usageMetadata" in e), None)
        reason = next((c["finishReason"] for e in reversed(events) for c in e.get("candidates") or []
                       if c.get("finishReason")), None)
        candidate = {"content": {"role": "model", "parts": parts}, "finishReason": reason}
        return {"candidates": [candidate], "usageMetadata": usage}

    def answer_text(self, payload: Body) -> str:
        return self.output_text(payload)

    def output_text(self, payload: Body) -> str:
        texts = (p.get("text") for p in _parts(payload) if not p.get("thought"))
        return "".join(t for t in texts if isinstance(t, str)).strip()

    def finished(self, payload: Body) -> bool:
        return (payload.get("candidates") or [{}])[0].get("finishReason") == "STOP"

    def usage(self, payload: Body) -> int | None:
        usage = payload.get("usageMetadata")
        tokens = usage.get("promptTokenCount") if isinstance(usage, dict) else None
        return tokens if isinstance(tokens, int) and tokens > 0 else None

    def cached(self, payload: Body) -> int | None:
        usage = payload.get("usageMetadata")
        return usage.get("cachedContentTokenCount") if isinstance(usage, dict) else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        _, message = error_message(payload)
        return status == 400 and any(p in message for p in OVERFLOW_PHRASES)


def _distinct(parts: list[Any]) -> list[dict[str, Any]]:
    """The parts, a part repeated right after itself once (Gemini CLI repeats tool results)."""

    kept: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, dict) and (not kept or canonical_json(part) != canonical_json(kept[-1])):
            kept.append(part)
    return kept


def _parts(payload: Body) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") or [{}]
    content = candidates[0].get("content") or {}
    return [part for part in content.get("parts") or [] if isinstance(part, dict)]


def _part_text(part: dict[str, Any]) -> str:
    if isinstance(part.get("text"), str):
        return part["text"]
    for key in ("functionCall", "functionResponse", "executableCode", "codeExecutionResult"):
        if _has(part, key):
            value = part.get(key, part.get(_snake(key)))
            if key == "functionResponse":
                if _output(value) is not None:
                    return _output(value)  # (the output itself, where `edit` writes it)
                value = {k: v for k, v in value.items() if k != "parts"}  # (its media are media)
            return json.dumps(value, ensure_ascii=False)
    return ""


def _output(response: Any) -> str | None:
    """A function's response that is its output alone (`{"output": "..."}`, Gemini CLI's), else None."""

    inner = response.get("response") if isinstance(response, dict) else None
    return inner["output"] if isinstance(inner, dict) and list(inner) == ["output"] and isinstance(inner["output"], str) else None


def _media(part: dict[str, Any]) -> list[Media]:
    """A part's media, those inside a function response included."""

    response = part.get("functionResponse") or part.get("function_response")
    if isinstance(response, dict):
        return [m for inner in response.get("parts") or [] if isinstance(inner, dict) for m in _media(inner)]
    for camel, snake in (("inlineData", "inline_data"), ("fileData", "file_data")):
        value = part.get(camel, part.get(snake))
        if isinstance(value, dict):
            mime = value.get("mimeType") or value.get("mime_type") or ""
            rest = {k: v for k, v in value.items() if k not in {"mimeType", "mime_type", "data", "fileUri", "file_uri"}}
            fields = {k: v for k, v in part.items() if k not in {camel, snake}} | ({camel: rest} if rest else {})
            return [Media(_kind(mime), mime, value.get("data") or "", url=value.get("fileUri") or value.get("file_uri") or "",
                          extra=extra(fields))]
    return []


def _kind(mime: str) -> str:
    return "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else "file"


def _content(text: str, media: tuple[Media, ...]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": text}] if text or not media else []
    for m in media:
        part = extras(m)
        if m.data:
            parts.append({**part, "inlineData": {**part.get("inlineData", {}), "mimeType": m.mime, "data": m.data}})
        else:
            parts.append({**part, "fileData": {**part.get("fileData", {}), "mimeType": m.mime, "fileUri": m.url}})
    return parts


def _has(part: dict[str, Any], name: str) -> bool:
    return name in part or _snake(name) in part


def _snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lower()
