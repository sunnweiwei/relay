"""OpenAI Responses API (`POST /v1/responses`), as spoken by Codex and the OpenAI SDK."""

from __future__ import annotations

import json
from typing import Any

from ..core.ir import Item, Kind, Media
from .base import (OVERFLOW_PHRASES, Body, WireItem, canonical_json, data_media, data_url, decoded, error_message,
                   json_text)

_ROLES = {
    "system": Kind.SYSTEM,
    "developer": Kind.SYSTEM,
    "user": Kind.USER,
    "assistant": Kind.ASSISTANT,
}
_TEXT_PARTS = {"input_text", "output_text", "text", "summary_text", "reasoning_text"}


class OpenAIResponses:
    name = "openai_responses"
    paths = ("/responses",)

    def managed(self, body: Body) -> bool:
        return (
            isinstance(body.get("input"), list)
            and body.get("previous_response_id") is None
            and body.get("conversation") is None
        )

    def items(self, body: Body) -> list[WireItem]:
        return list(body.get("input") or [])

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return {**body, "input": items}

    def classify(self, item: WireItem) -> Item:
        kind = item.get("type", "message")
        if kind != "message" and item.get("role") in {"system", "developer"}:
            return Item(Kind.SYSTEM, _string(item))  # e.g. Codex's `additional_tools` declarations
        if kind == "message":
            text, media = _content_text(item.get("content"))
            return Item(_ROLES.get(item.get("role"), Kind.OTHER), text, media=media)
        if kind == "reasoning":
            return Item(Kind.REASONING, _content_text(item.get("summary"))[0], opaque=decoded(item.get("encrypted_content")))
        if kind == "compaction":  # opaque server-side compaction
            return Item(Kind.SUMMARY, opaque=decoded(item.get("encrypted_content")))
        if kind.endswith("_output"):
            text, media = _content_text(item.get("output"))
            return Item(Kind.TOOL_RESULT, text, media=media)
        if kind.endswith("_call"):
            fields = (item.get(key) for key in ("name", "arguments", "input", "action"))
            return Item(Kind.TOOL_CALL, " ".join(_string(value) for value in fields if value))
        return Item(Kind.OTHER, _string(item))

    def canonical(self, item: WireItem) -> bytes:
        """A reasoning item is its encrypted content: resumed sessions drop its display summary
        (OpenClaw) or write absent content back as null or [] (Codex `exec resume`). Item ids and
        statuses are the API's metadata, which harnesses may drop (nanobot's compaction request),
        and call arguments compare as JSON values."""

        if item.get("type") != "item_reference":  # (a reference is its id)
            item = {k: v for k, v in item.items() if k not in {"id", "status"}}
        if item.get("type") == "reasoning":
            item = {k: v for k, v in item.items() if k != "summary" and not (k == "content" and not v)}
        if "arguments" in item:
            item = {**item, "arguments": json_text(item["arguments"])}
        return canonical_json(item)

    def boundaries(self, items: list[WireItem]) -> frozenset[int]:
        """Never separate a call from its output, or reasoning (and the commentary
        that may follow it) from the item it precedes."""

        legal, pending = {0}, set()
        for index, item in enumerate(items):
            kind, call_id = item.get("type", "message"), item.get("call_id")
            if call_id and kind.endswith("_output"):
                pending.discard(call_id)
            elif call_id and kind.endswith("_call"):
                pending.add(call_id)
            if not pending and kind != "reasoning" and item.get("phase") != "commentary":
                legal.add(index + 1)
        return frozenset(legal)

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        return head

    def user_message(self, text: str) -> WireItem:
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

    def write(self, item: Item) -> WireItem:
        if item.kind is Kind.ASSISTANT and not item.media:
            return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": item.text}]}
        if item.kind in (Kind.USER, Kind.SUMMARY, Kind.CONTEXT):
            return {"type": "message", "role": "user", "content": _content(item.text, item.media)}
        raise ValueError(f"Relay cannot write a {item.kind.value} item")

    def edit(self, wire: WireItem, text: str, media: tuple[Media, ...]) -> WireItem:
        kind = wire.get("type", "message")
        if kind == "message" and wire.get("role") == "assistant" and not media:
            return {**wire, "content": [{"type": "output_text", "text": text}]}
        if kind == "message" and wire.get("role") != "assistant":
            return {**wire, "content": _content(text, media)}
        if kind.endswith("_output"):
            return {**wire, "output": _content(text, media) if media else text}
        raise ValueError(f"the content of a {kind} item cannot change on its own")

    def orphans(self, items: list[WireItem]) -> set[int]:
        """Calls without an output, and outputs without their call (the API pairs them by call id)."""

        calls = {item["call_id"]: n for n, item in enumerate(items)
                 if item.get("call_id") and item.get("type", "").endswith("_call")}
        answered = {item["call_id"] for item in items if item.get("call_id") and item.get("type", "").endswith("_output")}
        lonely = {n for n, item in enumerate(items) if item.get("call_id") and item.get("type", "").endswith("_output")
                  and calls.get(item["call_id"], len(items)) > n}
        return lonely | {n for call_id, n in calls.items() if call_id not in answered}

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        return [*items, self.user_message(text)]

    def with_instructions(self, body: Body, text: str) -> Body:
        if body.get("instructions"):
            return {**body, "instructions": f"{_string(body['instructions'])}\n\n{text}"}
        # Codex sends its instructions as developer messages instead: one more after them.
        items = list(body.get("input") or [])
        at = next((n for n, item in enumerate(items) if item.get("role") not in ("developer", "system")), len(items))
        message = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}
        return {**body, "input": [*items[:at], message, *items[at:]]}

    def offers_tools(self, body: Body) -> bool:  # Codex lists its tools in an `additional_tools` item
        return bool(body.get("tools")) or any(item.get("type") == "additional_tools" for item in body.get("input") or [])

    def preamble(self, body: Body) -> str:
        return _string(body.get("instructions") or "") + _string(body.get("tools") or "")

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        request = {key: value for key, value in body.items() if key != "background"}
        request.update(input=[*items, self.user_message(prompt)], store=False)
        if request.get("tools") or any(item.get("type") == "additional_tools" for item in items):
            request["tool_choice"] = "none"
        if isinstance(request.get("text"), dict):  # no structured-output schema for the summary
            request["text"] = {k: v for k, v in request["text"].items() if k != "format"}
        return request

    def request(self, body: Body, system: str, items: list[WireItem]) -> Body:
        drop = {"background", "tools", "tool_choice", "previous_response_id", "prompt"}  # (Codex needs parallel_tool_calls kept)
        request = {key: value for key, value in body.items() if key not in drop}
        request.update(instructions=system, input=items, store=False)
        if isinstance(request.get("text"), dict):
            request["text"] = {k: v for k, v in request["text"].items() if k != "format"}
        return request

    def stream_result(self, events: list[Body]) -> Body:
        final = next(
            (e.get("response") for e in reversed(events) if e.get("type", "").startswith("response.")
             and isinstance(e.get("response"), dict) and e["response"].get("status") != "in_progress"),
            None,
        ) or {}
        done = [e["item"] for e in events if e.get("type") == "response.output_item.done"]
        return {**final, "output": final.get("output") or done}

    def output_text(self, payload: Body) -> str:
        parts: list[str] = []  # each message once: gpt-6 repeats its commentary at times
        for item in payload.get("output") or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                if (text := _content_text(item.get("content"))[0]) and text not in parts:
                    parts.append(text)
        return "\n".join(parts).strip() or str(
            payload.get("output_text") or ""
        ).strip()

    def finished(self, payload: Body) -> bool:
        return payload.get("status") == "completed"

    def usage(self, payload: Body) -> int | None:
        if payload.get("type") in {"response.completed", "response.incomplete"}:
            payload = payload.get("response") or {}
        usage = payload.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        return tokens if isinstance(tokens, int) else None

    def cached(self, payload: Body) -> int | None:
        if payload.get("type") in {"response.completed", "response.incomplete"}:
            payload = payload.get("response") or {}
        details = (payload.get("usage") or {}).get("input_tokens_details") if isinstance(payload.get("usage"), dict) else None
        return details.get("cached_tokens") if isinstance(details, dict) else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        code, message = error_message(payload)
        return status == 400 and (
            code == "context_length_exceeded" or any(p in message for p in OVERFLOW_PHRASES)
        )


def _content_text(content: Any) -> tuple[str, tuple[Media, ...]]:
    """Text and media of a message's content, or of a tool output."""

    if not isinstance(content, list):
        return content if isinstance(content, str) else "", ()
    texts, media = [], []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in _TEXT_PARTS and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "input_image":
            media.append(data_media("image", part.get("image_url") or part.get("file_id") or ""))
        elif kind == "input_file":
            media.append(data_media("file", part.get("file_data") or part.get("file_url") or part.get("file_id") or ""))
        elif kind == "input_audio" and isinstance(part.get("input_audio"), dict):
            audio = part["input_audio"]
            media.append(Media("audio", f"audio/{audio.get('format', '')}", audio.get("data") or ""))
        else:
            media.append(Media(str(kind)))
    return "\n".join(texts), tuple(media)


def _content(text: str, media: tuple[Media, ...]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": text}] if text or not media else []
    for m in media:
        if m.type == "image":
            parts.append({"type": "input_image", "image_url": data_url(m)} if m.data or "://" in m.url
                         else {"type": "input_image", "file_id": m.url})
        elif m.type == "audio":
            parts.append({"type": "input_audio", "input_audio": {"data": m.data, "format": m.mime.split("/")[-1]}})
        elif m.data or "://" in m.url:
            parts.append({"type": "input_file", **({"file_data": data_url(m)} if m.data else {"file_url": m.url})})
        else:
            parts.append({"type": "input_file", "file_id": m.url})
    return parts


def _string(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
