"""OpenAI Chat Completions (`POST .../chat/completions`), the common OpenAI-compatible API.

Items are messages. An assistant message with `tool_calls` and the `tool` messages
answering it form one group.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.ir import Item, Kind, Media
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, data_media, data_url, error_message, json_text


# Message fields that reach the model, including common provider extensions for reasoning.
MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id", "function_call", "refusal", "audio",
                  "reasoning", "reasoning_content", "reasoning_details", "thinking"}


class OpenAIChat:
    name = "openai_chat"
    paths = ("/chat/completions",)

    def managed(self, body: Body) -> bool:
        return isinstance(body.get("messages"), list)

    def items(self, body: Body) -> list[WireItem]:
        return list(body.get("messages") or [])

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return {**body, "messages": items}

    def classify(self, item: WireItem) -> Item:
        role = item.get("role")
        text, media = _content_text(item.get("content"))
        if role in {"system", "developer"}:
            return Item(Kind.SYSTEM, text)
        if role in {"tool", "function"}:
            return Item(Kind.TOOL_RESULT, text, media=media)
        if role == "assistant":
            calls = [c.get("function") or {} for c in item.get("tool_calls") or []]
            if calls:
                text = "\n".join([text, *(f"{c.get('name', '')}({c.get('arguments', '')})" for c in calls)])
                return Item(Kind.TOOL_CALL, text.strip())
            return Item(Kind.ASSISTANT, text)
        return Item(Kind.USER, text, media=media)

    def canonical(self, item: WireItem) -> bytes:
        """What the model reads of a message: no harness metadata (CodeBuddy adds `usage` and
        more on resume), no `cache_control`, string content as one text part, and no `name`
        on tool results (Hermes drops it on resume); call arguments compare as JSON values."""

        message = {k: v for k, v in item.items() if k in MESSAGE_FIELDS}
        if message.get("role") == "tool":
            message.pop("name", None)
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if isinstance(content, list):
            content = [{k: v for k, v in part.items() if k != "cache_control"} for part in content
                       if isinstance(part, dict)]
        calls = [{**call, "function": {**call["function"], "arguments": json_text(call["function"].get("arguments"))}}
                 if isinstance(call, dict) and isinstance(call.get("function"), dict) else call
                 for call in message.get("tool_calls") or []]
        return canonical_json({**message, "content": content, **({"tool_calls": calls} if calls else {})})

    def boundaries(self, items: list[WireItem]) -> frozenset[int]:
        legal, pending = {0}, set()
        for index, item in enumerate(items):
            if item.get("role") == "assistant":
                pending |= {c.get("id") for c in item.get("tool_calls") or []}
            elif item.get("role") == "tool":
                pending.discard(item.get("tool_call_id"))
            if not pending:
                legal.add(index + 1)
        return frozenset(legal)

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        return head

    def user_message(self, text: str) -> WireItem:
        return {"role": "user", "content": text}

    def write(self, item: Item) -> WireItem:
        if item.kind is Kind.ASSISTANT and not item.media:
            return {"role": "assistant", "content": item.text}
        if item.kind in (Kind.USER, Kind.SUMMARY, Kind.CONTEXT):
            return {"role": "user", "content": _content(item.text, item.media)}
        raise ValueError(f"Relay cannot write a {item.kind.value} item")

    def edit(self, wire: WireItem, text: str, media: tuple[Media, ...]) -> WireItem:
        if wire.get("tool_calls") or wire.get("function_call") or (media and wire.get("role") == "assistant"):
            raise ValueError("the content of a tool call cannot change on its own")
        return {**wire, "content": _content(text, media)}

    def legal(self, items: list[WireItem]) -> str | None:
        """The tool messages answering an assistant message's calls follow it, and only them."""

        waiting: set[str] = set()
        for index, item in enumerate(items):
            if item.get("role") == "tool":
                if item.get("tool_call_id") not in waiting:
                    return f"message {index} answers a tool call that is not waiting for it"
                waiting.discard(item.get("tool_call_id"))
                continue
            if waiting:
                return f"tool calls {sorted(waiting)} are not answered"
            if item.get("role") == "assistant":
                waiting = {call.get("id") for call in item.get("tool_calls") or []}
        return f"tool calls {sorted(waiting)} are not answered" if waiting else None

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        return [*items, self.user_message(text)]

    def offers_tools(self, body: Body) -> bool:
        return bool(body.get("tools"))

    def with_instructions(self, body: Body, text: str) -> Body:
        messages = list(body.get("messages") or [])
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content")
            content = [*content, {"type": "text", "text": text}] if isinstance(content, list) else f"{content}\n\n{text}"
            messages[0] = {**messages[0], "content": content}
        else:
            messages.insert(0, {"role": "system", "content": text})
        return {**body, "messages": messages}

    def preamble(self, body: Body) -> str:
        return json.dumps(body.get("tools") or [], ensure_ascii=False)

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        drop = {"stream", "stream_options", "response_format", "n"}
        request = {key: value for key, value in body.items() if key not in drop}
        request.update(messages=[*items, self.user_message(prompt)], stream=False)
        if request.get("tools"):
            request["tool_choice"] = "none"
        return request

    def stream_result(self, events: list[Body]) -> Body:
        choices = [choice for event in events for choice in event.get("choices") or []]
        text = "".join((choice.get("delta") or {}).get("content") or "" for choice in choices)
        reason = next((c["finish_reason"] for c in reversed(choices) if c.get("finish_reason")), None)
        return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": reason}]}

    def output_text(self, payload: Body) -> str:
        choices = payload.get("choices") or [{}]
        return _content_text((choices[0].get("message") or {}).get("content"))[0].strip()

    def finished(self, payload: Body) -> bool:
        """A finish reason arrived and it is not a cutoff (gateways name normal stops differently)."""

        reason = (payload.get("choices") or [{}])[0].get("finish_reason")
        return reason is not None and reason not in {"length", "content_filter"}

    def usage(self, payload: Body) -> int | None:
        usage = payload.get("usage")
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        return tokens if isinstance(tokens, int) else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        code, message = error_message(payload)
        return status in {400, 413} and (
            code == "context_length_exceeded" or any(p in message for p in OVERFLOW_PHRASES)
        )


def _content_text(content: Any) -> tuple[str, tuple[Media, ...]]:
    if isinstance(content, str):
        return content, ()
    texts, media = [], []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            media.append(data_media("image", url.get("url", "") if isinstance(url, dict) else str(url or "")))
        elif part.get("type") == "input_audio" and isinstance(part.get("input_audio"), dict):
            audio = part["input_audio"]
            media.append(Media("audio", f"audio/{audio.get('format', '')}", audio.get("data") or ""))
        elif part.get("type") == "file" and isinstance(part.get("file"), dict):
            file = part["file"]
            media.append(data_media("file", file.get("file_data") or file.get("file_id") or ""))
        else:
            media.append(Media(str(part.get("type"))))
    return "\n".join(texts), tuple(media)


def _content(text: str, media: tuple[Media, ...]) -> str | list[dict[str, Any]]:
    if not media:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for m in media:
        if m.type == "image":
            parts.append({"type": "image_url", "image_url": {"url": data_url(m)}})
        elif m.type == "audio":
            parts.append({"type": "input_audio", "input_audio": {"data": m.data, "format": m.mime.split("/")[-1]}})
        else:
            parts.append({"type": "file", "file": {"file_data": data_url(m)} if m.data else {"file_id": m.url}})
    return parts
