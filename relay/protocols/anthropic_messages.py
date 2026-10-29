"""Anthropic Messages API (`POST /v1/messages`), as spoken by Claude Code.

Items are whole messages, but a user message answering several tool uses (parallel calls) is
an item per tool result, joined again when sent: each call's result can be rewritten alone. A tool
call and its results live in an assistant message and the following user message, so a cut is
only illegal between those two, and between the results.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.ir import Item, Kind, Media
from .base import OVERFLOW_PHRASES, Body, WireItem, canonical_json, decoded, error_message

SUMMARY_MAX_TOKENS = 32_000


class AnthropicMessages:
    name = "anthropic_messages"
    paths = ("/messages",)

    def managed(self, body: Body) -> bool:
        return isinstance(body.get("messages"), list)

    def items(self, body: Body) -> list[WireItem]:
        return [part for message in body.get("messages") or [] for part in _answers(message)]

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return {**body, "messages": _rejoined(items)}

    def classify(self, item: WireItem) -> Item:
        role, blocks = item.get("role"), _blocks(item.get("content"))
        types = {block.get("type") for block in blocks}
        text = "\n".join(filter(None, (_block_text(block) for block in blocks)))
        if role == "system":
            return Item(Kind.SYSTEM, text)
        if role == "assistant":
            # Redacted thinking is encrypted thinking; a thinking block's signature only verifies it.
            opaque = sum(decoded(block.get("data")) for block in blocks if block.get("type") == "redacted_thinking")
            if "tool_use" in types:
                return Item(Kind.TOOL_CALL, text, opaque=opaque)
            if types <= {"thinking", "redacted_thinking"}:
                return Item(Kind.REASONING, text, opaque=opaque)
            return Item(Kind.ASSISTANT, text, opaque=opaque)
        media = tuple(m for block in blocks for m in _media(block))
        if "tool_result" in types:
            return Item(Kind.TOOL_RESULT, text, media=media)
        return Item(Kind.USER, text, media=media)

    def canonical(self, item: WireItem) -> bytes:
        """Clients move `cache_control` to the newest message and may resend a
        one-block list as a plain string, so both are normalized away."""

        blocks = [_without_cache_control(block) for block in _blocks(item.get("content"))]
        return canonical_json({**item, "content": blocks})

    def boundaries(self, items: list[WireItem]) -> frozenset[int]:
        legal = {0}
        for index, item in enumerate(items):
            if self.classify(item).kind is not Kind.TOOL_CALL and not (index + 1 < len(items) and _continues(item, items[index + 1])):
                legal.add(index + 1)
        return frozenset(legal)

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        """A `system` message may only follow a user message and precede the model's turn (or
        end the request): keep them at the end of a mid-turn head, drop them at a turn start."""

        rest = tuple(item for item in head if item.kind is not Kind.SYSTEM)
        return rest + tuple(item for item in head if item.kind is Kind.SYSTEM) if mid_turn else rest

    def user_message(self, text: str) -> WireItem:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def write(self, item: Item) -> WireItem:
        if item.kind is Kind.ASSISTANT and not item.media:
            return {"role": "assistant", "content": [{"type": "text", "text": item.text}]}
        if item.kind in (Kind.USER, Kind.SUMMARY, Kind.CONTEXT):
            return {"role": "user", "content": _content(item.text, item.media)}
        raise ValueError(f"Relay cannot write a {item.kind.value} item")

    def edit(self, wire: WireItem, text: str, media: tuple[Media, ...]) -> WireItem:
        """The text blocks become one; thinking stays as it was (its signature verifies it).
        A user message's tool results share the new content: the first carries it."""

        blocks = _blocks(wire.get("content"))
        if wire.get("role") == "assistant":
            if media or any(b.get("type") in {"tool_use", "server_tool_use"} for b in blocks):
                raise ValueError("the content of a tool call cannot change on its own")
            thinking = [b for b in blocks if b.get("type") in {"thinking", "redacted_thinking"}]
            return {**wire, "content": [*thinking, {"type": "text", "text": text}]}
        results = [b for b in blocks if b.get("type") == "tool_result"]
        if not results:
            return {**wire, "content": _content(text, media)}
        content = [{**results[0], "content": _content(text, media)},
                   *({**b, "content": ""} for b in results[1:])]
        return {**wire, "content": content}

    def orphans(self, items: list[WireItem]) -> set[int]:
        """Messages whose tool uses are not all answered in the very next message, and messages
        answering a tool use the message before them does not make (a message's results: each
        of its items)."""

        def ids(item: WireItem, kind: str, key: str) -> set[str]:
            return {b.get(key) for b in _blocks(item.get("content")) if b.get("type") == kind}

        groups = _groups(items)
        messages = [_joined([items[n] for n in group]) for group in groups]
        lonely = set()
        for n, item in enumerate(messages):
            asked = ids(item, "tool_use", "id") if item.get("role") == "assistant" else set()
            before = ids(messages[n - 1], "tool_use", "id") if n and messages[n - 1].get("role") == "assistant" else set()
            if (asked and not (n + 1 < len(messages) and asked <= ids(messages[n + 1], "tool_result", "tool_use_id"))
                    or not ids(item, "tool_result", "tool_use_id") <= before):
                lonely.update(groups[n])
        return lonely

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        # A system message may not precede a user turn: the note goes before trailing ones.
        end = len(items)
        while end and items[end - 1].get("role") == "system":
            end -= 1
        head, tail = items[:end], items[end:]
        if not head or head[-1].get("role") != "user":
            return [*head, self.user_message(text), *tail]
        content = head[-1].get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        return [*head[:-1], {**head[-1], "content": [*blocks, {"type": "text", "text": text}]}, *tail]

    def offers_tools(self, body: Body) -> bool:
        return bool(body.get("tools"))

    def with_instructions(self, body: Body, text: str) -> Body:
        system = body.get("system")
        if isinstance(system, list):
            return {**body, "system": [*system, {"type": "text", "text": text}]}
        return {**body, "system": f"{system}\n\n{text}" if system else text}

    def preamble(self, body: Body) -> str:
        system = body.get("system") or ""
        if isinstance(system, list):
            system = "\n".join(_block_text(block) for block in system)
        return system + json.dumps(body.get("tools") or [], ensure_ascii=False)

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        trailing = []  # a system message may not precede a user turn, but may end the request
        while items and items[-1].get("role") == "system":
            items, trailing = items[:-1], [items[-1], *trailing]
        thinking = body.get("thinking") if isinstance(body.get("thinking"), dict) else {}
        cap = SUMMARY_MAX_TOKENS + int(thinking.get("budget_tokens") or 0)
        request = {key: value for key, value in body.items() if key != "stream"}
        request.update(
            messages=[*_rejoined(items), self.user_message(prompt), *trailing],
            stream=False,
            max_tokens=min(int(body.get("max_tokens") or cap), cap),
        )
        if request.get("tools"):
            request["tool_choice"] = {"type": "none"}
        if isinstance(request.get("output_config"), dict):  # no structured-output schema
            request["output_config"] = {k: v for k, v in request["output_config"].items() if k != "format"}
        return request

    def request(self, body: Body, system: str, items: list[WireItem]) -> Body:
        request = self.summary_request({k: v for k, v in body.items() if k not in {"tools", "tool_choice"}}, items, "")
        request.update(messages=_rejoined(items), system=system)
        return request

    def stream_result(self, events: list[Body]) -> Body:
        deltas = (e.get("delta") or {} for e in events if e.get("type") == "content_block_delta")
        stop = next((e["delta"].get("stop_reason") for e in reversed(events)
                     if e.get("type") == "message_delta" and isinstance(e.get("delta"), dict)), None)
        return {"content": [{"type": "text", "text": "".join(d.get("text", "") for d in deltas)}], "stop_reason": stop}

    def output_text(self, payload: Body) -> str:
        return "\n".join(
            block["text"]
            for block in payload.get("content") or []
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ).strip()

    def finished(self, payload: Body) -> bool:
        return payload.get("stop_reason") in {"end_turn", "stop_sequence"}

    def usage(self, payload: Body) -> int | None:
        """Prompt tokens from a message, or from the stream's message_start / message_delta
        (gateways such as LiteLLM report zero at the start and the count at the end)."""

        if payload.get("type") == "message_start":
            payload = payload.get("message") or {}
        elif payload.get("type") not in {"message", "message_delta"}:
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), int):
            return None
        keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        return sum(usage.get(key) or 0 for key in keys) or None

    def cached(self, payload: Body) -> int | None:
        if payload.get("type") == "message_start":
            payload = payload.get("message") or {}
        usage = payload.get("usage")
        return usage.get("cache_read_input_tokens") if isinstance(usage, dict) else None

    def is_overflow(self, status: int, payload: Any) -> bool:
        _, message = error_message(payload)
        return status in {400, 413} and any(p in message for p in OVERFLOW_PHRASES)


class MessageParts(AnthropicMessages):
    """The Messages API of a harness that sends consecutive user messages as one (its own context,
    what the user typed and the results of the model's tool calls share a message): each part
    is an item, itself a user message, and consecutive user messages go out as one. `together`:
    the (opening, closing) markers of what the harness writes over several text blocks (its own
    summary, say), kept in one part."""

    def __init__(self, together: tuple[tuple[str, str], ...] = ()) -> None:
        self.together = together

    def items(self, body: Body) -> list[WireItem]:
        return [part for message in super().items(body) for part in _split(message, self.together)]

    def with_items(self, body: Body, items: list[WireItem]) -> Body:
        return super().with_items(body, _join(items))

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        request = super().summary_request(body, items, prompt)
        return {**request, "messages": _join(request["messages"])}

    def request(self, body: Body, system: str, items: list[WireItem]) -> Body:
        request = super().request(body, system, items)
        return {**request, "messages": _join(request["messages"])}


def _answers(message: WireItem) -> list[WireItem]:
    """A user message answering several tool uses as a message per tool result, each with the
    blocks after it up to the next result (those before the first, with the first)."""

    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list) or sum(map(_result, content)) < 2:
        return [message]
    parts: list[list[Any]] = []
    for block in content:
        if not parts or _result(block) and any(map(_result, parts[-1])):
            parts.append([])
        parts[-1].append(block)
    return [{**message, "content": part} for part in parts]


def _continues(previous: WireItem, item: WireItem) -> bool:
    """`item` answers tool uses beside `previous` (`_answers`): a user message starting with a tool
    result, right after one holding results."""

    content = item.get("content")
    return (previous.get("role") == item.get("role") == "user" and isinstance(content, list) and bool(content)
            and _result(content[0]) and any(map(_result, _blocks(previous.get("content")))))


def _groups(items: list[WireItem]) -> list[list[int]]:
    """The items of each message (`_answers`), by position."""

    groups: list[list[int]] = []
    for n, item in enumerate(items):
        if groups and _continues(items[n - 1], item):
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _joined(parts: list[WireItem]) -> WireItem:
    return parts[0] if len(parts) == 1 else {**parts[0], "content": [b for part in parts for b in _blocks(part.get("content"))]}


def _rejoined(items: list[WireItem]) -> list[WireItem]:
    """The messages `_answers` made items of, as one message again."""

    return [_joined([items[n] for n in group]) for group in _groups(items)]


def _result(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_result"


def _split(message: WireItem, together: tuple[tuple[str, str], ...] = ()) -> list[WireItem]:
    """A user message as its parts: the tool results together, then every text block with the
    blocks after it (its images), but a block that opens one of `together` and the blocks up to
    the one that closes it as one."""

    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return [message]
    parts: list[list[dict[str, Any]]] = []
    closing = None  # what the open part waits for
    for block in content:
        result = block.get("type") == "tool_result"
        held = {b.get("type") for b in parts[-1]} if parts else set()
        if closing is None and (not parts or result != ("tool_result" in held)
                                or (block.get("type") == "text" and "text" in held)):
            parts.append([])
        parts[-1].append(block)
        text = block.get("text", "") if block.get("type") == "text" else ""
        if closing is not None:
            closing = None if closing in text else closing
        else:
            closing = next((end for start, end in together if start in text and end not in text.split(start)[-1]), None)
    return [{**message, "content": part} for part in parts] if len(parts) > 1 else [message]


def _join(items: list[WireItem]) -> list[WireItem]:
    """Consecutive user messages as one, their blocks in order."""

    messages: list[WireItem] = []
    for item in items:
        if messages and messages[-1].get("role") == "user" == item.get("role"):
            messages[-1] = {**messages[-1], "content": [*_blocks(messages[-1].get("content")), *_blocks(item.get("content"))]}
        else:
            messages.append(item)
    return messages


def _blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [block for block in content or [] if isinstance(block, dict)]


def _block_text(block: dict[str, Any]) -> str:
    kind = block.get("type") or ""
    if kind == "text":
        return block.get("text") or ""
    if kind == "thinking":
        return block.get("thinking") or ""
    if kind in {"tool_use", "server_tool_use"}:
        return f"{block.get('name', '')}({json.dumps(block.get('input'), ensure_ascii=False)})"
    if kind == "tool_result" or kind.endswith("_tool_result"):
        content = block.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(_block_text(inner) for inner in _blocks(content))
    return ""


def _media(block: dict[str, Any]) -> list[Media]:
    """The images and documents of a block, a tool result's included."""

    kind = block.get("type")
    if kind == "tool_result":
        content = block.get("content")
        return [m for inner in _blocks(content) for m in _media(inner)] if isinstance(content, list) else []
    if kind not in {"image", "document"}:
        return []
    source = block.get("source") or {}
    media_type = "image" if kind == "image" else "file"
    return [Media(media_type, source.get("media_type", ""), source.get("data", "") if source.get("type") == "base64" else "",
                  source.get("url") or source.get("file_id") or "")]


def _content(text: str, media: tuple[Media, ...]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text or not media else []
    for m in media:
        source = ({"type": "base64", "media_type": m.mime, "data": m.data} if m.data
                  else {"type": "url", "url": m.url} if "://" in m.url else {"type": "file", "file_id": m.url})
        blocks.append({"type": "image" if m.type == "image" else "document", "source": source})
    return blocks


def _without_cache_control(block: dict[str, Any]) -> dict[str, Any]:
    block = {key: value for key, value in block.items() if key != "cache_control"}
    if isinstance(block.get("content"), list):
        block["content"] = [_without_cache_control(inner) for inner in _blocks(block["content"])]
    return block
