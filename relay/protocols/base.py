"""The codec interface: everything Relay needs to know about one wire protocol."""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..core.ir import Item, Media

Body = dict[str, Any]
WireItem = dict[str, Any]


class Codec(Protocol):
    name: str
    paths: tuple[str, ...]  # suffixes of the request paths this codec manages, e.g. "/responses"

    def managed(self, body: Body) -> bool:
        """False for requests Relay cannot rewrite (e.g. server-side conversation state)."""

    def items(self, body: Body) -> list[WireItem]: ...

    def with_items(self, body: Body, items: list[WireItem]) -> Body: ...

    def classify(self, item: WireItem) -> Item:
        """Protocol-level kind and flattened text of one wire item."""

    def canonical(self, item: WireItem) -> bytes:
        """Identity of an item for prefix matching, with volatile fields removed."""

    def boundaries(self, items: list[WireItem]) -> frozenset[int]: ...

    def user_message(self, text: str) -> WireItem: ...

    def write(self, item: Item) -> WireItem:
        """The wire item for an item Relay writes: a user message (with its media), or an
        assistant one. ValueError for any other kind."""

    def edit(self, wire: WireItem, text: str, media: tuple[Media, ...]) -> WireItem:
        """`wire` with `text` and `media` as its content, its structure (role, the tool call a
        result answers) kept. ValueError where content cannot change on its own (a tool call)."""

    def orphans(self, items: list[WireItem]) -> set[int]:
        """The items the API would reject where they stand: a tool call without its result, a
        result without its call."""

    def note(self, items: list[WireItem], text: str) -> list[WireItem]:
        """`items` with `text` added at the end, for the model to read before it answers."""

    def with_instructions(self, body: Body, text: str) -> Body:
        """`body` with `text` added to its system prompt."""

    def offers_tools(self, body: Body) -> bool:
        """Whether the request lets the model call tools: an agent's turn, not a side call (a title)."""

    def arrange(self, head: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        """Move or drop system items of a rewritten head where the protocol requires it."""

    def preamble(self, body: Body) -> str:
        """Instructions and tool definitions, used for cold token estimates."""

    def summary_request(self, body: Body, items: list[WireItem], prompt: str) -> Body:
        """Continue the same request with `prompt` as the final user turn."""

    def stream_result(self, events: list[Body]) -> Body:
        """The response body equivalent to a list of streamed events."""

    def output_text(self, payload: Body) -> str: ...

    def finished(self, payload: Body) -> bool:
        """Whether the response ended normally: not cut off by a token limit or a broken stream."""

    def usage(self, payload: Body) -> int | None:
        """Prompt tokens reported by a response body or a streaming event."""

    def is_overflow(self, status: int, payload: Any) -> bool:
        """Whether an error response means the prompt exceeded the context window."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def data_media(kind: str, url: str, mime: str = "") -> Media:
    """Media from a URL, which may be a `data:` URL carrying the content."""

    if url.startswith("data:") and ";base64," in url:
        head, data = url[5:].split(";base64,", 1)
        return Media(kind, head or mime, data)
    return Media(kind, mime, url=url)


def data_url(media: Media) -> str:
    return f"data:{media.mime};base64,{media.data}" if media.data else media.url


def decoded(value: Any) -> int:
    """Bytes of base64 content (encrypted reasoning, signatures) once decoded."""

    return len(value) * 3 // 4 if isinstance(value, str) else 0


def json_text(text: Any) -> Any:
    """A JSON-encoded string (tool call arguments) as its value, so that re-serializing it
    (nanobot's compaction request adds spaces) does not change an item's identity."""

    try:
        return json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return text


def error_message(payload: Any) -> tuple[str, str]:
    """(code, lowercase message) of a typical JSON error body."""

    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        return "", str(error).lower()
    return str(error.get("code") or error.get("type") or ""), str(error.get("message", "")).lower()


OVERFLOW_PHRASES = (
    "exceeds the maximum number of tokens",
    "input token count",
    "context window",
    "context length",
    "maximum context",
    "prompt is too long",
    "too many tokens",
)
