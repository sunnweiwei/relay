"""Token approximations shared by the engine and strategies (same rules as Codex)."""

from __future__ import annotations

import math
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .ir import Item

BYTES_PER_TOKEN = 4  # Codex's rule; OpenAI and Gemini tokenizers measured about 4.6, so it errs high
DENSER = (("claude", 2.5),)  # Claude's tokenizer measured about 2.7 bytes a token
_counter: ContextVar[Callable[[str], int] | None] = ContextVar("relay_token_counter", default=None)
_opaque_counter: ContextVar[Callable[[Item], int] | None] = ContextVar("relay_opaque_counter", default=None)
_media_counter: ContextVar[Callable[[Item], int] | None] = ContextVar("relay_media_counter", default=None)


def structured_tokens(value: object, counter: Callable[[str], int]) -> int:
    """Count decoded schema fields with explicit container/separator allowance.

    JSON escaping is a transport representation, not text in a description or
    grammar. Container pairs cost two units, separators and colons one each.
    The provider's private tool rendering remains unknown.
    """
    if isinstance(value, str):
        return counter(value)
    if isinstance(value, list):
        return 2 + max(0, len(value) - 1) + sum(structured_tokens(v, counter) for v in value)
    if isinstance(value, dict):
        return 2 + max(0, len(value) - 1) + sum(
            counter(str(k)) + 1 + structured_tokens(v, counter) for k, v in value.items())
    return counter(json.dumps(value))


def counted_item_tokens(item: Item, counter: Callable[[str], int],
                        opaque_counter: Callable[[Item], int] | None = None,
                        media_counter: Callable[[Item], int] | None = None) -> int:
    """The declared bounded-profile convention, shared with request dispatch."""
    opaque = opaque_counter(item) if opaque_counter else math.ceil(item.opaque / BYTES_PER_TOKEN)
    text = structured_tokens(json.loads(item.token_structure), counter) if item.token_structure else counter(item.text)
    media = media_counter(item) if media_counter else 0
    return text + opaque + media + 8


@contextmanager
def counting_with(counter: Callable[[str], int] | None,
                  opaque_counter: Callable[[Item], int] | None = None,
                  media_counter: Callable[[Item], int] | None = None) -> Iterator[None]:
    """Use the harness's declared text counter inside an unchanged strategy."""
    token = _counter.set(counter)
    opaque_token = _opaque_counter.set(opaque_counter)
    media_token = _media_counter.set(media_counter)
    try:
        yield
    finally:
        _counter.reset(token)
        _opaque_counter.reset(opaque_token)
        _media_counter.reset(media_token)


def bytes_per_token(model: object) -> float:
    name = model.rsplit("/", 1)[-1].lower() if isinstance(model, str) else ""
    return next((size for prefix, size in DENSER if name.startswith(prefix)), BYTES_PER_TOKEN)


def approx_tokens(text: str, per_token: float = BYTES_PER_TOKEN) -> int:
    if counter := _counter.get():
        return counter(text)
    return math.ceil(len(text.encode("utf-8")) / per_token)


def item_tokens(item: Item, per_token: float = BYTES_PER_TOKEN) -> int:
    """An item's text and opaque content (encrypted reasoning or compaction, signatures)."""

    if counter := _counter.get():
        return counted_item_tokens(item, counter, _opaque_counter.get(), _media_counter.get())
    return math.ceil((len(item.text.encode("utf-8")) + item.opaque) / per_token)


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep the head and tail of `text` within `max_tokens`, marking the cut."""

    if counter := _counter.get():
        if counter(text) <= max_tokens:
            return text
        marker = "…truncated…"
        if counter(marker) > max_tokens:
            return ""
        low, high, best = 0, len(text), marker
        while low <= high:
            kept = (low + high) // 2
            left, right = kept // 2, kept - kept // 2
            candidate = text[:left] + marker + (text[-right:] if right else "")
            if counter(candidate) <= max_tokens:
                best, low = candidate, kept + 1
            else:
                high = kept - 1
        return best
    budget = max(0, max_tokens) * BYTES_PER_TOKEN
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    left = encoded[: budget // 2].decode("utf-8", errors="ignore")
    right = encoded[len(encoded) - (budget - budget // 2) :].decode("utf-8", errors="ignore")
    removed = approx_tokens(text) - max(0, max_tokens)
    return f"{left}…{removed} tokens truncated…{right}"
