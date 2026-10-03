"""Token approximations shared by the engine and strategies (same rules as Codex)."""

from __future__ import annotations

import math

from .ir import Item

BYTES_PER_TOKEN = 4  # Codex's rule; OpenAI and Gemini tokenizers measured about 4.6, so it errs high
DENSER = (("claude", 2.5),)  # Claude's tokenizer measured about 2.7 bytes a token


def bytes_per_token(model: object) -> float:
    name = model.rsplit("/", 1)[-1].lower() if isinstance(model, str) else ""
    return next((size for prefix, size in DENSER if name.startswith(prefix)), BYTES_PER_TOKEN)


def approx_tokens(text: str, per_token: float = BYTES_PER_TOKEN) -> int:
    return math.ceil(len(text.encode("utf-8")) / per_token)


def item_tokens(item: Item, per_token: float = BYTES_PER_TOKEN) -> int:
    """An item's text and opaque content (encrypted reasoning or compaction, signatures)."""

    return math.ceil((len(item.text.encode("utf-8")) + item.opaque) / per_token)


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep the head and tail of `text` within `max_tokens`, marking the cut."""

    budget = max(0, max_tokens) * BYTES_PER_TOKEN
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    left = encoded[: budget // 2].decode("utf-8", errors="ignore")
    right = encoded[len(encoded) - (budget - budget // 2) :].decode("utf-8", errors="ignore")
    removed = approx_tokens(text) - max(0, max_tokens)
    return f"{left}…{removed} tokens truncated…{right}"
