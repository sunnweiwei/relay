"""Token approximations shared by the engine and strategies (same rules as Codex)."""

from __future__ import annotations

BYTES_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return (len(text.encode("utf-8")) + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN


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
