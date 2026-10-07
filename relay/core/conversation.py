"""Where a conversation can be cut, as compactions cut it: before a turn the user just started,
and the newest user messages within a budget."""

from __future__ import annotations

from .ir import AGENT_KINDS, Item, Kind, Request
from .tokens import approx_tokens, truncate_middle


def pending_cut(request: Request) -> int | None:
    """Everything before a pending user turn, or the whole conversation mid-turn."""

    items = request.current
    last_agent = max((i for i, item in enumerate(items) if item.kind in AGENT_KINDS), default=None)
    if last_agent is None:
        return None
    pending_turn = any(item.kind is Kind.USER for item in items[last_agent + 1 :])
    cut = last_agent + 1 if pending_turn else len(items)
    return cut if cut in request.boundaries else None


def recent_users(items: tuple[Item, ...], budget: int) -> list[Item]:
    """The newest user messages within `budget` tokens; the oldest one kept may be truncated."""

    kept: list[Item] = []
    remaining = budget
    for item in reversed(items):
        if item.kind is not Kind.USER:
            continue
        if remaining <= 0:
            break
        tokens = approx_tokens(item.text)
        if tokens <= remaining and not item.media:
            kept.append(item)
        else:  # rebuilt as text only, truncated to what is left (never keeps media)
            kept.append(Item(Kind.USER, truncate_middle(item.text, remaining)))
        if tokens > remaining:
            break
        remaining -= tokens
    return kept[::-1]
