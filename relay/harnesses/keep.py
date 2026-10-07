"""What a harness keeps verbatim when it compacts by itself (`Native.keep`): functions of the
request giving the items kept ahead of the summary and where the history kept after it begins
(None: nothing would be summarized)."""

from __future__ import annotations

from ..core.conversation import pending_cut, recent_users
from ..core.ir import AGENT_KINDS, Keep, Kind, Request
from ..core.tokens import item_tokens

ACTIONS = AGENT_KINDS - {Kind.TOOL_RESULT}  # what opens a round of the model's


def pending() -> Keep:
    """Only a turn the summary does not cover (Crush, Goose, WorkBuddy)."""

    return lambda request: _kept(pending_cut(request))


def users(budget: int) -> Keep:
    """The newest user messages within `budget` tokens, ahead of a summary of everything (Kimi Code)."""

    def keep(request: Request):
        if pending_cut(request) is None:
            return None
        return tuple(recent_users(request.current, budget)), len(request.current)

    return keep


def round() -> Keep:  # noqa: A001 (the model's round)
    """The model's last round: its last response, with the results or the new turn after it (Claude Code)."""

    return lambda request: _kept(_round(request))


def last() -> Keep:
    """A turn the summary does not cover, or mid-turn the model's last round (nanobot)."""

    def keep(request: Request):
        cut = pending_cut(request)
        return _kept(_round(request) if cut == len(request.current) else cut)

    return keep


def tail(share: float = 0, tokens: int = 0, cap: int = 0, head: int = 0, least: int = 0, most: int = 0) -> Keep:
    """The newest items within `tokens`, or `share` of the request's (at most `cap`), but at least
    `least` items and at most `most` of the model's rounds, never opening on a tool's result;
    after the `head` first items, which are kept ahead of the summary (pi, Hermes, OpenCode)."""

    def keep(request: Request):
        items, budget = request.current, _budget(request, share, tokens, cap)
        cut, kept = len(items), 0
        while cut > head and (kept + item_tokens(items[cut - 1]) <= budget or len(items) - cut < least):
            cut -= 1
            kept += item_tokens(items[cut])
        if most and len(starts := _rounds(request)) >= most:
            cut = max(cut, starts[-most])
        while cut > head and (cut not in request.boundaries or cut < len(items) and items[cut].kind is Kind.TOOL_RESULT):
            cut -= 1
        return (tuple(items[:head]), cut) if cut > head else None

    return keep


def turns(share: float) -> Keep:
    """The turn in progress, and the turns before it within `share` of the request (DeepSeek Harness)."""

    def keep(request: Request):
        items, budget = request.current, _budget(request, share)
        starts = [n for n, item in enumerate(items) if item.kind is Kind.USER]
        if not starts or starts[-1] == 0:
            return None
        cut = starts[-1]
        for start in reversed(starts[:-1]):
            if sum(item_tokens(item) for item in items[start:]) > budget:
                break
            cut = start
        return _kept(cut if cut in request.boundaries else None)

    return keep


def split(share: float) -> Keep:
    """From the first user message past `1 - share` of the history's text (else the last one);
    nothing when the model's last answer ended the turn (Gemini CLI)."""

    def keep(request: Request):
        items = request.current
        if items and items[-1].kind is Kind.ASSISTANT:
            return _kept(len(items))
        sizes = [len(item.text) for item in items]
        total, seen = sum(sizes), 0
        starts = [n for n, item in enumerate(items) if item.kind is Kind.USER and n in request.boundaries]
        for n, size in enumerate(sizes):
            if n in starts and seen >= (1 - share) * total:
                return _kept(n)
            seen += size
        return _kept(starts[-1] if starts else None)

    return keep


def _kept(cut: int | None):
    return ((), cut) if cut else None


def _round(request: Request) -> int | None:
    starts = _rounds(request)
    return starts[-1] if starts and starts[-1] in request.boundaries else None


def _rounds(request: Request) -> list[int]:
    items = request.current
    return [n for n, item in enumerate(items) if item.kind in ACTIONS and (n == 0 or items[n - 1].kind not in ACTIONS)]


def _budget(request: Request, share: float = 0, tokens: int = 0, cap: int = 0) -> float:
    """`tokens`, or `share` of the request's (the harness's threshold, when it compacts by
    itself: a share of what it compacts at), at most `cap`."""

    budget = tokens or share * request.tokens
    return min(budget, cap) if cap else budget
