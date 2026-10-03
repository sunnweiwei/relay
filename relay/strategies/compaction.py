"""Codex's local context compaction (codex-rs/core/src/compact.rs) as a Relay strategy.

When the prompt reaches the threshold (90% of the context window, or a growth budget
since the window began; 95% of the window always triggers), the conversation is
summarized with Codex's prompt and rebuilt like Codex's replacement history:

    mid-turn:    [system] [recent users…] [context] [last user] [summary]
    turn start:  [system] [recent users…] [summary] [context] [new turn…]

Mid-turn (the request ends with tool results) everything is summarized and the summary
comes last, with the initial context just above the last real user message. At the start
of a turn the summary covers everything before the new turn, and the initial context is
re-injected after it, followed by the new turn verbatim. Recent user messages are kept
within `retain_user_tokens`, newest first. System messages (Codex's base instructions)
stay first. A compaction that would free less than `min_gain` of the budget is skipped,
so a threshold set too close to the fixed prompt overhead cannot trigger a summary on
every request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind, Rewrite, View
from ..core.tokens import approx_tokens, truncate_middle
from ..prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from .base import Summarizer

DEFAULT_WINDOW = 128_000
HARD_LIMIT = 0.95  # Codex's effective_context_window_percent


@dataclass(frozen=True)
class Compaction:
    threshold: int | None = None  # prompt tokens; defaults to ratio × context window
    growth: int | None = None  # or: tokens added since the window began (Codex's BodyAfterPrefix)
    ratio: float = 0.9  # Codex's auto_compact_token_limit: 90% of the window
    retain_user_tokens: int = 20_000  # COMPACT_USER_MESSAGE_MAX_TOKENS in Codex
    min_gain: float = 0.1  # skip unless at least this share of the threshold is freed
    name: str = "compaction"

    def __post_init__(self) -> None:
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.growth is not None and self.growth <= 0:
            raise ValueError("growth must be positive")
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if self.retain_user_tokens < 0:
            raise ValueError("retain_user_tokens cannot be negative")
        if not 0 <= self.min_gain < 1:
            raise ValueError("min_gain must be in [0, 1)")

    @classmethod
    def from_env(cls) -> Compaction:
        threshold, growth = os.getenv("RELAY_COMPACT_THRESHOLD"), os.getenv("RELAY_COMPACT_GROWTH")
        return cls(
            threshold=int(threshold) if threshold else None,
            growth=int(growth) if growth else None,
            ratio=float(os.getenv("RELAY_COMPACT_RATIO", "0.9")),
            retain_user_tokens=int(os.getenv("RELAY_RETAIN_USER_TOKENS", "20000")),
            min_gain=float(os.getenv("RELAY_COMPACT_MIN_GAIN", "0.1")),
        )

    def fingerprint(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "growth": self.growth,
            "ratio": self.ratio,
            "retain_user_tokens": self.retain_user_tokens,
            "min_gain": self.min_gain,
            "prompt": SUMMARIZATION_PROMPT,
        }

    def limit(self, window: int | None) -> int:
        return self.threshold or int(self.ratio * (window or DEFAULT_WINDOW))

    def plan(self, view: View, summarizer: Summarizer) -> Rewrite | None:
        if self.growth:
            budget, over = self.growth, view.base is not None and view.tokens - view.base >= self.growth
        else:
            budget = self.limit(view.window)
            over = view.tokens >= budget
        over = over or (view.window is not None and view.tokens >= HARD_LIMIT * view.window)
        if not view.force and not over:
            return None
        cut = _cut(view)
        if cut is None:
            return None
        items = view.items[:cut]
        system = [item for item in view.initial if item.kind is Kind.SYSTEM]
        context = [item for item in view.initial if item.kind is not Kind.SYSTEM]
        users = self._recent_users(items)
        freed = sum(approx_tokens(i.text) for i in items) - sum(approx_tokens(i.text) for i in system + context + users)
        if not view.force and freed < self.min_gain * budget:
            return None
        summary = Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\n{summarizer.summarize(cut, SUMMARIZATION_PROMPT)}")
        if cut < len(view.items):  # turn start: context is re-injected for the new turn
            # A re-rendered context already holds the new turn's context updates, which go too.
            end = cut
            while view.current and end < len(view.items) and view.items[end].kind in CONTEXT_KINDS:
                end += 1
            return Rewrite(end if end in view.boundaries else cut, (*system, *users, summary, *context))
        return Rewrite(cut, (*system, *users[:-1], *context, *users[-1:], summary))

    def _recent_users(self, items: tuple[Item, ...]) -> list[Item]:
        """The newest user messages within budget; the oldest one kept may be truncated."""

        kept: list[Item] = []
        remaining = self.retain_user_tokens
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


def _cut(view: View) -> int | None:
    """Everything before a pending user turn, or the whole view mid-turn."""

    items = view.items
    last_agent = max((i for i, item in enumerate(items) if item.kind in AGENT_KINDS), default=None)
    if last_agent is None:
        return None
    pending_turn = any(item.kind is Kind.USER for item in items[last_agent + 1 :])
    cut = last_agent + 1 if pending_turn else len(items)
    return cut if cut in view.boundaries else None
