"""Codex's local context compaction (codex-rs/core/src/compact.rs) as a Relay strategy.

When the prompt reaches the threshold (90% of the context window, or a growth budget
since the window began; 95% of the window always triggers), the conversation is
summarized with Codex's prompt and replaced like Codex's replacement history: the newest
user messages within `retain_user_tokens` (newest first; the oldest one kept may be
truncated) and the summary last. Mid-turn (the request ends with tool results) all of it
is summarized; at the start of a turn, as Codex checks the threshold before the new user
message is recorded, the history before the turn must reach it, and the summary covers that
history while the new turn stays verbatim. The harness's own context is not the strategy's concern: the harness
profile puts its current state into the result (Codex: above the last user message kept). With
`min_gain` set (off, as in Codex), a compaction that would free less than that share of the
budget is skipped, so a threshold set too close to the fixed prompt overhead cannot trigger a
summary on every request.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..core.ir import AGENT_KINDS, Context, Item, Kind, Request
from ..core.tokens import approx_tokens, counting_with, item_tokens, truncate_middle
from ..prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from .base import Summarizer
from .. import settings

DEFAULT_WINDOW = 128_000
HARD_LIMIT = 0.95  # Codex's effective_context_window_percent


@dataclass(frozen=True)
class Compaction:
    threshold: int | None = None  # prompt tokens; defaults to ratio × context window
    growth: int | None = None  # or: tokens added since the window began (Codex's BodyAfterPrefix)
    ratio: float = 0.9  # Codex's auto_compact_token_limit: 90% of the window
    retain_user_tokens: int = 20_000  # COMPACT_USER_MESSAGE_MAX_TOKENS in Codex
    min_gain: float = 0.0  # skip unless at least this share of the threshold is freed
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
    def from_settings(cls) -> Compaction:
        threshold, growth = settings.value("compact_threshold"), settings.value("compact_growth")
        return cls(
            threshold=int(threshold) if threshold else None,
            growth=int(growth) if growth else None,
            ratio=float(settings.value("compact_ratio", "0.9")),
            retain_user_tokens=int(settings.value("retain_user_tokens", "20000")),
            min_gain=float(settings.value("compact_min_gain", "0")),
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

    def due(self, request: Request) -> tuple[bool, int]:
        """Whether the request reaches the threshold (or the window's hard limit), and the budget."""

        if self.growth:
            budget, over = self.growth, request.base is not None and request.tokens - request.base >= self.growth
        else:
            budget = self.limit(request.window)
            over = request.tokens >= budget
        return over or (request.window is not None and request.tokens >= HARD_LIMIT * request.window), budget

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        over, budget = self.due(request)
        if not request.force and not over:
            return None
        cut = summary_cut(request)
        if cut is None:
            return None
        if not request.force and cut < len(request.current):  # (a turn starts: the history before it decides)
            before = request.tokens - sum(map(item_tokens, request.current[cut:]))
            if not self.due(replace(request, tokens=before))[0]:
                return None
        items = request.current[:cut]
        users = self._recent_users(items)
        freed = sum(map(item_tokens, items)) - sum(map(item_tokens, users))
        if not request.force and freed < self.min_gain * budget:
            return None
        summary = Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\n{summarizer.summarize(cut, SUMMARIZATION_PROMPT)}")
        return Context((*users, summary, *request.current[cut:]))

    def _recent_users(self, items: tuple[Item, ...]) -> list[Item]:
        """The newest user messages within budget; the oldest one kept may be truncated. Counted and
        cut as Codex does (`approx_token_count`: a token for 4 bytes), whatever counter an
        evaluation holds the window with."""

        with counting_with(None):
            return self._within(items)

    def _within(self, items: tuple[Item, ...]) -> list[Item]:
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


def summary_cut(request: Request) -> int | None:
    """Everything before a pending user turn, or the whole conversation mid-turn."""

    items = request.current
    last_agent = max((i for i, item in enumerate(items) if item.kind in AGENT_KINDS), default=None)
    if last_agent is None:
        return None
    pending_turn = any(item.kind is Kind.USER for item in items[last_agent + 1 :])
    cut = last_agent + 1 if pending_turn else len(items)
    return cut if cut in request.boundaries else None
