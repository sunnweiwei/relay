"""Codex's local context compaction (codex-rs/core/src/compact.rs) as a Relay strategy.

When the prompt reaches the threshold (90% of the context window, or a growth budget
since the window began; 95% of the window always triggers), the conversation is
summarized with Codex's prompt and replaced like Codex's replacement history: the newest
user messages within `retain_user_tokens` (newest first; the oldest one kept may be
truncated) and the summary last. Mid-turn (the request ends with tool results) all of it
is summarized; at the start of a turn the summary covers everything before the new turn,
which stays verbatim. The harness's own context is not the strategy's concern: the harness
profile puts its current state into the result (Codex: above the last user message
mid-turn, after the summary at a turn start). A compaction that would free less than
`min_gain` of the budget is skipped, so a threshold set too close to the fixed prompt
overhead cannot trigger a summary on every request.

A harness that compacts differently says how (`Request.native`): its own prompt, what it keeps
verbatim (a function of the request), whether it compacts only as a turn starts, and its summary
message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..core.conversation import pending_cut, recent_users
from ..core.ir import Context, Item, Kind, Native, Request
from ..core.tokens import approx_tokens
from ..prompts import SUMMARIZATION_PROMPT, SUMMARY_PREFIX
from .base import Summarizer

CODEX = Native(SUMMARIZATION_PROMPT, f"{SUMMARY_PREFIX}\n")
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

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if self.growth:
            budget, over = self.growth, request.base is not None and request.tokens - request.base >= self.growth
        else:
            budget = self.limit(request.window)
            over = request.tokens >= budget
        over = over or (request.window is not None and request.tokens >= HARD_LIMIT * request.window)
        if not request.force and not over:
            return None
        native = request.native or CODEX
        hard = request.window is not None and request.tokens >= HARD_LIMIT * request.window
        if native.at_turns and not request.force and not hard and pending_cut(request) == len(request.current):
            return None  # mid-turn: the harness would go on until a turn starts
        if native.keep is None:
            cut = pending_cut(request)
            users = None if cut is None else recent_users(request.current[:cut], self.retain_user_tokens)
        elif kept := native.keep(request):
            users, cut = list(kept[0]), kept[1]
        else:
            cut = None
        if cut is None:
            return None
        items = request.current[:cut]
        freed = sum(approx_tokens(i.text) for i in items) - sum(approx_tokens(i.text) for i in users)
        if not request.force and freed < self.min_gain * budget:
            return None
        summary = Item(Kind.SUMMARY, native.message(summarizer.summarize(cut, native.prompt)))
        return Context((*users, summary, *request.current[cut:]))
