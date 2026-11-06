"""The truncation baseline: the conversation goes to the model as the harness sent it.

It is meant for bounded evaluations with Relay's overflow fallback (`overflow_fallback=truncate`).
When a request is over the window, its forced recovery changes nothing, so Relay's fallback drops
the oldest tool call groups until the request fits. Compared with a context-management strategy
under the same budget, it measures what the strategy adds over plain truncation. Without an input
budget it forwards every request unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.ir import Context, Request
from .base import Summarizer


@dataclass(frozen=True)
class Truncation:
    name: str = "truncation"

    @classmethod
    def from_settings(cls) -> Truncation:
        return cls()

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        return None
