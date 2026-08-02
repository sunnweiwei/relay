from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..strategies.base import BaseStrategy, PreparedInput
from .shared import compact_threshold, over_threshold, pending_call_boundaries, protected_prefix


@dataclass
class SlidingWindow(BaseStrategy):
    """Keep a protected instruction prefix and a tool-safe suffix of input items."""

    max_items: int = 64
    compact_threshold: int = 120_000
    name: str = field(default="sliding_window", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = compact_threshold(request, self.compact_threshold)
        if not over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        window = self.compact(responses, request, active)
        return PreparedInput(window, compacted=window != active)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.max_items < 1:
            raise ValueError("max_items must be positive")
        protected = len(protected_prefix(active))
        target = max(protected, len(active) - self.max_items)
        candidates = [
            index
            for index in pending_call_boundaries(active)
            if protected <= index <= target
        ]
        start = candidates[-1] if candidates else protected
        return [*protected_prefix(active), *deepcopy(active[start:])]
