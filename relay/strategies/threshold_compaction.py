from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, PreparedInput
from .shared import (
    compact_threshold,
    over_threshold,
    protected_prefix,
    summary_call,
    summary_item,
)


@dataclass
class ThresholdCompaction(BaseStrategy):
    """Threshold-triggered summary compaction using Codex's checkpoint prompt."""

    compact_threshold: int = 120_000
    manager_model: str | None = None
    keep_last_user_messages: int = 1
    name: str = field(default="threshold_compaction", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = compact_threshold(request, self.compact_threshold)
        if not over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        return PreparedInput(self.compact(responses, request, active), compacted=True)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        summary = summary_call(responses, request, active, self.manager_model)
        users = [
            deepcopy(item)
            for item in active
            if item.get("type", "message") == "message" and item.get("role") == "user"
        ]
        recent = users[-self.keep_last_user_messages :] if self.keep_last_user_messages else []
        return [*protected_prefix(active), *recent, summary_item(summary)]


# Backward-compatible descriptive alias; both names select Relay's implementation.
CodexPromptCompaction = ThresholdCompaction
