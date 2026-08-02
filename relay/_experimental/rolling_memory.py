from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..strategies.base import BaseStrategy
from .shared import protected_prefix, summary_call, summary_item


@dataclass
class RollingMemory(BaseStrategy):
    """Update a bounded handoff memory after every task-model response."""

    manager_model: str | None = None
    name: str = field(default="rolling_memory", init=False)

    def finish(
        self,
        responses: Any,
        request: dict[str, Any],
        sent: list[dict[str, Any]],
        output: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.compact(responses, request, [*sent, *output])

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        summary = summary_call(responses, request, active, self.manager_model)
        return [*protected_prefix(active), summary_item(summary)]
