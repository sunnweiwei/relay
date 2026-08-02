from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, PreparedInput
from .shared import compact_threshold, official_compact, over_threshold


@dataclass
class StandaloneCompaction(BaseStrategy):
    """Stateless use of the official `/responses/compact` endpoint."""

    compact_threshold: int = 120_000
    name: str = field(default="standalone_compaction", init=False)

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
        return official_compact(responses, request, active)
