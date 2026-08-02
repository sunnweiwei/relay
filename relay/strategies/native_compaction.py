from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, PreparedInput
from .shared import official_compact


@dataclass
class NativeCompaction(BaseStrategy):
    """Responses API server-side opaque compaction."""

    compact_threshold: int = 120_000
    name: str = field(default="native_compaction", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        management = list(deepcopy(request.get("context_management") or []))
        management = [item for item in management if item.get("type") != "compaction"]
        management.append(
            {"type": "compaction", "compact_threshold": self.compact_threshold}
        )
        return PreparedInput(deepcopy(active), {"context_management": management})

    def finish(
        self,
        responses: Any,
        request: dict[str, Any],
        sent: list[dict[str, Any]],
        output: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combined = [*sent, *output]
        latest = next(
            (
                index
                for index in range(len(combined) - 1, -1, -1)
                if combined[index].get("type") == "compaction"
            ),
            None,
        )
        return combined if latest is None else combined[latest:]

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return official_compact(responses, request, active)
