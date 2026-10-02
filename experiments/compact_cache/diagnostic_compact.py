"""Experiment-only Compact variant with an independent summary input budget."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from relay.strategies.base import GeneratedCheckpoint
from relay.strategies.compact import (
    Compact, _artifact, _compacted_input, _summarize,
)

from experiments.compact_cache.budget import MAX_INPUT_PER_REQUEST


class DiagnosticCompact(Compact):
    """Keep the normal trigger and output policy; decouple summary chunk size."""

    def __init__(self, *, compact_threshold: int = 120_000,
                 max_summary_input: int = MAX_INPUT_PER_REQUEST) -> None:
        super().__init__(compact_threshold=compact_threshold)
        self.max_summary_input = max_summary_input

    def _compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
        threshold: int,
    ) -> tuple[list[dict[str, Any]], tuple[GeneratedCheckpoint, ...]]:
        summaries = _summarize(responses, request, active, self.max_summary_input)
        checkpoints = tuple(
            GeneratedCheckpoint(
                covered_items=end,
                artifact=_artifact(_compacted_input(active[:end], summary)),
            )
            for end, summary in summaries
        )
        return deepcopy(checkpoints[-1].artifact["input"]), checkpoints
