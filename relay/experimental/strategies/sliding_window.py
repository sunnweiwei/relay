from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import (
    _input_tokens,
    _protected_prefix,
    _request_threshold,
    _safe_boundaries,
)


@dataclass
class SlidingWindow(BaseStrategy):
    """Keep the longest tool-safe suffix that fits a bounded input window."""

    max_input_tokens: int = 120_000
    name: str = field(default="sliding_window", init=False)

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")

    @classmethod
    def from_env(cls) -> SlidingWindow:
        return cls(
            max_input_tokens=int(
                os.getenv("RELAY_SLIDING_WINDOW_TOKENS", "120000")
            )
        )

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is not None:
            raise ValueError("sliding_window does not use checkpoint artifacts")
        return deepcopy(trajectory)

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        active = self.materialize(trajectory, checkpoint)
        limit = _request_threshold(request, self.max_input_tokens)
        window = self._window(responses, request, active, limit)
        return PreparedInput(window, compacted=window != active)

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        limit = _request_threshold(request, self.max_input_tokens)
        return self._window(responses, request, active, limit)

    def cache_scope(self) -> dict[str, int]:
        return {"max_input_tokens": self.max_input_tokens}

    @staticmethod
    def _window(
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("sliding window token limit must be positive")
        if _input_tokens(responses, request, active) <= limit:
            return deepcopy(active)

        prefix = _protected_prefix(active)
        protected = len(prefix)
        if protected == len(active):
            raise ValueError("protected input prefix exceeds the sliding window")
        candidates = [
            boundary
            for boundary in _safe_boundaries(active)
            if protected <= boundary < len(active)
        ]
        if not candidates:
            raise ValueError("the trajectory ends with an incomplete tool call")

        start = _earliest_fitting_start(
            responses,
            request,
            active,
            prefix,
            candidates,
            limit,
        )
        if start is None:
            raise ValueError(
                "the latest atomic trajectory segment exceeds the sliding window"
            )
        return [*prefix, *deepcopy(active[start:])]


def _earliest_fitting_start(
    responses: Any,
    request: dict[str, Any],
    active: Sequence[dict[str, Any]],
    prefix: list[dict[str, Any]],
    candidates: Sequence[int],
    limit: int,
) -> int | None:
    low = 0
    high = len(candidates) - 1
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        start = candidates[middle]
        window = [*prefix, *deepcopy(list(active[start:]))]
        if _input_tokens(responses, request, window) <= limit:
            best = start
            high = middle - 1
        else:
            low = middle + 1
    return best
