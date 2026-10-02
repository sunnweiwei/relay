"""Calibrate a live case without changing the Compact strategy's rules."""
from __future__ import annotations

from typing import Any

from relay.strategies.compact import (
    _input_tokens, _safe_boundaries, _summary_input, _summary_input_tokens,
)
from experiments.compact_cache.budget import MAX_SUMMARY_OUTPUT


def calibrate(responses: Any, first: dict[str, Any],
              second: dict[str, Any]) -> dict[str, int]:
    """Ensure an individual safe segment plus summary prompt fits the trigger."""
    first_tokens = _input_tokens(responses, first, first["input"])
    second_tokens = _input_tokens(responses, second, second["input"])
    boundaries = _safe_boundaries(second["input"])
    if not boundaries or boundaries[-1] != len(second["input"]):
        raise ValueError("second request ends with an incomplete tool transaction")
    atomic_counts = [
        _summary_input_tokens(
            responses, second,
            _summary_input(None if start == 0 else "x " * MAX_SUMMARY_OUTPUT,
                           second["input"][start:end]),
        )
        for start, end in zip(boundaries, boundaries[1:])
    ]
    largest_atomic = max(atomic_counts, default=0)
    threshold = max(first_tokens + 128, largest_atomic + 64)
    return {"first_input_tokens": first_tokens,
            "second_input_tokens": second_tokens,
            "largest_atomic_summary_tokens": largest_atomic,
            "compact_threshold": threshold}


def calibrate_trigger_only(responses: Any, first: dict[str, Any],
                           second: dict[str, Any]) -> dict[str, int]:
    """Put the trigger between two turns; summary size has its own budget."""
    first_tokens = _input_tokens(responses, first, first["input"])
    second_tokens = _input_tokens(responses, second, second["input"])
    if second_tokens <= first_tokens + 1:
        raise ValueError("second turn did not grow enough to set a trigger")
    return {"first_input_tokens": first_tokens,
            "second_input_tokens": second_tokens,
            "compact_threshold": min(first_tokens + 128, second_tokens - 1)}
