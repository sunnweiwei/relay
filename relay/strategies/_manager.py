from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .compact import _output_text, _protected_prefix

_CALL_TYPES = {"function_call", "custom_tool_call"}
_CALL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


def manager_model(request: dict[str, Any], configured: str | None) -> str:
    value = configured or request.get("model")
    if not isinstance(value, str) or not value:
        raise ValueError("a model is required for context management")
    return value


def manager_json(
    responses: Any,
    request: dict[str, Any],
    active: Sequence[dict[str, Any]],
    *,
    configured_model: str | None,
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    call: dict[str, Any] = {
        "model": manager_model(request, configured_model),
        "input": [
            *deepcopy(list(active)),
            {"type": "message", "role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": deepcopy(schema),
            }
        },
    }
    if "service_tier" in request:
        call["service_tier"] = request["service_tier"]
    value = json.loads(_output_text(responses.create(**call)))
    if not isinstance(value, dict):
        raise TypeError(f"{schema_name} manager returned a non-object")
    return value


def manager_text(
    responses: Any,
    request: dict[str, Any],
    active: Sequence[dict[str, Any]],
    *,
    configured_model: str | None,
    prompt: str,
    max_output_tokens: int,
) -> str:
    call: dict[str, Any] = {
        "model": manager_model(request, configured_model),
        "input": [
            *deepcopy(list(active)),
            {"type": "message", "role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if "service_tier" in request:
        call["service_tier"] = request["service_tier"]
    return _output_text(responses.create(**call))


def task_prefix_end(items: Sequence[dict[str, Any]]) -> int:
    """Keep instructions and the initial user task as an invariant prefix."""

    end = len(_protected_prefix(items))
    while end < len(items):
        item = items[end]
        if item.get("type") == "compaction" or item.get("role") == "user":
            end += 1
            continue
        break
    return end


def completed_interactions(
    items: Sequence[dict[str, Any]], start: int
) -> tuple[list[tuple[int, int]], int]:
    """Split completed assistant-action/observation transactions.

    Reasoning and parallel tool calls stay attached to their action. A regular
    trailing user message is deliberately left pending for the next task turn.
    """

    if start < 0 or start > len(items):
        raise ValueError("interaction start is outside the trajectory")
    segments: list[tuple[int, int]] = []
    segment_start = start
    pending: set[str] = set()
    has_action = False

    for index in range(start, len(items)):
        item = items[index]
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type in _CALL_TYPES:
            has_action = True
            if isinstance(call_id, str):
                pending.add(call_id)
        elif item_type in _CALL_OUTPUT_TYPES:
            if isinstance(call_id, str):
                pending.discard(call_id)
        elif item_type == "message" and item.get("role") == "assistant":
            has_action = True

        if has_action and not pending:
            segments.append((segment_start, index + 1))
            segment_start = index + 1
            has_action = False

    return segments, segment_start


def summary_message(title: str, summary: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": f"{title}\n{summary.strip()}",
    }
