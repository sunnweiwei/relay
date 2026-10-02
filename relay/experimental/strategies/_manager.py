from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .compact import _output_text, _protected_prefix

_CALL_TYPES = {"function_call", "custom_tool_call"}
_CALL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


@dataclass(frozen=True)
class ManagerToolResult:
    value: dict[str, Any]
    trajectory: list[dict[str, Any]]


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


def manager_tool_json(
    responses: Any,
    request: dict[str, Any],
    initial_input: Sequence[dict[str, Any]],
    *,
    configured_model: str | None,
    tools: Sequence[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], Any],
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    max_steps: int,
    require_first_tool: bool = True,
) -> dict[str, Any]:
    return manager_tool_session(
        responses,
        request,
        initial_input,
        configured_model=configured_model,
        tools=tools,
        execute_tool=execute_tool,
        schema_name=schema_name,
        schema=schema,
        max_output_tokens=max_output_tokens,
        max_steps=max_steps,
        require_first_tool=require_first_tool,
    ).value


def manager_tool_session(
    responses: Any,
    request: dict[str, Any],
    initial_input: Sequence[dict[str, Any]],
    *,
    configured_model: str | None,
    tools: Sequence[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], Any],
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    max_steps: int,
    require_first_tool: bool = True,
    compact_threshold: int | None = None,
) -> ManagerToolResult:
    """Run a replayable private Responses tool session.

    The complete model output is replayed between manager calls so reasoning and
    function-call items keep their normal Responses semantics. If server-side
    compaction emits an encrypted compaction item, that item becomes the new
    private-session prefix exactly as in a normal stateless Responses loop.
    """

    if max_steps < 2:
        raise ValueError("manager tool loops require at least two steps")
    manager_input = deepcopy(list(initial_input))
    for step in range(max_steps):
        call: dict[str, Any] = {
            "model": manager_model(request, configured_model),
            "input": deepcopy(manager_input),
            "tools": deepcopy(list(tools)),
            "tool_choice": "required" if require_first_tool and step == 0 else "auto",
            "parallel_tool_calls": False,
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
        if compact_threshold is not None:
            call["context_management"] = [
                {"type": "compaction", "compact_threshold": compact_threshold}
            ]
        response = responses.create(**call)
        output = [_manager_item_dict(item) for item in getattr(response, "output", ())]
        manager_input = _append_manager_output(manager_input, output)
        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        if not function_calls:
            value = json.loads(_output_text(response))
            if not isinstance(value, dict):
                raise TypeError(f"{schema_name} manager returned a non-object")
            return ManagerToolResult(value=value, trajectory=manager_input)

        for item in function_calls:
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise TypeError("manager returned an invalid function call")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else None
                if not isinstance(parsed, dict):
                    raise TypeError("function arguments must be a JSON object")
                result = execute_tool(name, parsed)
            except (TypeError, ValueError) as exc:
                result = {"error": str(exc)}
            manager_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        result,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                }
            )
    raise RuntimeError("context manager exceeded its hidden tool-step limit")


def _append_manager_output(
    manager_input: list[dict[str, Any]], output: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_compaction = next(
        (
            index
            for index in range(len(output) - 1, -1, -1)
            if output[index].get("type") == "compaction"
        ),
        None,
    )
    if latest_compaction is not None:
        return deepcopy(list(output[latest_compaction:]))
    return [*manager_input, *deepcopy(list(output))]


def _manager_item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return deepcopy(dict(item))
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    raise TypeError("manager response output contains an unsupported item")


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
