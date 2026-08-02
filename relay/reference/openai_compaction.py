from __future__ import annotations

from copy import deepcopy
from typing import Any


def server_side_request(
    request: dict[str, Any], *, compact_threshold: int
) -> dict[str, Any]:
    """Return the official server-side compaction request shape for comparison."""

    forwarded = deepcopy(request)
    management = [
        item
        for item in forwarded.get("context_management") or []
        if item.get("type") != "compaction"
    ]
    management.append(
        {"type": "compaction", "compact_threshold": compact_threshold}
    )
    forwarded["context_management"] = management
    return forwarded


def standalone_compact(
    responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Call OpenAI's canonical `/responses/compact` endpoint for comparison."""

    kwargs: dict[str, Any] = {
        "model": request["model"],
        "input": deepcopy(active),
    }
    for key in (
        "instructions",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "service_tier",
    ):
        if key in request:
            kwargs[key] = deepcopy(request[key])
    response = responses.compact(**kwargs)
    return [
        deepcopy(item)
        if isinstance(item, dict)
        else item.model_dump(mode="json", exclude_none=True)
        for item in response.output
    ]
