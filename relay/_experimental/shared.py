from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence


# openai/codex, codex-rs/prompts/templates/compact/prompt.md
# commit 2b5bdcf67547860f2e5c5a605009a70026796b2b
CODEX_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

# openai/codex, codex-rs/prompts/templates/compact/summary_prefix.md
CODEX_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its "
    "thinking process. You also have access to the state of the tools that were used by "
    "that language model. Use this to build on the work that has already been done and "
    "avoid duplicating work. Here is the summary produced by the other language model, "
    "use the information in this summary to assist with your own analysis:"
)


def count_kwargs(
    request: dict[str, Any], active: list[dict[str, Any]]
) -> dict[str, Any]:
    allowed = {
        "instructions",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "tool_choice",
        "tools",
        "truncation",
    }
    return {"input": active, **{key: request[key] for key in allowed if key in request}}


def over_threshold(
    responses: Any,
    request: dict[str, Any],
    active: list[dict[str, Any]],
    threshold: int,
) -> bool:
    counted = responses.input_tokens.count(**count_kwargs(request, active))
    return int(counted.input_tokens) >= threshold


def compact_threshold(request: dict[str, Any], default: int) -> int:
    """Honor the official `context_management` request shape locally."""

    for item in request.get("context_management") or ():
        if item.get("type") == "compaction" and "compact_threshold" in item:
            return int(item["compact_threshold"])
    return default


def model(request: dict[str, Any], manager_model: str | None) -> str:
    selected = manager_model or request.get("model")
    if not isinstance(selected, str) or not selected:
        raise ValueError("a model is required for model-backed context management")
    return selected


def output_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in getattr(response, "output", ()):
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", ())
        for part in content or ():
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise RuntimeError("management model returned no text")
    return "\n".join(parts).strip()


def summary_item(summary: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": f"{CODEX_SUMMARY_PREFIX}\n{summary}",
    }


def protected_prefix(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    end = 0
    while end < len(items) and items[end].get("role") in {"system", "developer"}:
        end += 1
    return deepcopy(list(items[:end]))


def summary_call(
    responses: Any,
    request: dict[str, Any],
    items: list[dict[str, Any]],
    manager_model: str | None,
    prompt: str = CODEX_COMPACTION_PROMPT,
) -> str:
    call: dict[str, Any] = {
        "model": model(request, manager_model),
        "input": [*deepcopy(items), {"role": "user", "content": prompt}],
        "store": False,
    }
    if isinstance(request.get("instructions"), str):
        call["instructions"] = request["instructions"]
    return output_text(responses.create(**call))


def pending_call_boundaries(items: Sequence[dict[str, Any]]) -> list[int]:
    """Indices where the preceding prefix has no unresolved function call."""

    pending: set[str] = set()
    boundaries = [0]
    for index, item in enumerate(items):
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type == "function_call" and isinstance(call_id, str):
            pending.add(call_id)
        elif item_type == "function_call_output" and isinstance(call_id, str):
            pending.discard(call_id)
        if not pending:
            boundaries.append(index + 1)
    return boundaries


def as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    return item.model_dump(mode="json", exclude_none=True)


def official_compact(
    responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model(request, None),
        "input": deepcopy(active),
    }
    for key in ("instructions", "prompt_cache_key", "prompt_cache_options", "service_tier"):
        if key in request:
            kwargs[key] = deepcopy(request[key])
    compacted = responses.compact(**kwargs)
    return [as_dict(item) for item in compacted.output]
