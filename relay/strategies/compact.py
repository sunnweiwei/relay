from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from typing import Any, Sequence

from .base import BaseStrategy, PreparedInput


# openai/codex, codex-rs/prompts/templates/compact/prompt.md
# commit 2b5bdcf67547860f2e5c5a605009a70026796b2b
CODEX_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

CODEX_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its "
    "thinking process. You also have access to the state of the tools that were used by "
    "that language model. Use this to build on the work that has already been done and "
    "avoid duplicating work. Here is the summary produced by the other language model, "
    "use the information in this summary to assist with your own analysis:"
)

CODEX_RETAINED_USER_MESSAGE_TOKENS = 20_000
_APPROX_BYTES_PER_TOKEN = 4


@dataclass
class Compact(BaseStrategy):
    """Codex-compatible prompt compaction for append-only Responses loops."""

    compact_threshold: int = 120_000
    name: str = field(default="compact", init=False)

    @classmethod
    def from_env(cls) -> Compact:
        return cls(
            compact_threshold=int(os.getenv("RELAY_COMPACT_THRESHOLD", "120000"))
        )

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = _request_threshold(request, self.compact_threshold)
        if not _over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        return PreparedInput(
            self._compact(responses, request, active, threshold), compacted=True
        )

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        threshold = _request_threshold(request, self.compact_threshold)
        return self._compact(responses, request, active, threshold)

    def _compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
        threshold: int,
    ) -> list[dict[str, Any]]:
        summary = _summarize(responses, request, active, threshold)
        retained = _retain_user_messages(
            active, max_tokens=CODEX_RETAINED_USER_MESSAGE_TOKENS
        )
        return [*_protected_prefix(active), *retained, _summary_item(summary)]


def _request_threshold(request: dict[str, Any], default: int) -> int:
    for item in request.get("context_management") or ():
        if item.get("type") == "compaction" and "compact_threshold" in item:
            return int(item["compact_threshold"])
    return default


def _over_threshold(
    responses: Any,
    request: dict[str, Any],
    active: list[dict[str, Any]],
    threshold: int,
) -> bool:
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
    count_request = {
        "input": active,
        **{key: request[key] for key in allowed if key in request},
    }
    counted = responses.input_tokens.count(**count_request)
    return int(counted.input_tokens) >= threshold


def _summarize(
    responses: Any,
    request: dict[str, Any],
    active: list[dict[str, Any]],
    max_input_tokens: int,
) -> str:
    model = request.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("a model is required for compaction")
    if max_input_tokens <= 0:
        raise ValueError("compact_threshold must be positive")

    boundaries = _safe_boundaries(active)
    if not active:
        summary_input = _summary_input(None, [])
        if _summary_input_tokens(responses, request, summary_input) > max_input_tokens:
            raise ValueError("the compaction prompt exceeds compact_threshold")
        return _summarize_chunk(responses, request, summary_input)
    if not boundaries or boundaries[-1] != len(active):
        raise ValueError("the trajectory ends with an incomplete tool call")

    summary: str | None = None
    start = 0
    while start < len(active):
        candidates = [boundary for boundary in boundaries if boundary > start]
        candidate_index = _largest_fitting_boundary(
            responses,
            request,
            active,
            start,
            candidates,
            summary,
            max_input_tokens,
        )
        if candidate_index is None:
            raise ValueError(
                "one atomic trajectory segment exceeds compact_threshold; "
                "splitting a single item or tool transaction is not supported"
            )

        while candidate_index >= 0:
            end = candidates[candidate_index]
            summary_input = _summary_input(summary, active[start:end])
            try:
                summary = _summarize_chunk(responses, request, summary_input)
                start = end
                break
            except Exception as exc:
                if not _is_context_window_error(exc):
                    raise
                candidate_index -= 1
        else:
            raise ValueError(
                "one atomic trajectory segment exceeds the model context window"
            )

    assert summary is not None
    return summary


def _largest_fitting_boundary(
    responses: Any,
    request: dict[str, Any],
    active: list[dict[str, Any]],
    start: int,
    candidates: list[int],
    summary: str | None,
    max_input_tokens: int,
) -> int | None:
    low = 0
    high = len(candidates) - 1
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        summary_input = _summary_input(summary, active[start : candidates[middle]])
        tokens = _summary_input_tokens(responses, request, summary_input)
        if tokens <= max_input_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _summary_input(
    previous_summary: str | None, chunk: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    prefix = [] if previous_summary is None else [_summary_item(previous_summary)]
    return [
        *prefix,
        *deepcopy(list(chunk)),
        {"role": "user", "content": CODEX_COMPACTION_PROMPT},
    ]


def _summary_input_tokens(
    responses: Any, request: dict[str, Any], summary_input: list[dict[str, Any]]
) -> int:
    count_request = {
        "input": summary_input,
        **{
            key: deepcopy(request[key])
            for key in ("instructions", "model", "reasoning")
            if key in request
        },
    }
    return int(responses.input_tokens.count(**count_request).input_tokens)


def _summarize_chunk(
    responses: Any, request: dict[str, Any], summary_input: list[dict[str, Any]]
) -> str:
    call: dict[str, Any] = {
        "model": request["model"],
        "input": summary_input,
        "store": False,
    }
    for key in (
        "instructions",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "service_tier",
    ):
        if key in request:
            call[key] = deepcopy(request[key])
    return _output_text(responses.create(**call))


def _safe_boundaries(items: Sequence[dict[str, Any]]) -> list[int]:
    """Return boundaries that do not split a tool call from its output."""

    pending: set[str] = set()
    boundaries = [0]
    for index, item in enumerate(items):
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type in {"function_call", "custom_tool_call"} and isinstance(
            call_id, str
        ):
            pending.add(call_id)
        elif item_type in {
            "function_call_output",
            "custom_tool_call_output",
        } and isinstance(call_id, str):
            pending.discard(call_id)
        if not pending:
            boundaries.append(index + 1)
    return boundaries


def _output_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in getattr(response, "output", ()):
        content = (
            item.get("content", ())
            if isinstance(item, dict)
            else getattr(item, "content", ())
        )
        for part in content or ():
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise RuntimeError("compaction model returned no summary")
    return "\n".join(parts).strip()


def _is_context_window_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    message = str(error.get("message", exc) if isinstance(error, dict) else exc).lower()
    return code in {"context_length_exceeded", "context_window_exceeded"} or (
        getattr(exc, "status_code", None) == 400
        and ("context window" in message or "maximum context length" in message)
    )


def _protected_prefix(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    end = 0
    while end < len(items) and items[end].get("role") in {"system", "developer"}:
        end += 1
    return deepcopy(list(items[:end]))


def _message_text(item: dict[str, Any]) -> str | None:
    if item.get("type", "message") != "message" or item.get("role") != "user":
        return None
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        part.get("text")
        for part in content
        if isinstance(part, dict)
        and part.get("type") in {"input_text", "output_text"}
        and isinstance(part.get("text"), str)
    ]
    return "\n".join(parts) if parts else None


def _retain_user_messages(
    items: Sequence[dict[str, Any]], *, max_tokens: int
) -> list[dict[str, Any]]:
    messages = [
        text
        for item in items
        if (text := _message_text(item)) is not None
        and not text.startswith(f"{CODEX_SUMMARY_PREFIX}\n")
    ]
    selected: list[str] = []
    remaining = max(0, max_tokens)
    for text in reversed(messages):
        if remaining == 0:
            break
        tokens = _approx_token_count(text)
        if tokens <= remaining:
            selected.append(text)
            remaining -= tokens
        else:
            selected.append(_truncate_middle(text, remaining))
            break
    selected.reverse()
    return [
        {"type": "message", "role": "user", "content": text}
        for text in selected
    ]


def _approx_token_count(text: str) -> int:
    byte_count = len(text.encode("utf-8"))
    return (byte_count + _APPROX_BYTES_PER_TOKEN - 1) // _APPROX_BYTES_PER_TOKEN


def _truncate_middle(text: str, max_tokens: int) -> str:
    max_bytes = max(0, max_tokens) * _APPROX_BYTES_PER_TOKEN
    encoded_bytes = len(text.encode("utf-8"))
    if max_tokens > 0 and encoded_bytes <= max_bytes:
        return text

    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    left: list[str] = []
    used = 0
    for char in text:
        size = len(char.encode("utf-8"))
        if used + size > left_budget:
            break
        left.append(char)
        used += size

    right: list[str] = []
    used = 0
    for char in reversed(text):
        size = len(char.encode("utf-8"))
        if used + size > right_budget:
            break
        right.append(char)
        used += size
    right.reverse()

    removed = max(0, encoded_bytes - max_bytes)
    removed_tokens = (removed + _APPROX_BYTES_PER_TOKEN - 1) // _APPROX_BYTES_PER_TOKEN
    return f"{''.join(left)}…{removed_tokens} tokens truncated…{''.join(right)}"


def _summary_item(summary: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": f"{CODEX_SUMMARY_PREFIX}\n{summary}",
    }
