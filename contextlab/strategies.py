from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Protocol, Sequence


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


@dataclass(frozen=True)
class PreparedInput:
    input: list[dict[str, Any]]
    overrides: dict[str, Any] = field(default_factory=dict)
    compacted: bool = False


class ContextStrategy(Protocol):
    name: str

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput: ...

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...

    def finish(
        self,
        responses: Any,
        request: dict[str, Any],
        sent: list[dict[str, Any]],
        output: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class BaseStrategy:
    name = "full_history"

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        return PreparedInput(deepcopy(active))

    def finish(
        self,
        responses: Any,
        request: dict[str, Any],
        sent: list[dict[str, Any]],
        output: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [*sent, *output]

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return deepcopy(active)


class FullHistory(BaseStrategy):
    pass


class OpenAITruncation(BaseStrategy):
    """Use the Responses API's built-in automatic truncation."""

    name = "openai_truncation"

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        return PreparedInput(deepcopy(active), {"truncation": "auto"})


@dataclass
class NativeCompaction(BaseStrategy):
    """Explicit comparison: Responses API server-side opaque compaction."""

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
        return _official_compact(responses, request, active)


def _count_kwargs(request: dict[str, Any], active: list[dict[str, Any]]) -> dict[str, Any]:
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


def _over_threshold(
    responses: Any,
    request: dict[str, Any],
    active: list[dict[str, Any]],
    threshold: int,
) -> bool:
    counted = responses.input_tokens.count(**_count_kwargs(request, active))
    return int(counted.input_tokens) >= threshold


def _compact_threshold(request: dict[str, Any], default: int) -> int:
    """Honor the official `context_management` request shape locally."""

    for item in request.get("context_management") or ():
        if item.get("type") == "compaction" and "compact_threshold" in item:
            return int(item["compact_threshold"])
    return default


def _model(request: dict[str, Any], manager_model: str | None) -> str:
    selected = manager_model or request.get("model")
    if not isinstance(selected, str) or not selected:
        raise ValueError("a model is required for model-backed context management")
    return selected


def _output_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in getattr(response, "output", ()):  # SDK models or test doubles
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


def _summary_item(summary: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": f"{CODEX_SUMMARY_PREFIX}\n{summary}",
    }


def _protected_prefix(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    end = 0
    while end < len(items) and items[end].get("role") in {"system", "developer"}:
        end += 1
    return deepcopy(list(items[:end]))


def _summary_call(
    responses: Any,
    request: dict[str, Any],
    items: list[dict[str, Any]],
    manager_model: str | None,
    prompt: str = CODEX_COMPACTION_PROMPT,
) -> str:
    call: dict[str, Any] = {
        "model": _model(request, manager_model),
        "input": [*deepcopy(items), {"role": "user", "content": prompt}],
        "store": False,
    }
    if isinstance(request.get("instructions"), str):
        call["instructions"] = request["instructions"]
    return _output_text(responses.create(**call))


@dataclass
class StandaloneCompaction(BaseStrategy):
    """Explicit, stateless use of the official `/responses/compact` endpoint."""

    compact_threshold: int = 120_000
    name: str = field(default="standalone_compaction", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = _compact_threshold(request, self.compact_threshold)
        if not _over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        return PreparedInput(
            self.compact(responses, request, active), compacted=True
        )

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _official_compact(responses, request, active)


def _official_compact(
    responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": _model(request, None),
        "input": deepcopy(active),
    }
    for key in ("instructions", "prompt_cache_key", "prompt_cache_options", "service_tier"):
        if key in request:
            kwargs[key] = deepcopy(request[key])
    compacted = responses.compact(**kwargs)
    return [_as_dict(item) for item in compacted.output]


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    return item.model_dump(mode="json", exclude_none=True)


@dataclass
class ThresholdCompaction(BaseStrategy):
    """Threshold-triggered summary compaction using Codex's checkpoint prompt."""

    compact_threshold: int = 120_000
    manager_model: str | None = None
    keep_last_user_messages: int = 1
    name: str = field(default="threshold_compaction", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = _compact_threshold(request, self.compact_threshold)
        if not _over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        return PreparedInput(
            self.compact(responses, request, active), compacted=True
        )

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        summary = _summary_call(responses, request, active, self.manager_model)
        users = [
            deepcopy(item)
            for item in active
            if item.get("type", "message") == "message" and item.get("role") == "user"
        ]
        recent = users[-self.keep_last_user_messages :] if self.keep_last_user_messages else []
        return [*_protected_prefix(active), *recent, _summary_item(summary)]


# Backward-compatible descriptive alias; both names select our implementation,
# never the provider's `/responses/compact` endpoint.
CodexPromptCompaction = ThresholdCompaction


FOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "start_index": {"type": "integer"},
        "summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["start_index", "summary", "reason"],
    "additionalProperties": False,
}


def _pending_call_boundaries(items: Sequence[dict[str, Any]]) -> list[int]:
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


@dataclass
class SlidingWindow(BaseStrategy):
    """Keep a protected instruction prefix and a tool-safe suffix of input items."""

    max_items: int = 64
    compact_threshold: int = 120_000
    name: str = field(default="sliding_window", init=False)

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = _compact_threshold(request, self.compact_threshold)
        if not _over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        window = self.compact(responses, request, active)
        return PreparedInput(window, compacted=window != active)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.max_items < 1:
            raise ValueError("max_items must be positive")
        protected = len(_protected_prefix(active))
        target = max(protected, len(active) - self.max_items)
        candidates = [
            index
            for index in _pending_call_boundaries(active)
            if protected <= index <= target
        ]
        start = candidates[-1] if candidates else protected
        return [*_protected_prefix(active), *deepcopy(active[start:])]


def _latest_step_start(items: Sequence[dict[str, Any]], protected: int) -> int:
    """Keep a final tool observation together with the call that produced it."""

    final = items[-1]
    if final.get("type") != "function_call_output" or not isinstance(
        final.get("call_id"), str
    ):
        return len(items) - 1
    call_id = final["call_id"]
    call_index = next(
        (
            index
            for index in range(len(items) - 2, protected - 1, -1)
            if items[index].get("type") == "function_call"
            and items[index].get("call_id") == call_id
        ),
        len(items) - 1,
    )
    safe = [
        index
        for index in _pending_call_boundaries(items[: call_index + 1])
        if index <= call_index
    ]
    return max(protected, safe[-1] if safe else protected)


@dataclass
class ModelFold(BaseStrategy):
    mode: str = "rollback"
    compact_threshold: int = 120_000
    manager_model: str | None = None

    @property
    def name(self) -> str:
        return "rollback_folding" if self.mode == "rollback" else "agent_fold"

    def prepare(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> PreparedInput:
        threshold = _compact_threshold(request, self.compact_threshold)
        if not _over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        folded = self.compact(responses, request, active)
        return PreparedInput(folded, compacted=folded != active)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        protected = len(_protected_prefix(active))
        if len(active) - protected < (1 if self.mode == "rollback" else 2):
            return deepcopy(active)
        if self.mode == "rollback":
            rule = (
                "Choose the start of a completed active suffix. Replace every item from "
                "start_index through the end with one return summary."
            )
            fold_end = len(active)
        elif self.mode == "agent_fold":
            fold_end = _latest_step_start(active, protected)
            rule = (
                "Choose a suffix of previous items ending immediately before the final item. "
                "The final active step is the new observation and must remain verbatim."
            )
        else:
            raise ValueError("mode must be 'rollback' or 'agent_fold'")
        candidates = [
            index
            for index in _pending_call_boundaries(active[:fold_end])
            if protected <= index < fold_end
        ]
        if not candidates:
            return deepcopy(active)
        prompt = (
            f"{CODEX_COMPACTION_PROMPT}\n\n{rule}\n"
            f"start_index must be one of {candidates}. "
            "The summary must contain everything needed to continue the coding task."
        )
        listing = json.dumps(
            [{"index": index, "item": item} for index, item in enumerate(active)],
            ensure_ascii=False,
        )
        response = responses.create(
            model=_model(request, self.manager_model),
            store=False,
            instructions=prompt,
            input=[{"role": "user", "content": listing}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "context_fold",
                    "strict": True,
                    "schema": {
                        **FOLD_SCHEMA,
                        "properties": {
                            **FOLD_SCHEMA["properties"],
                            "start_index": {"type": "integer", "enum": candidates},
                        },
                    },
                }
            },
        )
        decision = json.loads(_output_text(response))
        start = int(decision["start_index"])
        if start not in candidates:
            raise RuntimeError("management model selected an invalid fold boundary")
        replacement = _summary_item(str(decision["summary"]))
        if self.mode == "rollback":
            folded = [*active[:start], replacement]
        else:
            folded = [*active[:start], replacement, *active[fold_end:]]
        return deepcopy(folded)


class RollbackFolding(ModelFold):
    def __init__(
        self, compact_threshold: int = 120_000, manager_model: str | None = None
    ) -> None:
        super().__init__("rollback", compact_threshold, manager_model)


class AgentFold(ModelFold):
    def __init__(
        self, compact_threshold: int = 120_000, manager_model: str | None = None
    ) -> None:
        super().__init__("agent_fold", compact_threshold, manager_model)


@dataclass
class RollingMemory(BaseStrategy):
    """Update a bounded handoff memory after every completed task-model response."""

    manager_model: str | None = None
    name: str = field(default="rolling_memory", init=False)

    def finish(
        self,
        responses: Any,
        request: dict[str, Any],
        sent: list[dict[str, Any]],
        output: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combined = [*sent, *output]
        return self.compact(responses, request, combined)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        summary = _summary_call(responses, request, active, self.manager_model)
        return [*_protected_prefix(active), _summary_item(summary)]
