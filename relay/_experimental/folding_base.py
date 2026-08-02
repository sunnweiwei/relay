from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Sequence

from ..strategies.base import BaseStrategy, PreparedInput
from .shared import (
    CODEX_COMPACTION_PROMPT,
    compact_threshold,
    model,
    output_text,
    over_threshold,
    pending_call_boundaries,
    protected_prefix,
    summary_item,
)


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


def latest_step_start(items: Sequence[dict[str, Any]], protected: int) -> int:
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
        for index in pending_call_boundaries(items[: call_index + 1])
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
        threshold = compact_threshold(request, self.compact_threshold)
        if not over_threshold(responses, request, active, threshold):
            return PreparedInput(deepcopy(active))
        folded = self.compact(responses, request, active)
        return PreparedInput(folded, compacted=folded != active)

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        protected = len(protected_prefix(active))
        if len(active) - protected < (1 if self.mode == "rollback" else 2):
            return deepcopy(active)
        if self.mode == "rollback":
            rule = (
                "Choose the start of a completed active suffix. Replace every item from "
                "start_index through the end with one return summary."
            )
            fold_end = len(active)
        elif self.mode == "agent_fold":
            fold_end = latest_step_start(active, protected)
            rule = (
                "Choose a suffix of previous items ending immediately before the final item. "
                "The final active step is the new observation and must remain verbatim."
            )
        else:
            raise ValueError("mode must be 'rollback' or 'agent_fold'")
        candidates = [
            index
            for index in pending_call_boundaries(active[:fold_end])
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
            model=model(request, self.manager_model),
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
        decision = json.loads(output_text(response))
        start = int(decision["start_index"])
        if start not in candidates:
            raise RuntimeError("management model selected an invalid fold boundary")
        replacement = summary_item(str(decision["summary"]))
        if self.mode == "rollback":
            folded = [*active[:start], replacement]
        else:
            folded = [*active[:start], replacement, *active[fold_end:]]
        return deepcopy(folded)
