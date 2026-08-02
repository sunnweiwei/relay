from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import _is_context_window_error, _output_text, _protected_prefix
from .compact import _safe_boundaries as safe_boundaries

# MEM1 reference loop:
# https://github.com/MIT-MI/MEM1/blob/2609aef4e7c46d8d0c0f06b9312bc4b4abe04b9d/
# Mem1/inference/data_pipelines.py
ROLLING_MEMORY_PROMPT = """Update the recurrent working memory for a coding agent.

The input contains the previous working memory, when one exists, followed by newly
completed trajectory items. Produce a replacement memory that is concise but
sufficient for the agent to continue.

Preserve:
- The user's goals, constraints, and preferences
- Decisions made and their rationale
- Repository state, relevant files, symbols, commands, and tool results
- Work completed, unresolved problems, and concrete next steps
- Errors, failed approaches, and facts needed to avoid repeating work

Discard redundant narration and stale detail. Resolve newer facts over contradicted
older facts. Output only the updated working memory, with no preamble or commentary."""

ROLLING_MEMORY_PREFIX = "Current recurrent working memory:\n"
_STATE_VERSION = 1
_ARTIFACT_KIND = "rolling_memory"


@dataclass
class RollingMemory(BaseStrategy):
    """Maintain bounded recurrent memory while keeping the newest segment raw."""

    manager_model: str | None = None
    max_memory_output_tokens: int = 4_000
    update_input_tokens: int = 120_000
    name: str = field(default="rolling_memory", init=False)

    def __post_init__(self) -> None:
        if self.max_memory_output_tokens <= 0:
            raise ValueError("max_memory_output_tokens must be positive")
        if self.update_input_tokens <= 0:
            raise ValueError("update_input_tokens must be positive")

    @classmethod
    def from_env(cls) -> RollingMemory:
        return cls(
            manager_model=os.getenv("RELAY_MEMORY_MODEL") or None,
            max_memory_output_tokens=int(
                os.getenv("RELAY_MEMORY_MAX_OUTPUT_TOKENS", "4000")
            ),
            update_input_tokens=int(
                os.getenv("RELAY_MEMORY_UPDATE_INPUT_TOKENS", "120000")
            ),
        )

    def cache_scope(self) -> dict[str, Any]:
        return {
            "manager_model": self.manager_model,
            "max_memory_output_tokens": self.max_memory_output_tokens,
            "update_input_tokens": self.update_input_tokens,
            "prompt_version": 1,
        }

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is None:
            return deepcopy(trajectory)
        protected = len(_protected_prefix(trajectory))
        memory, covered = self._state(checkpoint, trajectory, protected)
        return [
            *_protected_prefix(trajectory),
            _memory_item(memory),
            *deepcopy(trajectory[covered:]),
        ]

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        protected_items = _protected_prefix(trajectory)
        protected = len(protected_items)
        boundaries = _completed_boundaries(trajectory, protected)
        target = boundaries[-2] if len(boundaries) > 1 else protected

        if checkpoint is None:
            memory: str | None = None
            covered = protected
        else:
            memory, covered = self._state(checkpoint, trajectory, protected)
            if covered > target:
                raise ValueError(
                    "rolling memory checkpoint exceeds the completed history"
                )
            if covered not in boundaries:
                raise ValueError("rolling memory checkpoint splits a tool transaction")

        generated, memory = self._roll(
            responses,
            request,
            trajectory,
            boundaries,
            memory,
            covered,
            target,
            protected,
        )
        active = [
            *protected_items,
            *([] if memory is None else [_memory_item(memory)]),
            *deepcopy(trajectory[target:]),
        ]
        return PreparedInput(
            active,
            compacted=active != trajectory,
            checkpoints=tuple(generated),
            checkpoint=generated[-1] if generated else None,
        )

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        protected_items = _protected_prefix(active)
        protected = len(protected_items)
        boundaries = _completed_boundaries(active, protected)
        if protected == len(active):
            return protected_items
        _, memory = self._roll(
            responses,
            request,
            active,
            boundaries,
            None,
            protected,
            len(active),
            protected,
        )
        assert memory is not None
        return [*protected_items, _memory_item(memory)]

    def _state(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
        protected: int,
    ) -> tuple[str, int]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid rolling memory checkpoint artifact")
        if artifact.get("protected_items") != protected:
            raise ValueError("rolling memory protected prefix changed")
        memory = artifact.get("memory")
        if not isinstance(memory, str) or not memory.strip():
            raise ValueError("rolling memory checkpoint contains no memory")
        if checkpoint.covered_items < protected or checkpoint.covered_items > len(
            trajectory
        ):
            raise ValueError("rolling memory checkpoint exceeds the trajectory")
        return memory, checkpoint.covered_items

    def _roll(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: Sequence[dict[str, Any]],
        boundaries: Sequence[int],
        memory: str | None,
        covered: int,
        target: int,
        protected: int,
    ) -> tuple[list[GeneratedCheckpoint], str | None]:
        generated: list[GeneratedCheckpoint] = []
        while covered < target:
            candidates = [
                boundary
                for boundary in boundaries
                if covered < boundary <= target
            ]
            candidate_index = self._largest_fitting_boundary(
                responses,
                request,
                memory,
                trajectory,
                covered,
                candidates,
            )
            if candidate_index is None:
                raise ValueError(
                    "one atomic trajectory segment exceeds the rolling memory "
                    "update window"
                )

            while candidate_index >= 0:
                end = candidates[candidate_index]
                try:
                    memory = self._update(
                        responses,
                        request,
                        memory,
                        trajectory[covered:end],
                    )
                    covered = end
                    generated.append(
                        GeneratedCheckpoint(
                            covered_items=covered,
                            artifact=_artifact(memory, protected),
                        )
                    )
                    break
                except Exception as exc:
                    if not _is_context_window_error(exc):
                        raise
                    candidate_index -= 1
            else:
                raise ValueError(
                    "one atomic trajectory segment exceeds the manager model "
                    "context window"
                )
        return generated, memory

    def _largest_fitting_boundary(
        self,
        responses: Any,
        request: dict[str, Any],
        memory: str | None,
        trajectory: Sequence[dict[str, Any]],
        covered: int,
        candidates: Sequence[int],
    ) -> int | None:
        low = 0
        high = len(candidates) - 1
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            update_input = _update_input(
                memory,
                trajectory[covered : candidates[middle]],
            )
            counted = responses.input_tokens.count(
                model=_manager_model(request, self.manager_model),
                input=update_input,
            )
            if int(counted.input_tokens) <= self.update_input_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _update(
        self,
        responses: Any,
        request: dict[str, Any],
        memory: str | None,
        delta: Sequence[dict[str, Any]],
    ) -> str:
        call: dict[str, Any] = {
            "model": _manager_model(request, self.manager_model),
            "input": _update_input(memory, delta),
            "max_output_tokens": self.max_memory_output_tokens,
            "store": False,
        }
        if "service_tier" in request:
            call["service_tier"] = request["service_tier"]
        return _output_text(responses.create(**call))


def _completed_boundaries(
    trajectory: Sequence[dict[str, Any]], protected: int
) -> list[int]:
    boundaries = [
        boundary for boundary in safe_boundaries(trajectory) if boundary >= protected
    ]
    if not boundaries or boundaries[0] != protected:
        boundaries.insert(0, protected)
    if boundaries[-1] != len(trajectory):
        raise ValueError("the trajectory ends with an incomplete tool call")
    return boundaries


def _manager_model(request: dict[str, Any], configured: str | None) -> str:
    model = configured or request.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("a model is required for rolling memory updates")
    return model


def _update_input(
    memory: str | None, delta: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    previous = [] if memory is None else [_memory_item(memory)]
    return [
        *previous,
        *deepcopy(list(delta)),
        {"role": "user", "content": ROLLING_MEMORY_PROMPT},
    ]


def _memory_item(memory: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": f"{ROLLING_MEMORY_PREFIX}{memory}",
    }


def _artifact(memory: str, protected: int) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "kind": _ARTIFACT_KIND,
        "protected_items": protected,
        "memory": memory,
    }
