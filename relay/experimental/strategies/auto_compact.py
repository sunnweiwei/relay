from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ._manager import (
    completed_interactions,
    manager_json,
    manager_text,
    summary_message,
    task_prefix_end,
)
from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import _input_tokens

# AutoCompact currently publishes a runtime description and trained-policy
# results, but its project page does not link source code or released weights.
AUTOCOMPACT_PROJECT = "https://autocompact.github.io/"
_STATE_VERSION = 1
_ARTIFACT_KIND = "auto_compact"
AUTO_CONTEXT_SUMMARY = "# Auto Context Summary"

AUTO_COMPACT_MANAGER_PROMPT = """Decide whether AutoCompact should compact the coding-agent context now.

Choose keep while the current phase is unresolved, the agent is mid-derivation, or raw details are still needed. Choose compact at a useful phase transition when earlier exploration, failed attempts, or noisy tool output can be replaced by a clean working state without harming future work.

If action is compact, write a self-contained Auto Context Summary preserving the objective, conclusions and ruled-out hypotheses, relevant identifiers and paths, repository or environment changes, verification status, unresolved issues, and the next action. If action is keep, summary must be an empty string. Return only the requested JSON object."""

AUTO_COMPACT_FORCE_PROMPT = """The fallback context limit has been reached. Produce a self-contained # Auto Context Summary that preserves the objective, conclusions and ruled-out hypotheses, relevant identifiers and paths, repository or environment changes, verification status, unresolved issues, and the next action. Return only the summary body."""

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["keep", "compact"]},
        "summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "summary", "reason"],
    "additionalProperties": False,
}


@dataclass
class AutoCompact(BaseStrategy):
    """Task-aware summarize-and-continue compaction from AutoCompact.

    Because no official inference code or model is currently released, Relay
    uses a hidden manager to supply the learned compact/keep policy while
    preserving the project's published context transformation.
    """

    manager_model: str | None = None
    fallback_threshold: int = 120_000
    keep_recent_interactions: int = 2
    min_interactions: int = 1
    max_manager_output_tokens: int = 4_000
    name: str = field(default="auto_compact", init=False)

    def __post_init__(self) -> None:
        if self.fallback_threshold <= 0:
            raise ValueError("fallback_threshold must be positive")
        if self.keep_recent_interactions <= 0:
            raise ValueError("keep_recent_interactions must be positive")
        if self.min_interactions <= 0:
            raise ValueError("min_interactions must be positive")
        if self.max_manager_output_tokens <= 0:
            raise ValueError("max_manager_output_tokens must be positive")

    @classmethod
    def from_env(cls) -> AutoCompact:
        return cls(
            manager_model=os.getenv("RELAY_AUTO_COMPACT_MODEL") or None,
            fallback_threshold=int(
                os.getenv("RELAY_AUTO_COMPACT_FALLBACK_THRESHOLD", "120000")
            ),
            keep_recent_interactions=int(
                os.getenv("RELAY_AUTO_COMPACT_KEEP_RECENT", "2")
            ),
            min_interactions=int(
                os.getenv("RELAY_AUTO_COMPACT_MIN_INTERACTIONS", "1")
            ),
            max_manager_output_tokens=int(
                os.getenv("RELAY_AUTO_COMPACT_MAX_OUTPUT_TOKENS", "4000")
            ),
        )

    def cache_scope(self) -> dict[str, Any]:
        return {
            "project": AUTOCOMPACT_PROJECT,
            "manager_model": self.manager_model,
            "fallback_threshold": self.fallback_threshold,
            "keep_recent_interactions": self.keep_recent_interactions,
            "min_interactions": self.min_interactions,
            "max_manager_output_tokens": self.max_manager_output_tokens,
            "hidden_manager": True,
            "prompt_version": 1,
        }

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is None:
            return deepcopy(trajectory)
        _, active = self._state(checkpoint, trajectory)
        return [
            *active,
            *deepcopy(trajectory[checkpoint.covered_items :]),
        ]

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        if checkpoint is None:
            prefix_end = task_prefix_end(trajectory)
            prefix = deepcopy(trajectory[:prefix_end])
            active = deepcopy(trajectory)
        else:
            prefix, base = self._state(checkpoint, trajectory)
            active = [
                *base,
                *deepcopy(trajectory[checkpoint.covered_items :]),
            ]

        active_prefix_end = len(prefix)
        interactions, _ = completed_interactions(active, active_prefix_end)
        if len(interactions) < self.min_interactions:
            return PreparedInput(active, compacted=active != trajectory)

        forced = (
            _input_tokens(responses, request, active) >= self.fallback_threshold
        )
        summary: str | None = None
        if forced:
            summary = manager_text(
                responses,
                request,
                active,
                configured_model=self.manager_model,
                prompt=AUTO_COMPACT_FORCE_PROMPT,
                max_output_tokens=self.max_manager_output_tokens,
            )
        else:
            decision = manager_json(
                responses,
                request,
                active,
                configured_model=self.manager_model,
                prompt=AUTO_COMPACT_MANAGER_PROMPT,
                schema_name="relay_auto_compact_decision",
                schema=_DECISION_SCHEMA,
                max_output_tokens=self.max_manager_output_tokens,
            )
            action = decision.get("action")
            candidate = decision.get("summary")
            if action == "keep":
                if candidate not in {None, ""}:
                    raise ValueError("AutoCompact keep action must not include a summary")
                return PreparedInput(active, compacted=active != trajectory)
            if action != "compact":
                raise ValueError("AutoCompact returned an invalid action")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("AutoCompact compact action requires a summary")
            summary = candidate.strip()

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("AutoCompact returned an empty summary")
        raw_prefix_end = task_prefix_end(trajectory)
        recent = _recent_suffix(
            trajectory,
            raw_prefix_end,
            self.keep_recent_interactions,
        )
        compacted = [
            *prefix,
            summary_message(AUTO_CONTEXT_SUMMARY, summary),
            *recent,
        ]
        generated = GeneratedCheckpoint(
            covered_items=len(trajectory),
            artifact=_artifact(prefix, compacted),
        )
        return PreparedInput(
            compacted,
            compacted=True,
            checkpoint=generated,
        )

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prefix_end = task_prefix_end(active)
        prefix = deepcopy(active[:prefix_end])
        summary = manager_text(
            responses,
            request,
            active,
            configured_model=self.manager_model,
            prompt=AUTO_COMPACT_FORCE_PROMPT,
            max_output_tokens=self.max_manager_output_tokens,
        )
        return [
            *prefix,
            summary_message(AUTO_CONTEXT_SUMMARY, summary),
            *_recent_suffix(active, prefix_end, self.keep_recent_interactions),
        ]

    def _state(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid AutoCompact checkpoint artifact")
        if checkpoint.covered_items > len(trajectory):
            raise ValueError("AutoCompact checkpoint exceeds the trajectory")
        prefix = artifact.get("prefix")
        active = artifact.get("active")
        if not isinstance(prefix, list) or not isinstance(active, list):
            raise TypeError("invalid AutoCompact checkpoint state")
        if active[: len(prefix)] != prefix:
            raise ValueError("AutoCompact checkpoint lost its invariant task prefix")
        return deepcopy(prefix), deepcopy(active)


def _artifact(
    prefix: Sequence[dict[str, Any]], active: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "kind": _ARTIFACT_KIND,
        "prefix": deepcopy(list(prefix)),
        "active": deepcopy(list(active)),
    }


def _recent_suffix(
    items: Sequence[dict[str, Any]], prefix_end: int, count: int
) -> list[dict[str, Any]]:
    segments, trailing_start = completed_interactions(items, prefix_end)
    if segments:
        start = segments[max(0, len(segments) - count)][0]
    else:
        start = trailing_start
    return deepcopy(list(items[start:]))
