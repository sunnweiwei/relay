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

# Official AgentFold inference implementation reviewed at this revision:
# https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/AgentFold
OFFICIAL_AGENT_FOLD_COMMIT = "f72f75d8c3eb842f2bbbab096a12206ff66e270f"
_STATE_VERSION = 1
_ARTIFACT_KIND = "agent_fold"

AGENT_FOLD_MANAGER_PROMPT = """Perform the folding directive from AgentFold for the Latest Interaction in the context above.

The history is an ordered Multi-Scale State Summary followed by one full Latest Interaction. Fold a suffix that must end at Latest Interaction step {latest_step}. Choose:
- granular condensation: start at {latest_step}, preserving this interaction as a fine-grained state summary; or
- deep consolidation: start at one of {valid_starts}, fusing that chain of prior summaries and the Latest Interaction into a coarser summary when a subtask has completed.

The replacement must preserve the sequential logic and every fact, identifier, source, repository or environment state change, failure, and next dependency needed later. Use the official inference fields: compress_range and compress_text. compress_range must be exactly [start, {latest_step}]. Return only the requested JSON object."""

AGENT_FOLD_FORCE_PROMPT = """Compress the AgentFold workspace above into one self-contained state summary. Preserve the task objective, sequential conclusions, identifiers, evidence, repository or environment state, failures, and next actions. Return only the summary text."""

_DIRECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "compress_range": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "compress_text": {"type": "string"},
    },
    "required": ["compress_range", "compress_text"],
    "additionalProperties": False,
}


@dataclass
class AgentFold(BaseStrategy):
    """AgentFold's multi-scale summaries plus one raw Latest Interaction.

    The paper's fold-and-act response is split across a hidden manager call and
    the unchanged task-model call. The task model still receives C_t (old
    summaries plus the raw latest interaction); the fold updates C_{t+1}.
    """

    manager_model: str | None = None
    max_manager_output_tokens: int = 4_000
    name: str = field(default="agent_fold", init=False)

    def __post_init__(self) -> None:
        if self.max_manager_output_tokens <= 0:
            raise ValueError("max_manager_output_tokens must be positive")

    @classmethod
    def from_env(cls) -> AgentFold:
        return cls(
            manager_model=os.getenv("RELAY_AGENT_FOLD_MODEL") or None,
            max_manager_output_tokens=int(
                os.getenv("RELAY_AGENT_FOLD_MAX_OUTPUT_TOKENS", "4000")
            ),
        )

    def cache_scope(self) -> dict[str, Any]:
        return {
            "official_commit": OFFICIAL_AGENT_FOLD_COMMIT,
            "manager_model": self.manager_model,
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
        prefix, summaries, _ = self._state(checkpoint, trajectory)
        return [
            *_workspace(prefix, summaries),
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
            covered = task_prefix_end(trajectory)
            prefix = deepcopy(trajectory[:covered])
            summaries: list[dict[str, Any]] = []
            next_step = 1
            needs_initial_checkpoint = covered > 0
        else:
            prefix, summaries, next_step = self._state(checkpoint, trajectory)
            covered = checkpoint.covered_items
            needs_initial_checkpoint = False

        segments, trailing_start = completed_interactions(trajectory, covered)
        generated: list[GeneratedCheckpoint] = []
        sent_before_fold: list[dict[str, Any]] | None = None

        for start, end in segments:
            latest = deepcopy(trajectory[start:end])
            sent_before_fold = [*_workspace(prefix, summaries), *latest]
            directive = self._directive(
                responses,
                request,
                sent_before_fold,
                summaries,
                next_step,
            )
            summaries = _apply_directive(summaries, directive, next_step)
            next_step += 1
            covered = end
            generated.append(
                GeneratedCheckpoint(
                    covered_items=covered,
                    artifact=_artifact(prefix, summaries, next_step),
                )
            )

        if not generated and needs_initial_checkpoint:
            generated.append(
                GeneratedCheckpoint(
                    covered_items=covered,
                    artifact=_artifact(prefix, summaries, next_step),
                )
            )

        current_workspace = (
            sent_before_fold
            if sent_before_fold is not None
            else _workspace(prefix, summaries)
        )
        sent = [*current_workspace, *deepcopy(trajectory[trailing_start:])]
        return PreparedInput(
            input=sent,
            compacted=sent != trajectory,
            checkpoints=tuple(generated),
            checkpoint=generated[-1] if generated else None,
        )

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prefix_end = task_prefix_end(active)
        prefix = deepcopy(active[:prefix_end])
        if prefix_end == len(active):
            return prefix
        summary = manager_text(
            responses,
            request,
            active,
            configured_model=self.manager_model,
            prompt=AGENT_FOLD_FORCE_PROMPT,
            max_output_tokens=self.max_manager_output_tokens,
        )
        segments, _ = completed_interactions(active, prefix_end)
        end_step = max(1, len(segments))
        return _workspace(
            prefix,
            [{"start": 1, "end": end_step, "content": summary}],
        )

    def _directive(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
        summaries: Sequence[dict[str, Any]],
        latest_step: int,
    ) -> dict[str, Any]:
        valid_starts = [int(item["start"]) for item in summaries]
        valid_starts.append(latest_step)
        value = manager_json(
            responses,
            request,
            active,
            configured_model=self.manager_model,
            prompt=AGENT_FOLD_MANAGER_PROMPT.format(
                latest_step=latest_step,
                valid_starts=valid_starts,
            ),
            schema_name="relay_agent_fold_directive",
            schema=_DIRECTIVE_SCHEMA,
            max_output_tokens=self.max_manager_output_tokens,
        )
        compress_range = value.get("compress_range")
        compress_text = value.get("compress_text")
        if (
            not isinstance(compress_range, list)
            or len(compress_range) != 2
            or compress_range[0] not in valid_starts
            or compress_range[1] != latest_step
        ):
            raise ValueError("AgentFold returned an invalid compression range")
        if not isinstance(compress_text, str) or not compress_text.strip():
            raise ValueError("AgentFold returned an empty compression summary")
        return {
            "compress_range": [int(compress_range[0]), latest_step],
            "compress_text": compress_text.strip(),
        }

    def _state(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid AgentFold checkpoint artifact")
        if checkpoint.covered_items > len(trajectory):
            raise ValueError("AgentFold checkpoint exceeds the trajectory")
        prefix = artifact.get("prefix")
        summaries = artifact.get("summaries")
        next_step = artifact.get("next_step")
        if not isinstance(prefix, list) or not isinstance(summaries, list):
            raise TypeError("invalid AgentFold workspace")
        if not isinstance(next_step, int) or next_step <= 0:
            raise ValueError("invalid AgentFold next step")
        expected = 1
        normalized: list[dict[str, Any]] = []
        for item in summaries:
            if not isinstance(item, dict):
                raise TypeError("invalid AgentFold summary block")
            start = item.get("start")
            end = item.get("end")
            content = item.get("content")
            if (
                not isinstance(start, int)
                or start != expected
                or not isinstance(end, int)
                or end < start
                or end >= next_step
                or not isinstance(content, str)
                or not content
            ):
                raise ValueError("invalid AgentFold summary partition")
            normalized.append({"start": start, "end": end, "content": content})
            expected = end + 1
        if expected != next_step:
            raise ValueError("AgentFold summaries do not partition folded history")
        return deepcopy(prefix), normalized, next_step


def _artifact(
    prefix: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    next_step: int,
) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "kind": _ARTIFACT_KIND,
        "prefix": deepcopy(list(prefix)),
        "summaries": deepcopy(list(summaries)),
        "next_step": next_step,
    }


def _apply_directive(
    summaries: Sequence[dict[str, Any]],
    directive: dict[str, Any],
    latest_step: int,
) -> list[dict[str, Any]]:
    start = int(directive["compress_range"][0])
    kept = [deepcopy(item) for item in summaries if int(item["end"]) < start]
    kept.append(
        {
            "start": start,
            "end": latest_step,
            "content": directive["compress_text"],
        }
    )
    return kept


def _workspace(
    prefix: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not summaries:
        return deepcopy(list(prefix))
    return [
        *deepcopy(list(prefix)),
        summary_message(
            "### Multi-Scale State Summaries",
            _format_summaries(summaries),
        ),
    ]


def _format_summaries(summaries: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for item in summaries:
        start = int(item["start"])
        end = int(item["end"])
        if start == end:
            header = f"[Compressed Step {start}]"
        else:
            header = f"[Compressed Step {start} to {end}]"
        blocks.append(f"**{header}**\n{item['content']}")
    return "\n\n".join(blocks)
