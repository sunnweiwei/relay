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
from .compact import Compact

# Official open-source reimplementation reviewed at this revision:
# https://github.com/sunnweiwei/FoldAgent
OFFICIAL_CONTEXT_FOLDING_COMMIT = "58a2d6964ecebe99940529eace50a0558901b8a5"
_STATE_VERSION = 1
_ARTIFACT_KIND = "context_folding"

CONTEXT_FOLDING_RETURN_PREFIX = (
    "Branch has finished its task, the returned message is:"
)

CONTEXT_FOLDING_MANAGER_PROMPT = """You manage a hidden Context Folding branch for a coding agent. The task model must not see branch-control actions.

The newest completed interaction is the candidate branch work. Choose exactly one action:
- none: no focused subtask has begun and no branch is open.
- open: a focused exploratory or trial-and-error subtask began in the newest interaction. Give its narrow objective. The branch bookmark will be placed immediately before that interaction.
- continue: the open subtask still needs work.
- return: the open subtask is complete. Return a concise, self-contained report preserving results, repository or environment state changes, key evidence, failures, dependencies, and pending work.

Do not open a branch for ordinary linear progress. Do not return while the subtask is unresolved. The action must respect the CURRENT BRANCH STATE below.

CURRENT BRANCH STATE:
{state}

Return only the requested JSON object."""

CONTEXT_FOLDING_LIMIT_PROMPT = """The hidden Context Folding branch reached its configured length limit. Finish the assigned subtask report now. Clearly state progress, results, repository or environment state changes, and pending work. Summarize only the branch subtask and return a self-contained report."""

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["none", "open", "continue", "return"],
        },
        "objective": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["action", "objective", "summary"],
    "additionalProperties": False,
}


@dataclass
class ContextFolding(BaseStrategy):
    """Transparent branch/return folding with hidden branch-control decisions.

    The branch state follows FoldAgent's branch -> sub-trajectory -> return
    transition. Relay moves only the branch trigger into a separate manager so
    the task agent's append-only trajectory contains no management tool calls.
    """

    manager_model: str | None = None
    max_manager_output_tokens: int = 2_000
    max_branch_steps: int = 200
    max_branch_tokens: int = 32_768
    max_branches: int = 10
    name: str = field(default="context_folding", init=False)

    def __post_init__(self) -> None:
        if self.max_manager_output_tokens <= 0:
            raise ValueError("max_manager_output_tokens must be positive")
        if self.max_branch_steps <= 0:
            raise ValueError("max_branch_steps must be positive")
        if self.max_branch_tokens <= 0:
            raise ValueError("max_branch_tokens must be positive")
        if self.max_branches <= 0:
            raise ValueError("max_branches must be positive")

    @classmethod
    def from_env(cls) -> ContextFolding:
        return cls(
            manager_model=os.getenv("RELAY_CONTEXT_FOLDING_MODEL") or None,
            max_manager_output_tokens=int(
                os.getenv("RELAY_CONTEXT_FOLDING_MAX_OUTPUT_TOKENS", "2000")
            ),
            max_branch_steps=int(
                os.getenv("RELAY_CONTEXT_FOLDING_MAX_BRANCH_STEPS", "200")
            ),
            max_branch_tokens=int(
                os.getenv("RELAY_CONTEXT_FOLDING_MAX_BRANCH_TOKENS", "32768")
            ),
            max_branches=int(
                os.getenv("RELAY_CONTEXT_FOLDING_MAX_BRANCHES", "10")
            ),
        )

    def cache_scope(self) -> dict[str, Any]:
        return {
            "official_commit": OFFICIAL_CONTEXT_FOLDING_COMMIT,
            "manager_model": self.manager_model,
            "max_manager_output_tokens": self.max_manager_output_tokens,
            "max_branch_steps": self.max_branch_steps,
            "max_branch_tokens": self.max_branch_tokens,
            "max_branches": self.max_branches,
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
        active, _, _ = self._state(checkpoint, trajectory)
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
            covered = task_prefix_end(trajectory)
            active = deepcopy(trajectory[:covered])
            branch: dict[str, Any] | None = None
            branch_count = 0
            needs_initial_checkpoint = covered > 0
        else:
            active, branch, branch_count = self._state(checkpoint, trajectory)
            covered = checkpoint.covered_items
            needs_initial_checkpoint = False

        segments, trailing_start = completed_interactions(trajectory, covered)
        generated: list[GeneratedCheckpoint] = []
        for start, end in segments:
            active.extend(deepcopy(trajectory[start:end]))
            active, branch, branch_count = self._observe(
                responses,
                request,
                active,
                branch,
                branch_count,
                segment_start=len(active) - (end - start),
            )
            covered = end
            generated.append(
                GeneratedCheckpoint(
                    covered_items=covered,
                    artifact=_artifact(active, branch, branch_count),
                )
            )

        if not generated and needs_initial_checkpoint:
            generated.append(
                GeneratedCheckpoint(
                    covered_items=covered,
                    artifact=_artifact(active, branch, branch_count),
                )
            )

        sent = [*active, *deepcopy(trajectory[trailing_start:])]
        current = generated[-1] if generated else None
        return PreparedInput(
            input=sent,
            compacted=sent != trajectory,
            checkpoints=tuple(generated),
            checkpoint=current,
        )

    def compact(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return Compact().compact(responses, request, active)

    def _observe(
        self,
        responses: Any,
        request: dict[str, Any],
        active: list[dict[str, Any]],
        branch: dict[str, Any] | None,
        branch_count: int,
        *,
        segment_start: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
        branch_over_limit = branch is not None and (
            int(branch["steps"]) + 1 >= self.max_branch_steps
            or _branch_tokens(responses, request, active, int(branch["anchor"]))
            >= self.max_branch_tokens
        )
        if branch_over_limit:
            summary = manager_text(
                responses,
                request,
                active,
                configured_model=self.manager_model,
                prompt=CONTEXT_FOLDING_LIMIT_PROMPT,
                max_output_tokens=self.max_manager_output_tokens,
            )
            return _return(active, branch, summary), None, branch_count

        if branch is None and branch_count >= self.max_branches:
            return active, None, branch_count

        state = (
            "none"
            if branch is None
            else f"open; objective={branch['objective']!r}; completed_steps={branch['steps']}"
        )
        allowed_actions = ["none", "open"] if branch is None else ["continue", "return"]
        schema = deepcopy(_DECISION_SCHEMA)
        schema["properties"]["action"]["enum"] = allowed_actions
        decision = manager_json(
            responses,
            request,
            active,
            configured_model=self.manager_model,
            prompt=CONTEXT_FOLDING_MANAGER_PROMPT.format(state=state),
            schema_name="relay_context_folding_decision",
            schema=schema,
            max_output_tokens=self.max_manager_output_tokens,
        )
        action = decision.get("action")
        objective = decision.get("objective")
        summary = decision.get("summary")

        if branch is None:
            if action == "none":
                return active, None, branch_count
            if action != "open":
                raise ValueError("Context Folding manager action is invalid without a branch")
            if not isinstance(objective, str) or not objective.strip():
                raise ValueError("Context Folding open action requires an objective")
            return (
                active,
                {
                    "anchor": segment_start,
                    "objective": objective.strip(),
                    "steps": 1,
                },
                branch_count + 1,
            )

        if action == "continue":
            updated = deepcopy(branch)
            updated["steps"] = int(updated["steps"]) + 1
            return active, updated, branch_count
        if action != "return":
            raise ValueError("Context Folding manager action is invalid for an open branch")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Context Folding return action requires a summary")
        return _return(active, branch, summary), None, branch_count

    def _state(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid Context Folding checkpoint artifact")
        if checkpoint.covered_items > len(trajectory):
            raise ValueError("Context Folding checkpoint exceeds the trajectory")
        active = artifact.get("active")
        branch = artifact.get("branch")
        branch_count = artifact.get("branch_count")
        if not isinstance(active, list):
            raise TypeError("invalid Context Folding active context")
        if branch is not None:
            if not isinstance(branch, dict):
                raise TypeError("invalid Context Folding branch state")
            anchor = branch.get("anchor")
            objective = branch.get("objective")
            steps = branch.get("steps")
            if (
                not isinstance(anchor, int)
                or not 0 <= anchor <= len(active)
                or not isinstance(objective, str)
                or not objective
                or not isinstance(steps, int)
                or steps <= 0
            ):
                raise ValueError("invalid Context Folding branch state")
        if not isinstance(branch_count, int) or branch_count < 0:
            raise ValueError("invalid Context Folding branch count")
        return deepcopy(active), deepcopy(branch), branch_count


def _artifact(
    active: Sequence[dict[str, Any]],
    branch: dict[str, Any] | None,
    branch_count: int,
) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "kind": _ARTIFACT_KIND,
        "active": deepcopy(list(active)),
        "branch": deepcopy(branch),
        "branch_count": branch_count,
    }


def _return(
    active: Sequence[dict[str, Any]], branch: dict[str, Any], summary: str
) -> list[dict[str, Any]]:
    anchor = int(branch["anchor"])
    return [
        *deepcopy(list(active[:anchor])),
        summary_message(CONTEXT_FOLDING_RETURN_PREFIX, summary),
    ]


def _branch_tokens(
    responses: Any,
    request: dict[str, Any],
    active: Sequence[dict[str, Any]],
    anchor: int,
) -> int:
    counted = responses.input_tokens.count(
        model=request.get("model"),
        input=deepcopy(list(active[anchor:])),
    )
    return int(counted.input_tokens)
