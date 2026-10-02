from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import BaseStrategy, GeneratedCheckpoint, PreparedInput
from .compact import (
    Compact,
    _protected_prefix,
    _safe_boundaries,
    _summarize_chunk,
    _summary_input,
    _summary_item,
)

_STATE_VERSION = 1
_ARTIFACT_KIND = "checkpoint_tree"


@dataclass
class Checkpoint(BaseStrategy):
    """Checkpoint chunks eagerly, but replace their source text only under pressure."""

    checkpoint_threshold: int = 30_000
    context_threshold: int = 120_000
    name: str = field(default="checkpoint", init=False)

    def __post_init__(self) -> None:
        if self.checkpoint_threshold <= 0:
            raise ValueError("checkpoint_threshold must be positive")
        if self.context_threshold <= self.checkpoint_threshold:
            raise ValueError(
                "context_threshold must be greater than checkpoint_threshold"
            )

    @classmethod
    def from_env(cls) -> Checkpoint:
        return cls(
            checkpoint_threshold=int(
                os.getenv("RELAY_CHECKPOINT_THRESHOLD", "30000")
            ),
            context_threshold=int(os.getenv("RELAY_CONTEXT_THRESHOLD", "120000")),
        )

    def cache_scope(self) -> dict[str, int]:
        return {
            "checkpoint_threshold": self.checkpoint_threshold,
            "context_threshold": self.context_threshold,
        }

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        if checkpoint is None:
            return deepcopy(trajectory)
        chunks = self._chunks(checkpoint, trajectory)
        return self._materialize_chunks(trajectory, chunks)

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        chunks = self._chunks(checkpoint, trajectory) if checkpoint else []
        changed = False
        generated: list[GeneratedCheckpoint] = []
        tail_start = (
            int(chunks[-1]["end"])
            if chunks
            else len(_protected_prefix(trajectory))
        )

        while tail_start < len(trajectory):
            tail = trajectory[tail_start:]
            if self._tokens(responses, request, tail) < self.checkpoint_threshold:
                break
            boundary = self._checkpoint_boundary(responses, request, tail)
            end = tail_start + boundary
            chunks.append(
                {
                    "start": tail_start,
                    "end": end,
                    "summary": self._summarize(
                        responses, request, trajectory[tail_start:end]
                    ),
                    "level": 0,
                    "evicted": False,
                }
            )
            tail_start = end
            changed = True
            generated.append(
                GeneratedCheckpoint(
                    covered_items=end,
                    artifact=self._artifact(chunks),
                )
            )

        active = self._materialize_chunks(trajectory, chunks)
        while self._tokens(responses, request, active) >= self.context_threshold:
            raw_index = next(
                (
                    index
                    for index, chunk in enumerate(chunks)
                    if not chunk["evicted"]
                ),
                None,
            )
            if raw_index is None:
                raise ValueError(
                    "checkpoint hierarchy cannot reduce the context below "
                    "context_threshold"
                )
            chunks[raw_index]["evicted"] = True
            changed = True
            chunks = self._merge_ready(responses, request, chunks)
            active = self._materialize_chunks(trajectory, chunks)

        current = (
            GeneratedCheckpoint(
                covered_items=len(trajectory),
                artifact=self._artifact(chunks),
            )
            if changed
            else None
        )
        return PreparedInput(
            input=active,
            compacted=active != trajectory,
            checkpoints=tuple(generated),
            checkpoint=current,
        )

    def compact(
        self, responses: Any, request: dict[str, Any], active: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return Compact(compact_threshold=self.context_threshold).compact(
            responses, request, active
        )

    def _chunks(
        self,
        checkpoint: GeneratedCheckpoint,
        trajectory: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        artifact = checkpoint.artifact
        if artifact.get("version") != _STATE_VERSION or artifact.get(
            "kind"
        ) != _ARTIFACT_KIND:
            raise ValueError("invalid checkpoint-tree artifact")
        checkpoint_threshold = artifact.get("checkpoint_threshold")
        context_threshold = artifact.get("context_threshold")
        if (
            checkpoint_threshold != self.checkpoint_threshold
            or context_threshold != self.context_threshold
        ):
            raise ValueError(
                "checkpoint strategy thresholds changed during a trajectory"
            )
        if checkpoint.covered_items > len(trajectory):
            raise ValueError("checkpoint artifact exceeds the trajectory")
        values = artifact.get("chunks")
        if not isinstance(values, list):
            raise TypeError("invalid checkpoint-tree chunks")

        chunks: list[dict[str, Any]] = []
        previous_end = len(_protected_prefix(trajectory))
        for value in values:
            if not isinstance(value, dict):
                raise TypeError("invalid checkpoint-tree chunk")
            start = value.get("start")
            end = value.get("end")
            summary = value.get("summary")
            level = value.get("level")
            evicted = value.get("evicted")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < previous_end
                or end <= start
                or end > checkpoint.covered_items
                or not isinstance(summary, str)
                or not isinstance(level, int)
                or level < 0
                or not isinstance(evicted, bool)
            ):
                raise ValueError("invalid checkpoint-tree chunk")
            chunks.append(deepcopy(value))
            previous_end = end
        return chunks

    def _artifact(self, chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "kind": _ARTIFACT_KIND,
            "checkpoint_threshold": self.checkpoint_threshold,
            "context_threshold": self.context_threshold,
            "chunks": deepcopy(list(chunks)),
        }

    @staticmethod
    def _materialize_chunks(
        trajectory: list[dict[str, Any]], chunks: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        cursor = 0
        for chunk in chunks:
            start = int(chunk["start"])
            end = int(chunk["end"])
            active.extend(deepcopy(trajectory[cursor:start]))
            if chunk["evicted"]:
                active.append(_summary_item(str(chunk["summary"])))
            else:
                active.extend(deepcopy(trajectory[start:end]))
            cursor = end
        active.extend(deepcopy(trajectory[cursor:]))
        return active

    def _checkpoint_boundary(
        self,
        responses: Any,
        request: dict[str, Any],
        tail: list[dict[str, Any]],
    ) -> int:
        boundaries = [boundary for boundary in _safe_boundaries(tail) if boundary > 0]
        if not boundaries:
            raise ValueError("the trajectory ends with an incomplete tool call")
        for boundary in boundaries:
            if (
                self._tokens(responses, request, tail[:boundary])
                >= self.checkpoint_threshold
            ):
                return boundary
        return boundaries[-1]

    def _merge_ready(
        self,
        responses: Any,
        request: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        while True:
            candidate = self._merge_candidate(responses, request, chunks)
            if candidate is None:
                return chunks
            start, end = candidate
            source = [
                _summary_item(str(chunk["summary"]))
                for chunk in chunks[start:end]
            ]
            merged = {
                "start": chunks[start]["start"],
                "end": chunks[end - 1]["end"],
                "summary": self._summarize(responses, request, source),
                "level": int(chunks[start]["level"]) + 1,
                "evicted": True,
            }
            chunks[start:end] = [merged]

    def _merge_candidate(
        self,
        responses: Any,
        request: dict[str, Any],
        chunks: Sequence[dict[str, Any]],
    ) -> tuple[int, int] | None:
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            if not chunk["evicted"]:
                index += 1
                continue
            level = chunk["level"]
            run_end = index
            while (
                run_end < len(chunks)
                and chunks[run_end]["evicted"]
                and chunks[run_end]["level"] == level
            ):
                run_end += 1
            for end in range(index + 2, run_end + 1):
                summaries = [
                    _summary_item(str(value["summary"]))
                    for value in chunks[index:end]
                ]
                if (
                    self._tokens(responses, request, summaries)
                    >= self.checkpoint_threshold
                ):
                    return index, end
            index = run_end
        return None

    def _summarize(
        self,
        responses: Any,
        request: dict[str, Any],
        items: Sequence[dict[str, Any]],
    ) -> str:
        return _summarize_chunk(
            responses,
            request,
            _summary_input(None, items),
        )

    @staticmethod
    def _tokens(
        responses: Any,
        request: dict[str, Any],
        items: Sequence[dict[str, Any]],
    ) -> int:
        counted = responses.input_tokens.count(
            model=request["model"],
            input=deepcopy(list(items)),
        )
        return int(counted.input_tokens)
