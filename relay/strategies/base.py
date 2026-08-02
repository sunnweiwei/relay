from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedCheckpoint:
    covered_items: int
    artifact: dict[str, Any]


@dataclass(frozen=True)
class PreparedInput:
    input: list[dict[str, Any]]
    overrides: dict[str, Any] = field(default_factory=dict)
    compacted: bool = False
    checkpoints: tuple[GeneratedCheckpoint, ...] = ()
    checkpoint: GeneratedCheckpoint | None = None


class ContextStrategy(Protocol):
    name: str

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]: ...

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
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

    def materialize(
        self,
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        return deepcopy(trajectory)

    def prepare(
        self,
        responses: Any,
        request: dict[str, Any],
        trajectory: list[dict[str, Any]],
        checkpoint: GeneratedCheckpoint | None = None,
    ) -> PreparedInput:
        return PreparedInput(deepcopy(trajectory))

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
