from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedCheckpoint:
    covered_items: int
    input: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedInput:
    input: list[dict[str, Any]]
    overrides: dict[str, Any] = field(default_factory=dict)
    compacted: bool = False
    checkpoints: tuple[GeneratedCheckpoint, ...] = ()


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
