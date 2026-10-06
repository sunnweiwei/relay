"""The strategy interface: protocol-neutral decisions about what the model sees."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..core.ir import Context, Item, Request


class Summarizer(Protocol):
    """The task's own model, through the harness's upstream and credentials."""

    def summarize(self, cut: int, prompt: str) -> str:
        """The model answering `prompt` after `request.current[:cut]`, sent as the harness sent it
        (its own instructions included, so the prompt cache holds)."""

    def complete(self, items: Sequence[Item]) -> str:
        """The model answering `items` alone: SYSTEM items as its instructions, then user and
        assistant messages; no tools."""


class Strategy(Protocol):
    name: str

    def fingerprint(self) -> dict[str, Any]:
        """Configuration that, when changed, must not reuse previously stored state."""

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        """The context the model sees for this request, or None: `request.current`, state kept."""
