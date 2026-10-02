"""The strategy interface: protocol-neutral decisions about what the model sees."""

from __future__ import annotations

from typing import Any, Protocol

from ..core.ir import Rewrite, View


class Summarizer(Protocol):
    def summarize(self, cut: int, prompt: str) -> str:
        """Ask the task's own upstream to answer `prompt` after `view.items[:cut]`."""


class Strategy(Protocol):
    name: str

    def fingerprint(self) -> dict[str, Any]:
        """Configuration that, when changed, must not reuse previously stored state."""

    def plan(self, view: View, summarizer: Summarizer) -> Rewrite | None:
        """Return how to rewrite the view, or None to forward it unchanged."""
