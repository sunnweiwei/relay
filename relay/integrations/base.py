"""What every integration path provides; the proxy path; summaries a hook supplies."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..core.ir import Item, Kind
from ..harnesses import Harness
from ..install import Setting


@dataclass(frozen=True)
class Installation:
    settings: list[Setting]  # what `relay install` writes into the harness's own config
    notes: tuple[str, ...] = ()  # where the path cannot follow the strategy exactly


class Integration(Protocol):
    """One way to run a strategy inside a harness: it shows the strategy the conversation before
    every request, makes its rewrite take effect, and makes the summaries it asks for."""

    name: str  # "proxy", or "hook" for a harness's native hook

    def installation(self, relay_url: str) -> Installation: ...


@dataclass(frozen=True)
class Proxy:
    """Every request through Relay: the harness's model endpoint points at it, and where the
    harness can be extended, its side reports to Relay what its requests do not show
    (`report`: the settings that add it, for Relay at a URL)."""

    harness: Harness
    report: Callable[[str], Installation] | None = None
    name = "proxy"

    def installation(self, relay_url: str) -> Installation:
        side = self.report(relay_url) if self.report else Installation([])
        return Installation([*self.harness.settings(), *side.settings], side.notes)


class SummaryNeeded(Exception):
    """The strategy asked for a summary the hook has not made yet."""

    def __init__(self, key: str, cut: int, prompt: str) -> None:
        super().__init__(f"summary {key} needed")
        self.key, self.cut, self.prompt = key, cut, prompt


@dataclass
class Supplied:
    """Summaries a hook made with the harness's own model, by key. Asking for any other stops the
    plan (`SummaryNeeded`); the hook makes it and asks again, and the plan, which depends only
    on the conversation, runs again to the same point."""

    summaries: dict[str, str] = field(default_factory=dict)

    def summarize(self, cut: int, prompt: str) -> str:
        key = hashlib.sha256(f"{cut}\0{prompt}".encode()).hexdigest()[:16]
        if key not in self.summaries:
            raise SummaryNeeded(key, cut, prompt)
        return self.summaries[key]

    def complete(self, items: Sequence[Item]) -> str:
        raise NotImplementedError("a hook can only continue the harness's own conversation; this strategy needs the proxy")


LABELS = {Kind.USER: "User", Kind.SUMMARY: "Summary", Kind.ASSISTANT: "Assistant",
          Kind.REASONING: "Assistant, thinking", Kind.TOOL_CALL: "Tool call", Kind.TOOL_RESULT: "Tool result"}


def render(items: list[Item]) -> str:
    """Conversation items as text, for a model that cannot see them as messages."""

    return "\n\n".join(f"[{LABELS.get(item.kind, item.kind.value)}]: {item.text}" for item in items if item.text)
