"""Harness profiles: what Relay knows about an agent beyond its wire protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..core.ir import Item, Kind
from ..install import Setting
from ..prompts import SUMMARY_PREFIX
from ..protocols.base import Codec, WireItem

# Claude Code and the harnesses modelled on it (Kimi Code, CodeBuddy, OpenCode) inject
# context as user messages made of these blocks.
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


@dataclass(frozen=True)
class InitialContext:
    """The initial context a harness re-renders for a compacted history, as view items: kept
    request items (by `ref`) or items Relay writes (`wire`). SYSTEM items stay first, the rest
    form the context block. `current` is set when it also reflects the context updates that
    open a pending turn, which compaction then replaces."""

    items: tuple[Item, ...]
    current: bool = False


class Harness:
    """The generic profile; subclasses describe a specific agent harness."""

    name = "generic"

    def matches(self, headers: Mapping[str, str]) -> bool:
        return False

    def refine(self, item: Item) -> Item:
        """Reclassify user-role items the harness wrote itself (context, summaries)."""

        if item.kind is Kind.USER and item.text.startswith(SUMMARY_PREFIX):
            return replace(item, kind=Kind.SUMMARY)
        if item.kind is Kind.USER and item.text.strip() and not REMINDER.sub("", item.text).strip():
            return replace(item, kind=Kind.CONTEXT)
        return item

    def volatile(self, item: Item) -> bool:
        """Whether a trailing item is regenerated on every request rather than appended."""

        return False

    def initial_context(self, codec: Codec, items: list[WireItem]) -> InitialContext | None:
        """The initial context as the harness would re-render it now, or None to reuse the
        system and context items it wrote before the model's first action."""

        return None

    def settings(self) -> list[Setting]:
        """Config-file settings that point the harness at Relay (`relay install`)."""

        raise ValueError(f"Relay cannot install itself into the {self.name} harness yet")

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        """Command line and environment that run the harness through Relay."""

        raise ValueError(f"Relay does not know how to launch the {self.name} harness")
