"""Harness profiles: what Relay knows about an agent beyond its wire protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace

from ..core.ir import Item, Kind
from ..install import Setting
from ..prompts import SUMMARY_PREFIX

# Claude Code and the harnesses modelled on it (Kimi Code, CodeBuddy, OpenCode) inject
# context as user messages made of these blocks.
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


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

    def settings(self) -> list[Setting]:
        """Config-file settings that point the harness at Relay (`relay install`)."""

        raise ValueError(f"Relay cannot install itself into the {self.name} harness yet")

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        """Command line and environment that run the harness through Relay."""

        raise ValueError(f"Relay does not know how to launch the {self.name} harness")
