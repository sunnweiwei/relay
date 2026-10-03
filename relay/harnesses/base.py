"""Harness profiles: what Relay knows about an agent beyond its wire protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind
from ..install import Setting
from ..prompts import SUMMARY_PREFIX
from ..protocols.base import Codec, WireItem

# Claude Code and the harnesses modelled on it (Kimi Code, CodeBuddy, OpenCode, DeepSeek
# Harness) inject context as user messages made of these blocks, at times with attributes.
REMINDER = re.compile(r"<system-reminder\b[^>]*>.*?</system-reminder>", re.S)


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
    injected: tuple[re.Pattern[str], ...] = (REMINDER,)  # blocks the harness writes into user messages

    def matches(self, headers: Mapping[str, str]) -> bool:
        return False

    def refine(self, item: Item) -> Item:
        """Reclassify user-role items the harness wrote itself (context, summaries)."""

        if item.kind is Kind.USER and item.text.startswith(SUMMARY_PREFIX):
            return replace(item, kind=Kind.SUMMARY)
        if item.kind is Kind.USER and item.text.strip() and not self.visible(item.text):
            return replace(item, kind=Kind.CONTEXT)
        return item

    def visible(self, text: str) -> str:
        """`text` without the blocks the harness injected into it."""

        for pattern in self.injected:
            text = pattern.sub("", text)
        return text.strip()

    def state_key(self, item: Item) -> str | None:
        """Which piece of the harness's state an injected context item restates (a date, a mode,
        a snapshot); compaction re-injects only the latest item of each. None: not state."""

        return None

    def identity(self, codec: Codec, items: list[WireItem]) -> list[bytes]:
        """What prefix matching compares. Before the model's first action harnesses re-render
        their instructions in place (OpenCode keeps AGENTS.md in its system message, Gemini CLI
        GEMINI.md in its first user message), so system and context items there compare by
        position and user messages without the injected blocks. The request's own items are
        still what is forwarded, so the model sees the current version."""

        keys = [codec.canonical(item) for item in items]
        for index, wire in enumerate(items):
            item = self.refine(codec.classify(wire))
            if item.kind in AGENT_KINDS:
                break
            if item.kind in CONTEXT_KINDS:
                keys[index] = b"\0" + item.kind.value.encode()
            elif item.kind is Kind.USER and (text := self.visible(item.text)) != item.text.strip():
                keys[index] = b"\0user\0" + text.encode() + (b"\0media" if item.media else b"")
        return keys

    def volatile(self, item: Item) -> bool:
        """Whether a trailing item is regenerated on every request rather than appended."""

        return False

    def initial_context(self, codec: Codec, items: list[WireItem]) -> InitialContext | None:
        """The initial context as the harness would re-render it now, or None to reuse the
        system and context items it wrote before the model's first action.

        By default, a profile that names state (`state_key`) gets the context it wrote before
        the model's first action with each piece of state at its latest version, plus state
        that first appeared later. A pending turn keeps its own context after the summary."""

        view = [self.refine(codec.classify(wire)) for wire in items]
        keys = {i: key for i, item in enumerate(view) if item.kind in CONTEXT_KINDS and (key := self.state_key(item))}
        agents = [i for i, item in enumerate(view) if item.kind in AGENT_KINDS]
        if not keys or not agents:
            return None
        pending = any(item.kind is Kind.USER for item in view[agents[-1] + 1 :])
        end = agents[-1] + 1 if pending else len(view)
        latest = {key: i for i, key in sorted(keys.items()) if i < end}
        chosen: list[int] = []
        for i in range(agents[0]):
            if view[i].kind in CONTEXT_KINDS and (pick := latest.get(keys.get(i, ""), i)) not in chosen:
                chosen.append(pick)
        chosen += [i for i in latest.values() if i not in chosen]
        return InitialContext(tuple(replace(view[i], ref=i) for i in chosen))

    def settings(self) -> list[Setting]:
        """Config-file settings that point the harness at Relay (`relay install`)."""

        raise ValueError(f"Relay cannot install itself into the {self.name} harness yet")

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        """Command line and environment that run the harness through Relay."""

        raise ValueError(f"Relay does not know how to launch the {self.name} harness")
