"""Harness profiles: what Relay knows about an agent beyond its wire protocol.

A strategy sees only the conversation. Everything the harness itself writes into the request
(instructions, environment, reminders, modes) is the profile's business: which items are
injected (`refine`), what makes two requests the same conversation (`identity`), what the
harness's state is now (`state`), where that state goes after a compaction (`place`, or all of
the layout around the summary: `compose`), and how the harness compacts by itself (`native`).
What the harness's side reports of its session (`relay.core.local`) is found by `session`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import SUMMARY_PREFIX
from ..protocols.base import Codec, WireItem

# Claude Code and the harnesses modelled on it (Kimi Code, CodeBuddy, OpenCode, DeepSeek
# Harness) inject context as user messages made of these blocks, at times with attributes.
REMINDER = re.compile(r"<system-reminder\b[^>]*>.*?</system-reminder>", re.S)
# A size no session reaches. `relay install` turns each harness's own auto-compaction off (Relay
# compacts instead); where the harness has no switch, its trigger size is set to this.
NEVER = 10**9


@dataclass(frozen=True)
class State:
    """What the harness has told the model about its state, as it stands now: request items
    kept by `ref`, or items Relay writes (`wire`). SYSTEM items go first, the rest form one
    context block."""

    items: tuple[Item, ...] = ()


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
        a snapshot); the default `state` keeps only the latest item of each. None: not state."""

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

    def compacting(self, codec: Codec, items: list[WireItem]) -> bool:
        """Whether the request is the harness compacting its own history (its own prompt, or a
        server-side trigger). It gets the stored compaction like any request, as only items before
        the first changed one can be covered, but never a new one."""

        return False

    def volatile(self, item: Item) -> bool:
        """Whether a trailing item is regenerated on every request rather than appended."""

        return False

    def state(self, codec: Codec, items: list[WireItem]) -> State:
        """The harness's state at the end of `items`: by default the system and context items it
        wrote before the model's first action, with each piece of state the profile names
        (`state_key`) at its latest version, and named state that first appeared later."""

        view = [self.refine(codec.classify(wire)) for wire in items]
        first = next((i for i, item in enumerate(view) if item.kind in AGENT_KINDS), len(view))
        keys = {i: key for i, item in enumerate(view) if item.kind in CONTEXT_KINDS and (key := self.state_key(item))}
        latest = {key: i for i, key in sorted(keys.items())}
        chosen: list[int] = []
        for i in range(first):
            if view[i].kind in CONTEXT_KINDS and (pick := latest.get(keys.get(i, ""), i)) not in chosen:
                chosen.append(pick)
        chosen += [i for i in latest.values() if i not in chosen]
        return State(tuple(replace(view[i], ref=i) for i in chosen))

    def place(self, head: tuple[Item, ...], state: tuple[Item, ...], mid_turn: bool) -> tuple[Item, ...]:
        """Put the harness's state into a compacted head (ending with its summary), the way Codex
        does after compacting: system items first; mid-turn the context sits just above the last
        real user message (or the summary, which stays last), and at a turn start it follows the
        head, ahead of the new turn."""

        system, context = self.split_state(state)
        if not mid_turn:
            return (*system, *head, *context)
        last = lambda kind: next((n for n in range(len(head) - 1, -1, -1) if head[n].kind is kind), None)
        at = next(n for n in (last(Kind.USER), last(Kind.SUMMARY), len(head)) if n is not None)
        return (*system, *head[:at], *context, *head[at:])

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """A compacted request around what follows its summary (`tail`, the harness's own items
        there included): the items before it (the head, ending with the summary, with the
        harness's state placed), the tail (its items may get new text) and the items after it.
        `request` is every item the harness sent."""

        return codec.arrange(self.place(head, state, mid_turn), mid_turn), tail, ()

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native | None:
        """How the harness compacts its own history (None: as Codex does), for the conversation of
        `request` (every item the harness sent)."""

        return None

    @staticmethod
    def split_state(state: tuple[Item, ...]) -> tuple[tuple[Item, ...], tuple[Item, ...]]:
        """The harness's state as its system items (first in any request) and the rest."""

        return (tuple(item for item in state if item.kind is Kind.SYSTEM),
                tuple(item for item in state if item.kind is not Kind.SYSTEM))

    def said(self, codec: Codec, kind: Kind, text: str) -> Item:
        """A message of the harness's own (a user or assistant message it writes after compacting):
        context, as what it writes is."""

        return Item(Kind.CONTEXT, text, wire=json.dumps(codec.write(Item(kind, text))))

    def session(self, headers: Mapping[str, str], body: Mapping) -> tuple[str, str] | None:
        """The session a request belongs to, as the harness's side reports it, and the request's
        first user message (which of the session's conversations it continues)."""

        return None

    def settings(self) -> list[Setting]:
        """Config-file settings that point the harness at Relay (`relay install`)."""

        raise ValueError(f"Relay cannot install itself into the {self.name} harness yet")

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        """Command line and environment that run the harness through Relay."""

        raise ValueError(f"Relay does not know how to launch the {self.name} harness")
