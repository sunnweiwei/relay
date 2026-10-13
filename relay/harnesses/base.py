"""Harness profiles: what Relay knows about an agent beyond its wire protocol.

A strategy sees only the conversation. Everything the harness itself writes into the request
(instructions, environment, reminders, modes) is the profile's business: which items are
injected (`refine`), what makes two requests the same conversation (`identity`), what the
harness's state is now (`state`), and where that state goes after a compaction (`place`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ..core.ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind
from ..install import Setting, read
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
    summaries: tuple[str, ...] = ()  # how the summaries of its own compaction start (Codex's: every harness)

    def matches(self, headers: Mapping[str, str]) -> bool:
        return False

    def codec(self, codec: Codec) -> Codec:
        """The codec for this harness's requests: the protocol's own, or a variant of it where the
        harness uses the protocol in a way that needs one."""

        return codec

    def refine(self, item: Item) -> Item:
        """Reclassify user-role items the harness wrote itself (context, summaries)."""

        if item.kind is Kind.USER and self.visible(item.text).startswith((SUMMARY_PREFIX, *self.summaries)):
            return replace(item, kind=Kind.SUMMARY)
        if item.kind is Kind.USER and item.text.strip() and not self.visible(item.text):
            return replace(item, kind=Kind.CONTEXT)
        return item

    def visible(self, text: str) -> str:
        """`text` without the blocks the harness injected into it."""

        for pattern in self.injected:
            text = pattern.sub("", text)
        return text.strip()

    def blocks(self, text: str) -> list[str]:
        """The blocks the harness injected into `text`, in order."""

        return [block for _, block in sorted((m.start(), m.group(0)) for p in self.injected for m in p.finditer(text))]

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
        """The harness's state at the end of `items`: by default what it wrote before the model's
        first action (its system and context items, and the blocks it put into user messages, as
        context of their own), with each piece of state the profile names (`state_key`) at its
        latest version, and named state that first appeared later."""

        view = [self.refine(codec.classify(wire)) for wire in items]
        first = next((i for i, item in enumerate(view) if item.kind in AGENT_KINDS), len(view))
        pieces = []
        for i, item in enumerate(view):
            if item.kind in CONTEXT_KINDS:
                pieces.append(replace(item, ref=i))
            elif item.kind is Kind.USER:  # (sent as a message of the block alone)
                pieces += [Item(Kind.CONTEXT, block, ref=i) for block in self.blocks(item.text)]
        keys = {n: key for n, piece in enumerate(pieces) if (key := self.state_key(piece))}
        latest = {key: n for n, key in sorted(keys.items())}
        chosen: list[int] = []
        for n, piece in enumerate(pieces):
            if piece.ref < first and (pick := latest.get(keys.get(n, ""), n)) not in chosen:
                chosen.append(pick)
        chosen += [n for n in latest.values() if n not in chosen]
        return State(tuple(pieces[n] for n in chosen))

    def place(self, head: tuple[Item, ...], state: tuple[Item, ...]) -> tuple[Item, ...]:
        """Put the harness's state into a compacted head (ending with its summary), the way Codex
        (0.162) does after compacting, mid-turn and at a turn start alike: system items first, the
        context just above the last real user message (or the summary, which stays last). The user
        messages kept are what the user wrote, as Codex keeps them: what the harness injected into
        them is state, or was summarized."""

        head = tuple(kept for item in head if (kept := self._typed(item)) is not None)
        system = tuple(item for item in state if item.kind is Kind.SYSTEM)
        context = tuple(item for item in state if item.kind is not Kind.SYSTEM)
        last = lambda kind: next((n for n in range(len(head) - 1, -1, -1) if head[n].kind is kind), None)
        at = next(n for n in (last(Kind.USER), last(Kind.SUMMARY), len(head)) if n is not None)
        return (*system, *head[:at], *context, *head[at:])

    def _typed(self, item: Item) -> Item | None:
        """A user message as the user wrote it (None: the harness wrote all of it)."""

        if item.kind is not Kind.USER or (text := self.visible(item.text)) == item.text.strip():
            return item
        return replace(item, text=text) if text or item.media else None

    def window(self, model: str | None) -> int | None:
        """The model's context window as the harness itself knows it (its model catalog), if it
        does: Relay compacts where the harness would."""

        return None

    def starter(self, command: str) -> list[Setting]:
        """Settings that have the harness run `command` before a session's first model request
        (`relay setup`: it makes sure Relay runs), where the harness runs such commands unasked."""

        return []

    def settings(self) -> list[Setting]:
        """Config-file settings that point the harness at Relay (`relay install`)."""

        raise ValueError(f"Relay cannot install itself into the {self.name} harness yet")

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        """Command line and environment that run the harness through Relay."""

        raise ValueError(f"Relay does not know how to launch the {self.name} harness")


def command_hook(path: Path, event: str, command: str, timeout: int = 30) -> Setting:
    """A command hook in Claude Code's format (`hooks.<event>` in a JSON settings file), after the
    user's own; uninstalling puts the user's back as they were."""

    hooks = read(path, ("hooks", event)) or []
    return Setting(path, ("hooks", event), [*hooks, {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}])


PLUGINS = Path(__file__).resolve().parent.parent / "integrations" / "plugins"


def plugin_file(path: Path, source: str, command: str) -> Setting:
    """A plugin of Relay's (`relay/integrations/plugins/<source>`) written where the harness loads
    its plugins, with `command` in it; uninstalling removes it."""

    return Setting(path, (), (PLUGINS / source).read_text(encoding="utf-8").replace("__COMMAND__", json.dumps(command)))
