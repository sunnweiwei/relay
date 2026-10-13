"""DeepSeek Harness (`dsh`); providers are plugin entries in the home-level cordis.patch.yml.

The DeepSeek route takes a `baseURL` (its DEEPSEEK_BASE_URL may only be exported, never
written to a file). Of the pi-ai routes (OpenAI, Google, Anthropic, gateways) only those the
user already declared are wrapped: a patch replaces a plugin's whole config, so adding a
pi-ai entry could hide routes configured in other layers.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting, yaml_keys
from ..protocols import MessageParts
from ..protocols.base import Codec, WireItem
from .base import NEVER, Harness, State

DEEPSEEK = ("[id=llm-deepseek]", "config", "baseURL")
ROUTES = ("[id=llm-pi-ai]", "config", "providers")
DEFAULTS = {  # pi-ai catalog endpoints, for routes without an explicit baseURL
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com",
}


# A runtime snapshot ("Current runtime context. This snapshot supersedes earlier ...") or their
# end ("Current runtime context: none. Earlier runtime-context snapshots no longer apply.").
RUNTIME = "Current runtime context"
# Instruction files after the baseline: one newly in scope, one changed, one removed (several may
# come in one message), and a whole new baseline.
CHANGED = re.compile(r"^(?:Additional instructions from|Updated instructions from|Instructions removed): (.+)$", re.M)
REBASED = "This complete workspace instruction baseline replaces all earlier workspace instruction baselines."
# What it tells the model in the user's role besides its state (like Codex's notifications): a
# sub-agent's message, how one ended and its closing message (where a message's blocks are items,
# what the sub-agent wrote is one of the conversation's); a tool call the model keeps repeating; a
# background job that finished; plan mode switched by the user.
NOTICE = re.compile(
    r"Agent \S+ sent a message:|Its closing message:|It left no closing message\.|"
    r"Background subagent \S+ (?:finished|was stopped|ran out of room|declined|failed|ended abnormally)|"
    r"You are repeating the exact same tool call|Repeated tool call detected:|"
    r"background job \S+(?: \(|\n)|The user switched this session (?:to plan mode|back to the default mode)\.")
PARTS = MessageParts()


class DeepSeekHarness(Harness):
    name = "deepseek_harness"

    def codec(self, codec: Codec) -> Codec:
        """Its own route (the Messages API to DeepSeek) sends consecutive user messages as one, so
        what it wrote, what the user typed and tool results share a message: as parts, they are
        its messages again, as its other routes send them."""

        return PARTS if codec.name == "anthropic_messages" else codec

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and (item.text.lstrip().startswith(RUNTIME) or NOTICE.match(item.text.lstrip())):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def state_key(self, item: Item) -> str | None:
        """Runtime snapshots (each supersedes the earlier ones) and what became of the instruction
        files: a message about the same files stands in for the one before it."""

        if item.text.lstrip().startswith(RUNTIME):
            return "runtime"
        if REBASED in item.text:
            return "instructions"
        if paths := CHANGED.findall(item.text):
            return "instructions:" + "|".join(sorted(paths))
        return None

    def state(self, codec: Codec, items: list[WireItem]) -> State:
        """In the order the harness wrote it: an instruction update replaces what came before it."""

        return State(tuple(sorted(super().state(codec, items).items, key=lambda item: item.ref)))

    # No `starter`: its hooks (through its Claude Code hooks bridge) run in its sandbox, where
    # nothing can start Relay.
    def settings(self) -> list[Setting]:
        patch = Path(os.getenv("DSH_HOME", "~/.dsh")).expanduser() / "cordis.patch.yml"
        deepseek = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/anthropic"
        return [Setting(patch, DEEPSEEK, endpoint=deepseek)] + [
            Setting(patch, (*ROUTES, route, "baseURL"), endpoint=DEFAULTS.get(route, ""))
            for route in yaml_keys(patch, ROUTES)
        ] + [
            Setting(patch, ("[id=compaction-basic]", "config", "auto"), False),
            # It trims old tool results in place under token pressure.
            Setting(patch, ("[id=tool-result-pruner]", "config", "thresholdChars"), NEVER),
        ]
