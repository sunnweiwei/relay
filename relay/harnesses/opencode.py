"""OpenCode (anomalyco/opencode), built on the AI SDK.

Compacting (1.18.35), OpenCode keeps the newest messages (a quarter of its usable context, at
most 15k tokens) and summarizes what came before; the summary is its answer to "What did we do
so far?", and a prompt to go on follows what it kept. Kilo, its fork, keeps two of the model's
rounds at most, adds environment details to each message it writes, and at a turn start
summarizes everything before the new prompt.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..core.ir import AGENT_KINDS, Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import OPENCODE_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import Harness

ASKED = "What did we do so far?"
CONTINUE = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
DETAILS = re.compile(r"\n*<environment_details>\nMessage time: [^\n]*\n(.*?)</environment_details>", re.S)


class OpenCode(Harness):
    name = "opencode"
    config_file = "opencode/opencode.json"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(OPENCODE_PROMPT, keep=keep.tail(share=0.25, cap=15_000))

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """Its question, the summary as its answer, what it kept, and the prompt to go on."""

        *before, summary = head
        answer = Item(Kind.SUMMARY, summary.text, wire=json.dumps(codec.write(Item(Kind.ASSISTANT, summary.text))))
        systems, others = self.split_state(state)
        front = (*systems, *before, self.written(codec, ASKED, request), answer, *others)
        return front, tail, (self.written(codec, CONTINUE, request),) if self.goes_on(tail) else ()

    def written(self, codec: Codec, text: str, request: tuple[Item, ...]) -> Item:
        """A message it writes itself."""

        return self.said(codec, Kind.USER, text)

    def goes_on(self, tail: tuple[Item, ...]) -> bool:
        return True

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / self.config_file
        provider = ("provider", "{}", "options", "baseURL")
        defaults = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1",
                    "google": "https://generativelanguage.googleapis.com/v1beta"}
        return [
            *(Setting(config, tuple(k.format(name) for k in provider), endpoint=url) for name, url in defaults.items()),
            # Pruning rewrites old tool outputs, which would break prefix reuse.
            Setting(config, ("compaction", "prune"), False),
            Setting(config, ("compaction", "auto"), False),
        ]


class Kilo(OpenCode):
    """Kilo CLI, an OpenCode fork with the same configuration schema."""

    name = "kilo"
    config_file = "kilo/config.json"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        """At a turn start (no step taken since the new prompt), all before it is summarized."""

        last = next((item for item in reversed(request) if item.kind is Kind.USER or item.kind in AGENT_KINDS), None)
        if last is not None and last.kind is Kind.USER:
            return Native(OPENCODE_PROMPT, keep=keep.pending())
        return Native(OPENCODE_PROMPT, keep=keep.tail(share=0.25, cap=15_000, most=2))  # (its tail_turns)

    def written(self, codec: Codec, text: str, request: tuple[Item, ...]) -> Item:
        """With its environment details, as of now, in a part of their own."""

        said = super().written(codec, text, request)
        details = next((m for item in reversed(request) if (m := DETAILS.search(item.text))), None)
        wire = json.loads(said.wire or "{}")
        if details is None or not isinstance(wire.get("content"), list) or not wire["content"]:
            return said
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        extra = f"\n\n<environment_details>\nMessage time: {now}\n{details[1]}</environment_details>"
        wire["content"] = [*wire["content"], {**wire["content"][-1], "text": extra}]
        return replace(said, text=f"{text}{extra}", wire=json.dumps(wire))

    def goes_on(self, tail: tuple[Item, ...]) -> bool:
        """Not after a turn start's replay."""

        return not any(item.kind is Kind.USER for item in tail)
