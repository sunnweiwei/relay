"""Charm Crush; providers are overridden by id in crush.json.

Crush (0.97.1) summarizes the whole session after a step and starts over from the summary,
given as the user's: a step that left the turn unfinished goes on from a message telling the
model so, with the turn's request repeated; otherwise the next prompt follows.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import CRUSH_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import REMINDER, Harness

DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}


INTERRUPTED = "The previous session was interrupted because it got too long, the initial user request was: `{}`"


class Crush(Harness):
    name = "crush"
    injected = (REMINDER, re.compile(r"<system_reminder>.*?</system_reminder>", re.S))

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(f"{CRUSH_PROMPT}\n\nProvide a detailed summary of our conversation above.", keep=keep.pending())

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """Its system prompt and reminders, the summary; mid-turn, the turn's request again."""

        systems, others = self.split_state(state)
        front = (*systems, *others, *head)
        prompt = next((item for item in reversed(request) if item.kind is Kind.USER), None)
        again = (self.said(codec, Kind.USER, INTERRUPTED.format(prompt.text)),) if mid_turn and prompt else ()
        return front, tail, again

    def settings(self) -> list[Setting]:
        """Wrap the providers the user configured (or, if none, the built-in ones)."""

        config = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / "crush/crush.json"
        configured = json.loads(config.read_text()).get("providers", {}) if config.exists() else {}
        providers = {name: DEFAULTS.get(name) or DEFAULTS.get(p.get("type"), "") for name, p in configured.items()}
        return [
            *(Setting(config, ("providers", name, "base_url"), endpoint=url)
              for name, url in (providers or DEFAULTS).items()
              if url or configured.get(name, {}).get("base_url")),
            Setting(config, ("options", "disable_auto_summarize"), True),
        ]
