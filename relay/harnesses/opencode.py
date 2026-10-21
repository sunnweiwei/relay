"""OpenCode (anomalyco/opencode), built on the AI SDK."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness, plugin_file

# After compacting itself: the question its summary (an assistant message) answers, and, after an
# automatic compaction, a nudge to go on.
QUESTION = "What did we do so far?"
NUDGE = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."


class OpenCode(Harness):
    name = "opencode"
    verified = "1.19.0"
    agent = re.compile(r"\bopencode/(\d[\w.-]*)")
    config_file = "opencode/opencode.json"
    summaries = (QUESTION,)

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and item.text.rstrip().endswith(NUDGE):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def starter(self, command: str) -> list[Setting]:
        """A plugin of Relay's: it loads every file in its global plugin folder."""

        folder = (Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / self.config_file).parent / "plugin"
        return [plugin_file(folder / "relay-start.js", "opencode/relay-start.js", command)]

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
    verified = "7.8.8"
    agent = re.compile(r"\b(?:Kilo-Code|opencode)/(\d[\w.-]*)")  # (some of its requests still say opencode)
    config_file = "kilo/config.json"
