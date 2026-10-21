"""Charm Crush; providers are overridden by id in crush.json."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import REMINDER, Harness

DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}


class Crush(Harness):
    name = "crush"
    verified = "0.98.1"
    agent = re.compile(r"\bCharm-Crush/v?(\d[\w.-]*)")
    # Its reminders of the todo list, as messages of their own.
    injected = (REMINDER, re.compile(r"<system_reminder>.*?</system_reminder>", re.S))
    summaries = ("## Current State",)  # (its summary prompt's first required section)

    def state_key(self, item: Item) -> str | None:
        return "todo" if item.kind is Kind.CONTEXT and "todo list" in item.text else None

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
