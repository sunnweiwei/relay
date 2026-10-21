"""pi coding agent (earendil-works/pi). Speaks each provider's own API."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import re

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness, plugin_file


SECTION = re.compile(r'Updated system prompt section "([^"]+)"')
# Its compaction's summary, and a branch's it came back from (OpenClaw, built on pi, too).
SUMMARIES = ("The conversation history before this point was compacted into the following summary:",
             "The following is a summary of a branch that this conversation came back from:")


class Pi(Harness):
    name = "pi"
    verified = "1.1.0"
    summaries = SUMMARIES

    def state_key(self, item: Item) -> str | None:
        """A resumed session appends its system prompt anew, with instruction files as they are now;
        a changed section alone comes as an update of that section (the prompt it updates stays)."""

        if item.kind is not Kind.SYSTEM:
            return None
        section = SECTION.match(item.text)
        return f"section:{section.group(1)}" if section else "system"

    def starter(self, command: str) -> list[Setting]:
        """An extension of Relay's: pi loads every file in its extensions folder."""

        folder = Path(os.getenv("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "extensions"
        return [plugin_file(folder / "relay-start.ts", "pi/relay-start.ts", command)]

    def matches(self, headers: Mapping[str, str]) -> bool:
        return headers.get("user-agent", "").startswith(("pi/", "pi ("))

    def settings(self) -> list[Setting]:
        # A provider entry with only `baseUrl` keeps pi's built-in models and logins.
        models = Path(os.getenv("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "models.json"
        return [
            Setting(models.with_name("settings.json"), ("compaction", "enabled"), False),
            Setting(models, ("providers", "openai", "baseUrl"), endpoint="https://api.openai.com/v1"),
            Setting(models, ("providers", "anthropic", "baseUrl"), endpoint="https://api.anthropic.com"),
            Setting(models, ("providers", "google", "baseUrl"),
                    endpoint="https://generativelanguage.googleapis.com/v1beta"),
        ]
