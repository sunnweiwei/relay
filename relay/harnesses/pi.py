"""pi coding agent (earendil-works/pi). Speaks each provider's own API."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness


class Pi(Harness):
    name = "pi"

    def state_key(self, item: Item) -> str | None:
        """A resumed session appends its system prompt anew, with instruction files as they are now."""

        return "system" if item.kind is Kind.SYSTEM else None

    def matches(self, headers: Mapping[str, str]) -> bool:
        return headers.get("user-agent", "").startswith(("pi/", "pi ("))

    def settings(self) -> list[Setting]:
        # A provider entry with only `baseUrl` keeps pi's built-in models and logins.
        models = Path(os.getenv("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "models.json"
        return [
            Setting(models, ("providers", "openai", "baseUrl"), endpoint="https://api.openai.com/v1"),
            Setting(models, ("providers", "anthropic", "baseUrl"), endpoint="https://api.anthropic.com"),
            Setting(models, ("providers", "google", "baseUrl"),
                    endpoint="https://generativelanguage.googleapis.com/v1beta"),
        ]
