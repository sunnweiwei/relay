"""Kimi Code CLI; wraps every provider already listed in ~/.kimi-code/config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from ..core.ir import Item
from ..install import Setting
from .base import NEVER, Harness

DEFAULTS = {  # by provider type, for providers without an explicit base_url
    "kimi": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openai_responses": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google-genai": "https://generativelanguage.googleapis.com",
}


class KimiCode(Harness):
    name = "kimi_code"

    def state_key(self, item: Item) -> str | None:
        """Reminders that restate the date, or the approval mode, whenever it changes."""

        if "Today's date is" in item.text:
            return "date"
        if " mode is active" in item.text:
            return "mode"
        return None

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("KIMI_CODE_HOME", "~/.kimi-code")).expanduser() / "config.toml"
        current = tomllib.loads(config.read_text()) if config.exists() else {}
        return [
            *(Setting(config, ("providers", name, "base_url"), endpoint=DEFAULTS.get(provider.get("type"), ""))
              for name, provider in current.get("providers", {}).items()
              if provider.get("base_url") or provider.get("type") in DEFAULTS),
            # Kimi Code compacts near a model's max_context_size, which must be positive.
            *(Setting(config, ("models", name, "max_context_size"), NEVER) for name in current.get("models", {})),
        ]
