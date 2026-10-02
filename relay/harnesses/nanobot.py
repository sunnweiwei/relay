"""HKUDS nanobot; providers take an `apiBase` in ~/.nanobot/config.json.

nanobot uses the Responses API only when the OpenAI base URL is api.openai.com, so the
profile pins `apiType: responses` for a provider that was talking to OpenAI directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..install import Setting
from .base import Harness

DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",  # OpenAI-compatible
    "anthropic": "https://api.anthropic.com",
}


class Nanobot(Harness):
    name = "nanobot"

    def settings(self) -> list[Setting]:
        """Wrap the providers the user configured (or, if none, the built-in ones)."""

        config = Path("~/.nanobot/config.json").expanduser()
        providers = json.loads(config.read_text()).get("providers", {}) if config.exists() else {}
        names = [name for name in DEFAULTS if name in providers] or list(DEFAULTS)
        settings = [Setting(config, ("providers", name, "apiBase"), endpoint=DEFAULTS[name]) for name in names]
        openai = providers.get("openai") or {}
        if "openai" in names and "api.openai.com" in (openai.get("apiBase") or DEFAULTS["openai"]) \
                and openai.get("apiType", "auto") == "auto":
            settings.append(Setting(config, ("providers", "openai", "apiType"), value="responses"))
        return settings
