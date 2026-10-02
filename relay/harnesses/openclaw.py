"""OpenClaw; built-in providers are overridden under models.providers in openclaw.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness

DEFAULTS = {  # provider -> (default endpoint, native request adapter)
    "openai": ("https://api.openai.com/v1", None),
    "anthropic": ("https://api.anthropic.com", "anthropic-messages"),
    "google": ("https://generativelanguage.googleapis.com/v1beta", "google-generative-ai"),
}


INTERNAL_CONTEXT = "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>"


class OpenClaw(Harness):
    name = "openclaw"

    def volatile(self, item: Item) -> bool:
        # OpenClaw appends a fresh internal-context message (sessions, subagents) to every
        # request and drops the previous one.
        return item.kind in {Kind.USER, Kind.CONTEXT} and item.text.lstrip().startswith(INTERNAL_CONTEXT)

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("OPENCLAW_CONFIG_PATH", "~/.openclaw/openclaw.json")).expanduser()
        providers = json.loads(config.read_text()).get("models", {}).get("providers", {}) if config.exists() else {}
        settings = []
        for name, (url, api) in DEFAULTS.items():
            settings.append(Setting(config, ("models", "providers", name, "baseUrl"), endpoint=url))
            if api and "api" not in providers.get(name, {}):
                # An entry with a baseUrl but no api would fall back to openai-completions.
                settings.append(Setting(config, ("models", "providers", name, "api"), value=api))
        return settings
