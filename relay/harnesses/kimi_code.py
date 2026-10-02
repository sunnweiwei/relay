"""Kimi Code CLI; wraps every provider already listed in ~/.kimi-code/config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from ..install import Setting
from .base import Harness

DEFAULTS = {  # by provider type, for providers without an explicit base_url
    "kimi": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openai_responses": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google-genai": "https://generativelanguage.googleapis.com",
}


class KimiCode(Harness):
    name = "kimi_code"

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("KIMI_CODE_HOME", "~/.kimi-code")).expanduser() / "config.toml"
        providers = tomllib.loads(config.read_text()).get("providers", {}) if config.exists() else {}
        return [
            Setting(config, ("providers", name, "base_url"), endpoint=DEFAULTS.get(provider.get("type"), ""))
            for name, provider in providers.items()
            if provider.get("base_url") or provider.get("type") in DEFAULTS
        ]
