"""Nous Research's Hermes Agent; `hermes model` stores the endpoint in config.yaml.

Hermes picks the wire protocol from the endpoint's host, so the profile pins the one it
had: the Responses API for api.openai.com, and for Google's native Gemini API, which
Hermes only speaks to Google's host, Gemini's OpenAI-compatible API. Native Anthropic is
left alone: Hermes ignores Anthropic URLs on hosts it does not recognize.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from ..install import Setting, read
from .base import Harness, plugin_file

DEFAULTS = {
    "openai-api": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


class Hermes(Harness):
    name = "hermes"
    verified = "0.19.0"
    summaries = ("[CONTEXT COMPACTION — REFERENCE ONLY]", "[CONTEXT SUMMARY]:")  # (and its older one)

    def starter(self, command: str) -> list[Setting]:
        """A plugin of Relay's in its user plugins, enabled (user plugins are opt-in)."""

        home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
        folder, config = home / "plugins" / "relay-start", home / "config.yaml"
        enabled = read(config, ("plugins", "enabled")) or []
        return [plugin_file(folder / "plugin.yaml", "hermes/relay-start/plugin.yaml", command),
                plugin_file(folder / "__init__.py", "hermes/relay-start/__init__.py", command),
                Setting(config, ("plugins", "enabled"), [*enabled, "relay-start"])]

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser() / "config.yaml"
        provider = read(config, ("model", "provider"))
        if not provider or provider == "anthropic":
            raise ValueError("configure an OpenAI-compatible or Gemini provider with `hermes model` first")
        base_url = (read(config, ("model", "base_url")) or DEFAULTS.get(provider, "")).rstrip("/")
        settings = []
        if urlsplit(base_url).hostname == "api.openai.com" and not read(config, ("model", "api_mode")):
            settings.append(Setting(config, ("model", "api_mode"), value="codex_responses"))
        if "generativelanguage.googleapis.com" in base_url and not base_url.endswith("/openai"):
            base_url += "/openai"
            settings.append(Setting(config, ("model", "base_url"), value=base_url))
        return [*settings, Setting(config, ("model", "base_url"), endpoint=base_url),
                Setting(config, ("compression", "enabled"), False)]
