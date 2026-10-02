"""OpenCode (anomalyco/opencode), built on the AI SDK."""

from __future__ import annotations

import os
from pathlib import Path

from ..install import Setting
from .base import Harness


class OpenCode(Harness):
    name = "opencode"
    config_file = "opencode/opencode.json"

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / self.config_file
        provider = ("provider", "{}", "options", "baseURL")
        defaults = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1",
                    "google": "https://generativelanguage.googleapis.com/v1beta"}
        return [
            *(Setting(config, tuple(k.format(name) for k in provider), endpoint=url) for name, url in defaults.items()),
            # Pruning rewrites old tool outputs, which would break prefix reuse.
            Setting(config, ("compaction", "prune"), False),
        ]


class Kilo(OpenCode):
    """Kilo CLI, an OpenCode fork with the same configuration schema."""

    name = "kilo"
    config_file = "kilo/config.json"
