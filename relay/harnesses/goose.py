"""Block's Goose; provider hosts are top-level keys of config.yaml."""

from __future__ import annotations

import os
from pathlib import Path

from ..install import Setting
from .base import Harness

DEFAULTS = {
    "OPENAI_HOST": "https://api.openai.com",
    "ANTHROPIC_HOST": "https://api.anthropic.com",
    "GOOGLE_HOST": "https://generativelanguage.googleapis.com",
}


class Goose(Harness):
    name = "goose"

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / "goose/config.yaml"
        return [*(Setting(config, (key,), endpoint=url) for key, url in DEFAULTS.items()),
                # No switch, and values outside (0, 1) may mean the default: the window's edge.
                Setting(config, ("GOOSE_AUTO_COMPACT_THRESHOLD",), 0.99)]
