"""SWE-agent's mini-swe-agent; litellm reads provider base URLs from its global .env."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..install import Setting
from .base import Harness

DEFAULTS = {  # litellm's own defaults for these variables
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "GEMINI_API_BASE": "https://generativelanguage.googleapis.com/v1beta",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
}


class MiniSwe(Harness):
    name = "mini_swe"

    def settings(self) -> list[Setting]:
        env = _config_dir() / ".env"
        return [Setting(env, (key,), endpoint=url) for key, url in DEFAULTS.items()]


def _config_dir() -> Path:
    """platformdirs' user_config_dir("mini-swe-agent"), unless MSWEA_GLOBAL_CONFIG_DIR is set."""

    if path := os.getenv("MSWEA_GLOBAL_CONFIG_DIR"):
        return Path(path).expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/mini-swe-agent").expanduser()
    return Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / "mini-swe-agent"
