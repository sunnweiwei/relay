"""DeepSeek Harness (`dsh`); providers are plugin entries in the home-level cordis.patch.yml.

The DeepSeek route takes a `baseURL` (its DEEPSEEK_BASE_URL may only be exported, never
written to a file). Of the pi-ai routes (OpenAI, Google, Anthropic, gateways) only those the
user already declared are wrapped: a patch replaces a plugin's whole config, so adding a
pi-ai entry could hide routes configured in other layers.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..install import Setting, yaml_keys
from .base import Harness

DEEPSEEK = ("[id=llm-deepseek]", "config", "baseURL")
ROUTES = ("[id=llm-pi-ai]", "config", "providers")
DEFAULTS = {  # pi-ai catalog endpoints, for routes without an explicit baseURL
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com",
}


class DeepSeekHarness(Harness):
    name = "deepseek_harness"

    def settings(self) -> list[Setting]:
        patch = Path(os.getenv("DSH_HOME", "~/.dsh")).expanduser() / "cordis.patch.yml"
        deepseek = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/anthropic"
        return [Setting(patch, DEEPSEEK, endpoint=deepseek)] + [
            Setting(patch, (*ROUTES, route, "baseURL"), endpoint=DEFAULTS.get(route, ""))
            for route in yaml_keys(patch, ROUTES)
        ]
