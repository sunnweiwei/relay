"""OpenClaw; built-in providers are overridden under models.providers in openclaw.json."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness
from .pi import SUMMARIES

DEFAULTS = {  # provider -> (default endpoint, native request adapter)
    "openai": ("https://api.openai.com/v1", None),
    "anthropic": ("https://api.anthropic.com", "anthropic-messages"),
    "google": ("https://generativelanguage.googleapis.com/v1beta", "google-generative-ai"),
}


INTERNAL_CONTEXT = "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>"
# Since 2026.10, the context of the turn (sessions, subagents, media tasks) in a user message of
# its own on all but native endpoints (Relay's too), kept after the user message it goes with.
RUNTIME_CONTEXT = "OpenClaw runtime context"
# Messages it sends the model as the user would (`sendCustomMessage`): a check that a saved plan
# still has steps left, a tool call cut off at the output limit, a turn the user interrupted.
NOTICES = ("This run’s latest successfully saved plan", "Runtime notice: The tool call", "<turn_aborted>")


class OpenClaw(Harness):
    name = "openclaw"
    verified = "2026.10.1"
    summaries = SUMMARIES

    def refine(self, item: Item) -> Item:
        # Its internal-context message (sessions, subagents) follows each user message: OpenClaw's.
        if item.kind is Kind.USER and item.text.lstrip().startswith((INTERNAL_CONTEXT, RUNTIME_CONTEXT, *NOTICES)):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def state_key(self, item: Item) -> str | None:
        return "runtime" if item.kind is Kind.CONTEXT and item.text.lstrip().startswith(RUNTIME_CONTEXT) else None

    def volatile(self, item: Item) -> bool:
        # Some versions append a fresh internal-context message to every request instead and drop
        # the previous one.
        return item.kind is Kind.CONTEXT and item.text.lstrip().startswith(INTERNAL_CONTEXT)

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("OPENCLAW_CONFIG_PATH", "~/.openclaw/openclaw.json")).expanduser()
        providers = json.loads(config.read_text()).get("models", {}).get("providers", {}) if config.exists() else {}
        settings = []
        for name, (url, api) in DEFAULTS.items():
            settings.append(Setting(config, ("models", "providers", name, "baseUrl"), endpoint=url))
            if api and "api" not in providers.get(name, {}):
                # An entry with a baseUrl but no api would fall back to openai-completions.
                settings.append(Setting(config, ("models", "providers", name, "api"), value=api))
        return [*settings, Setting(config, ("agents", "defaults", "compaction", "enabled"), False)]
