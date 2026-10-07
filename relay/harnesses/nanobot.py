"""HKUDS nanobot; providers take an `apiBase` in ~/.nanobot/config.json.

nanobot uses the Responses API only when the OpenAI base URL is api.openai.com, so the
profile pins `apiType: responses` for a provider that was talking to OpenAI directly.

Compacting (0.3.5, its consolidator), nanobot archives the summary in its system prompt and
goes on from a prompt pointing at it, with the new turn (or mid-turn, its last round).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import NANOBOT_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import NEVER, Harness

DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",  # OpenAI-compatible
    "anthropic": "https://api.anthropic.com",
}


ARCHIVED = "\n\n---\n\n[Archived Context Summary]\n\nPrevious conversation summary (last active {when}):\n{summary}"
CONTINUE = "Continue the active task from the working-memory checkpoint above."


class Nanobot(Harness):
    name = "nanobot"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(NANOBOT_PROMPT, keep=keep.last())

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The summary archived in the system prompt (the summary item stands for it); the prompt
        pointing at it, before the new turn's own (or mid-turn, the last round)."""

        *before, summary = head
        system = next((item for item in state if item.kind is Kind.SYSTEM), None)
        when = datetime.now(timezone.utc).isoformat()
        archive = (system.text if system else "") + ARCHIVED.format(when=when, summary=summary.text)
        wire = codec.write(Item(Kind.USER, archive))
        wire = {**wire, "role": "system"}
        front = [replace(summary, wire=json.dumps(wire)), *(i for i in state if i is not system), *before]
        prompt = next((n for n, item in enumerate(tail) if item.kind is Kind.USER), None)
        if prompt is None:
            return (*front, self.said(codec, Kind.USER, CONTINUE)), tail, ()
        if not tail[prompt].text.startswith(CONTINUE):
            tail = (*tail[:prompt], replace(tail[prompt], text=f"{CONTINUE}\n\n{tail[prompt].text}"), *tail[prompt + 1:])
        return tuple(front), tail, ()

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
        # Its checkpoints follow contextWindowTokens (0 would starve its memory archive).
        return [*settings, Setting(config, ("agents", "defaults", "contextWindowTokens"), NEVER)]
