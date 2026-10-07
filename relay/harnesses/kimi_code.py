"""Kimi Code CLI; wraps every provider already listed in ~/.kimi-code/config.toml.

Compacting (2.1.1), Kimi Code summarizes the whole history and keeps only the user's own
messages (20k tokens' worth), ahead of its summary; a reminder to go on and its reminders, sent
anew, follow.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import KIMI_PROMPT, KIMI_RECOVERY
from ..protocols.base import Codec
from . import keep
from .base import NEVER, Harness

COMPACTED = ("The conversation so far has been compacted to free up context. What follows is your own working summary of "
             "this task — use it to continue your train of thought rather than starting over. Treat it as notes, not "
             "proof: where it says a step was done, tests passed, or a fix worked, verify that yourself before relying "
             "on it. Any user messages earlier in this context are preserved verbatim from the compacted conversation; "
             "where a system-reminder note among them marks an omitted middle section, the user messages it replaced are "
             "covered by this summary. The summary records which earlier requests were already addressed.\n")
USERS = 20_000  # tokens of the user's messages it keeps
GO_ON = "<system-reminder>\nContext compaction is complete — continue the work that was in progress when it began.\n</system-reminder>"

DEFAULT_CONTEXT = 262_144  # Kimi Code's max_context_size where a model names none
DEFAULTS = {  # by provider type, for providers without an explicit base_url
    "kimi": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openai_responses": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google-genai": "https://generativelanguage.googleapis.com",
}


class KimiCode(Harness):
    name = "kimi_code"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        """Its summary ends with where to find what it left out, where its hooks report that."""

        lines = (local.extra.get("lines") if local else 0) or 0
        if not (local and local.transcript and lines):
            return Native(KIMI_PROMPT, COMPACTED, keep=keep.users(USERS))
        windows = f"  window 1: lines 1–{lines}   ← the conversation this note summarizes\n  window 2 (the one you are in now)"
        recovery = KIMI_RECOVERY.replace("{path}", local.transcript).replace("{windows}", windows)
        return Native(KIMI_PROMPT, COMPACTED, recovery.replace("{next}", str(lines + 1)), keep.users(USERS))

    def session(self, headers: Mapping[str, str], body: Mapping) -> tuple[str, str] | None:
        session = body.get("prompt_cache_key")
        return (session, "") if isinstance(session, str) and session else None

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """Its system items first, the user's messages and the summary, the reminder to go on,
        its reminders again."""

        systems, others = self.split_state(state)
        return (*systems, *head), tail, (self.said(codec, Kind.USER, GO_ON), *others)

    def state_key(self, item: Item) -> str | None:
        """Reminders that restate the date, or the approval mode, whenever it changes."""

        if "Today's date is" in item.text:
            return "date"
        if " mode is active" in item.text:
            return "mode"
        return None

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("KIMI_CODE_HOME", "~/.kimi-code")).expanduser() / "config.toml"
        current = tomllib.loads(config.read_text()) if config.exists() else {}
        return [
            *(Setting(config, ("providers", name, "base_url"), endpoint=DEFAULTS.get(provider.get("type"), ""))
              for name, provider in current.get("providers", {}).items()
              if provider.get("base_url") or provider.get("type") in DEFAULTS),
            # Kimi Code compacts near a model's max_context_size, which must be positive. It also asks
            # for up to that size less the prompt as output, unless max_output_size caps it: the cap
            # keeps the output it asked for before (an upstream may reject a billion tokens).
            *(Setting(config, ("models", name, "max_context_size"), NEVER) for name in current.get("models", {})),
            *(Setting(config, ("models", name, "max_output_size"), model.get("max_context_size", DEFAULT_CONTEXT))
              for name, model in current.get("models", {}).items() if "max_output_size" not in model),
        ]
