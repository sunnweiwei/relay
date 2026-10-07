"""Google Gemini CLI (Gemini API with an API key; reads ~/.gemini/.env).

Compressing (0.63.0), Gemini CLI keeps the history from the first user message past 70% of it,
summarizes what came before into a state snapshot, and starts over from its session context
with the snapshot, acknowledged by the model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import GEMINI_PROMPT
from ..protocols.base import Codec, WireItem, canonical_json
from . import keep
from .base import NEVER, REMINDER, Harness

CONTEXT_PREFIX = "This is the Gemini CLI. We are setting up the context"
SESSION = re.compile(r"<session_context>.*?</session_context>", re.S)
ACKNOWLEDGED = "Got it. Thanks for the additional context!"


class GeminiCli(Harness):
    name = "gemini_cli"
    # The date, workspace and GEMINI.md, re-rendered in the first user message when they change
    # (`<loaded_context>` alone opens a sub-agent's conversation).
    injected = (REMINDER, re.compile(r"<session_context>.*?</session_context>", re.S),
                re.compile(r"<loaded_context>.*?</loaded_context>", re.S))

    def matches(self, headers: Mapping[str, str]) -> bool:
        return headers.get("user-agent", "").startswith("GeminiCLI")

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and item.text.lstrip().startswith(CONTEXT_PREFIX):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(GEMINI_PROMPT, keep=keep.split(0.3))

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """Its session context and the summary in one message, the model's acknowledgement."""

        *before, summary = head
        first = next((item for item in request if item.kind in (Kind.USER, Kind.CONTEXT)), None)
        session = SESSION.search(first.text) if first else None
        parts = [{"text": text} for text in ((session[0] if session else ""), summary.text) if text]
        merged = replace(summary, wire=json.dumps({"role": "user", "parts": parts}))
        systems, _ = self.split_state(state)
        front = (*systems, *before, merged, self.said(codec, Kind.ASSISTANT, ACKNOWLEDGED))
        return front, tail, ()

    def identity(self, codec: Codec, items: list[WireItem]) -> list[bytes]:
        """Gemini CLI masks bulky old tool outputs in place (`<tool_output_masked>`; there is no
        setting to stop it), so a tool result compares by the calls it answers, not its output."""

        keys = super().identity(codec, items)
        for index, item in enumerate(items):
            calls = [(part["functionResponse"].get("name"), part["functionResponse"].get("id"))
                     for part in item.get("parts") or [] if isinstance(part.get("functionResponse"), dict)]
            if calls and all(id for _, id in calls):
                keys[index] = b"\0result\0" + canonical_json(list(dict.fromkeys(calls)))
        return keys

    def settings(self) -> list[Setting]:
        env, settings = Path("~/.gemini/.env").expanduser(), Path("~/.gemini/settings.json").expanduser()
        return [Setting(env, ("GOOGLE_GEMINI_BASE_URL",), endpoint="https://generativelanguage.googleapis.com"),
                Setting(settings, ("model", "compressionThreshold"), NEVER)]  # a share of the window
