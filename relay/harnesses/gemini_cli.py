"""Google Gemini CLI (Gemini API with an API key; reads ~/.gemini/.env)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import REMINDER, Harness

CONTEXT_PREFIX = "This is the Gemini CLI. We are setting up the context"


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

    def settings(self) -> list[Setting]:
        env = Path("~/.gemini/.env").expanduser()
        return [Setting(env, ("GOOGLE_GEMINI_BASE_URL",), endpoint="https://generativelanguage.googleapis.com")]
