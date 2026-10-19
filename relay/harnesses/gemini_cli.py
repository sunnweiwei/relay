"""Google Gemini CLI (Gemini API with an API key; reads ~/.gemini/.env)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from ..protocols.base import Codec, WireItem, canonical_json
from .base import NEVER, REMINDER, Harness, command_hook

CONTEXT_PREFIX = "This is the Gemini CLI. We are setting up the context"


class GeminiCli(Harness):
    name = "gemini_cli"
    verified = "0.63.0"
    agent = re.compile(r"\bGeminiCLI[\w-]*/(\d[\w.-]*)")
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

    def state_key(self, item: Item) -> str | None:
        """The session context (date, workspace, GEMINI.md, the MCP servers' instructions), which it
        re-renders in place, and a sub-agent's loaded context."""

        text = item.text.lstrip()
        if text.startswith("<session_context>") or text.startswith(CONTEXT_PREFIX):
            return "session"
        return "loaded" if text.startswith("<loaded_context>") else None

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

    def starter(self, command: str) -> list[Setting]:
        return [command_hook(Path("~/.gemini/settings.json").expanduser(), "SessionStart", command, timeout=30_000)]  # (ms)

    def settings(self) -> list[Setting]:
        env, settings = Path("~/.gemini/.env").expanduser(), Path("~/.gemini/settings.json").expanduser()
        return [Setting(env, ("GOOGLE_GEMINI_BASE_URL",), endpoint="https://generativelanguage.googleapis.com"),
                Setting(settings, ("model", "compressionThreshold"), NEVER)]  # a share of the window
