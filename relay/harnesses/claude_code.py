"""Anthropic Claude Code (Messages API).

Claude Code wraps injected context in `<system-reminder>` blocks and slash-command
tags, and starts a self-compacted session with a fixed continuation preamble.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from ..protocols.base import Codec, WireItem
from .base import REMINDER, Harness
# Slash commands and their output, which Claude Code writes into a user message, at times
# together with what the user typed next (`/compact` followed by the next request).
COMMAND = re.compile(
    r"<(command-name|command-message|command-args|local-command-stdout|local-command-stderr|local-command-caveat)>"
    r".*?</\1>", re.S)
SUMMARY_PREFIX = "This session is being continued from a previous conversation"
COMPACT_PROMPT = "Your task is to create a detailed summary of the conversation so far"  # `/compact`, auto-compact


class ClaudeCode(Harness):
    name = "claude_code"
    injected = (REMINDER, COMMAND)

    def matches(self, headers: Mapping[str, str]) -> bool:
        return "x-claude-code-session-id" in headers or headers.get(
            "user-agent", ""
        ).startswith("claude-cli")

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and self.visible(item.text).startswith(SUMMARY_PREFIX):
            return replace(item, kind=Kind.SUMMARY)
        return super().refine(item)  # context only when nothing but injected blocks remains

    def compacting(self, codec: Codec, items: list[WireItem]) -> bool:
        """`/compact` (and auto-compact) appends its prompt to the last user message."""

        return bool(items) and COMPACT_PROMPT in codec.classify(items[-1]).text

    def state_key(self, item: Item) -> str | None:
        """System messages that restate the environment, or which MCP servers' instructions apply."""

        if item.kind is not Kind.SYSTEM:
            return None
        text = item.text.lstrip()
        if text.startswith("# Environment"):
            return "environment"
        if text.startswith("# MCP Server Instructions") or "MCP servers have disconnected" in text:
            return "mcp"
        return None

    def settings(self) -> list[Setting]:
        home = Path(os.getenv("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
        return [Setting(home / "settings.json", ("env", "ANTHROPIC_BASE_URL"), endpoint="https://api.anthropic.com")]

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        return ["claude", *args], {"ANTHROPIC_BASE_URL": relay_url}
