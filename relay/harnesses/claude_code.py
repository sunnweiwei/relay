"""Anthropic Claude Code (Messages API).

Claude Code wraps injected context in `<system-reminder>` blocks and slash-command
tags, and starts a self-compacted session with a fixed continuation preamble.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import REMINDER, Harness, command_hook
# Slash commands and their output, which Claude Code writes into a user message, at times
# together with what the user typed next (`/compact` followed by the next request).
COMMAND = re.compile(
    r"<(command-name|command-message|command-args|local-command-stdout|local-command-stderr|local-command-caveat)>"
    r".*?</\1>", re.S)
SUMMARY_PREFIX = "This session is being continued from a previous conversation"
# `/compact` and auto-compact, in every variant: "... of the conversation so far", "... of this
# conversation" (the earlier part, recent messages kept) and "... of the RECENT portion ...".
COMPACT_PROMPT = "Your task is to create a detailed summary of"
INSTRUCTION_FILE = re.compile(r"^Contents of (\S+) \(", re.M)
REMINDERS = {"instructions": "Codebase and user instructions are shown below",
             "reread:": "Instruction files were re-read",  # (by the files it names)
             "attribution": "Attribution for git commits"}


def settings_file() -> Path:
    return Path(os.getenv("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / "settings.json"


class ClaudeCode(Harness):
    name = "claude_code"
    injected = (REMINDER, COMMAND)
    summaries = (SUMMARY_PREFIX,)
    compaction_prompts = (COMPACT_PROMPT,)  # `/compact` and auto-compact append it to the last user message

    def matches(self, headers: Mapping[str, str]) -> bool:
        return "x-claude-code-session-id" in headers or headers.get(
            "user-agent", ""
        ).startswith("claude-cli")

    def state_key(self, item: Item) -> str | None:
        """Reminders that restate its state, wherever they come (after compacting itself it
        re-injects them below its summary): the instruction files, those re-read since, the
        attribution guidance; and the system message with the environment (the agents, the MCP
        servers' instructions), or which MCP servers' instructions apply."""

        text = item.text.lstrip()
        if item.kind is Kind.CONTEXT:
            for key, start in REMINDERS.items():
                if text.startswith(f"<system-reminder>\n{start}"):
                    return key + ",".join(sorted(set(INSTRUCTION_FILE.findall(text)))) if key.endswith(":") else key
            return None
        if item.kind is not Kind.SYSTEM:
            return None
        if "# Environment\n" in text or text.startswith("# Environment"):
            return "environment"
        if text.startswith("# MCP Server Instructions") or "MCP servers have disconnected" in text:
            return "mcp"
        return None

    def starter(self, command: str) -> list[Setting]:
        return [command_hook(settings_file(), "SessionStart", command)]  # (it runs before the first request)

    def settings(self) -> list[Setting]:
        return [Setting(settings_file(), ("env", "ANTHROPIC_BASE_URL"), endpoint="https://api.anthropic.com"),
                Setting(settings_file(), ("env", "DISABLE_AUTO_COMPACT"), "1")]  # `/compact` still works

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        return ["claude", *args], {"ANTHROPIC_BASE_URL": relay_url}
