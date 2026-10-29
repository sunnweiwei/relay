"""Anthropic Claude Code (Messages API).

Claude Code wraps injected context in `<system-reminder>` blocks and slash-command
tags, and starts a self-compacted session with a fixed continuation preamble.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from ..protocols.base import WireItem
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
# Its Read's answers pointing to an earlier Read of the same file, unchanged since.
UNCHANGED = ("Wasted call \u2014 file unchanged since your last Read.", "File unchanged since last read.")
INSTRUCTION_FILE = re.compile(r"^Contents of (\S+) \(", re.M)
REMINDERS = {"instructions": "Codebase and user instructions are shown below",
             "reread:": "Instruction files were re-read",  # (by the files it names)
             "attribution": "Attribution for git commits"}


def settings_file() -> Path:
    return Path(os.getenv("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / "settings.json"


class ClaudeCode(Harness):
    name = "claude_code"
    verified = "2.1.294"
    agent = re.compile(r"\bclaude-cli/(\d[\w.-]*)")
    injected = (REMINDER, COMMAND)
    summaries = (SUMMARY_PREFIX,)
    compaction_prompts = (COMPACT_PROMPT,)  # `/compact` and auto-compact append it to the last user message
    direct_window = 300_000  # its Claude models run with 1M windows (the catalog: 200k); about Codex's

    def recalled(self, items: list[Item], originals: list[Item], raw: list[WireItem]) -> list[Item]:
        """Its Read answers a file unchanged since an earlier Read of it (the same input) with a
        pointer to that result, which Relay may have folded or compacted away: there the earlier
        result answers the Read again (the file is unchanged, so it is the file as it is). Results
        are found by the call they answer; a pointer goes back where it is its message's one result
        (an item answering parallel calls is all of their results: it stays as it was), and an
        earlier result with media where its media are its own (its message's one result)."""

        shown = {item.ref for item in items if item.ref is not None
                 and (item.text, item.media) == (originals[item.ref].text, originals[item.ref].media)}
        reads: dict[str, str] = {}  # a Read's id -> its input
        answers: dict[str, tuple[int, str, tuple] | None] = {}  # a Read's input -> its last result in full (ref, text, media)
        pointers: dict[int, tuple[str, str, tuple]] = {}  # a pointer's ref -> (its text, what answers it again)
        for item, wire in zip(originals, raw):
            content = wire.get("content")
            blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
            reads.update({b.get("id"): json.dumps(b.get("input"), sort_keys=True) for b in blocks
                          if b.get("type") == "tool_use" and b.get("name") == "Read"})
            results = [b for b in blocks if b.get("type") == "tool_result"] if item.kind is Kind.TOOL_RESULT else []
            for result in results:
                if (use := reads.get(result.get("tool_use_id"))) is None:
                    continue
                text = _text(result.get("content"))
                if not text.startswith(UNCHANGED):
                    answers[use] = (item.ref, text, item.media) if len(results) == 1 or not item.media else None
                elif len(results) == 1 and (earlier := answers.get(use)) is not None and earlier[0] not in shown:
                    pointers[item.ref] = (text, earlier[1], earlier[2])
        if not pointers:
            return items
        recalled = []
        for item in items:
            if item.ref in pointers and item.ref in shown and pointers[item.ref][0] in item.text:  # (as the harness sent it)
                pointer, text, media = pointers[item.ref]
                item = replace(item, text=item.text.replace(pointer, text, 1), media=(*media, *item.media))
            recalled.append(item)
        return recalled

    def matches(self, headers: Mapping[str, str]) -> bool:
        return "x-claude-code-session-id" in headers or headers.get(
            "user-agent", ""
        ).startswith("claude-cli")

    def session(self, headers: Mapping[str, str]) -> str:
        return headers.get("x-claude-code-session-id", "")

    def compacts(self, session: str) -> str | None:
        return f'claude --resume {session} -p "/compact"' if re.fullmatch(r"[\w-]+", session) else None

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


def _text(content: object) -> str:
    """A tool result's text (its content's text blocks)."""

    if isinstance(content, str):
        return content
    return "\n".join(b.get("text") or "" for b in content or [] if isinstance(b, dict) and b.get("type") == "text")
