"""Anthropic Claude Code (Messages API).

Claude Code wraps injected context in `<system-reminder>` blocks and slash-command
tags, and starts a self-compacted session with a fixed continuation preamble.
"""

from __future__ import annotations

import os
from dataclasses import replace
import re
from collections.abc import Mapping
from pathlib import Path

from ..core.ir import Item, Kind, joined
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

    def recalled(self, items: list[Item], originals: list[Item]) -> list[Item]:
        """Its Read answers a file unchanged since an earlier Read of it (the same input) with a
        pointer to that result, which Relay may have folded or compacted away: there the earlier
        result answers the Read again (the file is unchanged, so it is the file as it is). Only a
        result that goes back whole (its text and its media, each where they belong) does; any
        other stays the pointer it was."""

        reads: dict[str, tuple[int, str, tuple]] = {}  # a Read's input -> its last result in full (ref, text, media)
        pointers: dict[tuple[int, str], tuple[str, tuple]] = {}  # (result ref, call id) -> what answers it again
        for call, result in zip(originals, originals[1:]):
            if call.kind is not Kind.TOOL_CALL or result.kind is not Kind.TOOL_RESULT:
                continue
            for id_, use in ([(i, t) for i, t in call.parts if i] if call.parts else [("", call.text)]):
                answer = next((t for i, t in result.parts if i == id_), None) if id_ and result.parts else result.text
                if not use.startswith("Read(") or answer is None:
                    continue
                key = (result.ref, id_ if result.parts else "")
                if not answer.startswith(UNCHANGED):  # (its media are its own only where it is the message's one result)
                    reads[use] = (result.ref, answer, result.media) if not (result.media and result.parts) else None
                elif (earlier := reads.get(use)) is not None and not (earlier[2] and result.parts) and not any(
                        item.ref == earlier[0] and earlier[1] in item.text and item.media == earlier[2] for item in items):
                    pointers[key] = earlier[1:]
        if not pointers:
            return items
        recalled = []
        for item in items:
            original = originals[item.ref] if item.ref is not None else None
            if original is not None and (item.text, item.media) == (original.text, original.media) and any(
                    ref == item.ref for ref, _ in pointers):
                parts = tuple((i, pointers.get((item.ref, i), (t,))[0]) for i, t in item.parts or (("", item.text),))
                media = next((m for (ref, _), (_, m) in pointers.items() if ref == item.ref and m), item.media)
                item = replace(item, text=joined(parts), media=media, parts=parts if item.parts else ())
            recalled.append(item)
        return recalled

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
