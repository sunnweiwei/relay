"""Anthropic Claude Code (Messages API).

Claude Code wraps injected context in `<system-reminder>` blocks and slash-command
tags, and starts a self-compacted session with a fixed continuation preamble.

Compacting, Claude Code (2.1.292) summarizes with its own prompt everything before the model's
last round, which it keeps, and lays the request out anew: its instruction files as they read
now, with the summary in one user message; the date; the round kept; then what it announces
again (the agent types and the environment), and what it re-reads (recent files, the plan, the
skills it ran, its background tasks). Relay's compaction does the same (`native`, `compose`),
with what the plugin `relay install` adds reports (`relay.core.local`).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import CLAUDE_CODE_PROMPT
from ..protocols.base import Codec, WireItem
from . import keep
from .base import REMINDER, Harness, State
# Slash commands and their output, which Claude Code writes into a user message, at times
# together with what the user typed next (`/compact` followed by the next request).
COMMAND = re.compile(
    r"<(command-name|command-message|command-args|local-command-stdout|local-command-stderr|local-command-caveat)>"
    r".*?</\1>", re.S)
SUMMARY_PREFIX = "This session is being continued from a previous conversation"
# `/compact` and auto-compact, in every variant: "... of the conversation so far", "... of this
# conversation" (the earlier part, recent messages kept) and "... of the RECENT portion ...".
COMPACT_PROMPT = "Your task is to create a detailed summary of"
CONTINUED = ("This session is being continued from a previous conversation that ran out of context. The summary "
             "below covers the earlier portion of the conversation.\n\n")
TRANSCRIPT = ("\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or "
              "content you generated), read the full transcript at: {path}")
RESUME = ("\nContinue the conversation from where it left off without asking the user any further questions. Resume "
          "directly — do not acknowledge the summary, do not recap what was happening, do not preface with \"I'll "
          "continue\" or similar. Pick up the last task as if the break never happened.")
# The instruction files, as Claude Code announces them first and after re-reading them.
INSTRUCTIONS = ("<system-reminder>\nCodebase and user instructions are shown below. Be sure to adhere to these "
                "instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them "
                "exactly as written.\n\n{files}\n</system-reminder>")
REREAD = "Instruction files were re-read when this session started"
ATTRIBUTION = "Attribution for git commits and pull requests"
FILES = 5  # the files read last that Claude Code re-reads after compacting
INSTRUCTION_FILE = re.compile(r"Contents of (\S+) \(([^)]*)\):\n\n(.*?)(?=\n\nContents of \S+ \(|\n</system-reminder>|\Z)", re.S)
DESCRIPTIONS = {"project": "project instructions, checked into the codebase",
                "user": "user's private global instructions for all projects",
                "local": "user's private project instructions, not checked in"}
# The sections of the system message that opens a conversation, in its order.
SECTIONS = ("# Environment", "Available agent types for the Agent tool:", "The following skills are available",
            "Today's date is")


@dataclass(frozen=True)
class Summary(Native):
    """Claude Code's summary message: its model's answer without the analysis, within its
    continuation preamble."""

    def message(self, answer: str) -> str:
        answer = re.sub(r"<analysis>.*?</analysis>", "", answer, flags=re.S)
        answer = re.sub(r"<summary>(.*?)</summary>", lambda m: f"Summary:\n{m[1].strip()}", answer, flags=re.S)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        return f"{self.prefix}{answer}{self.suffix}"


def settings_file() -> Path:
    return Path(os.getenv("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser() / "settings.json"


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
        """`/compact` (and auto-compact) appends its prompt to the last user message, which system
        messages such as `<total_tokens>` may follow."""

        last = next((item for item in map(codec.classify, reversed(items)) if item.kind is not Kind.SYSTEM), None)
        return last is not None and COMPACT_PROMPT in last.text

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

    def session(self, headers: Mapping[str, str], body: Mapping) -> tuple[str, str] | None:
        session = headers.get("x-claude-code-session-id")
        first = next((m.get("content") for m in body.get("messages") or [] if m.get("role") == "user"), "")
        text = first if isinstance(first, str) else "".join(b.get("text", "") for b in first or [] if isinstance(b, dict))
        return (session, text) if session else None

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        path = local.transcript if local else None
        return Summary(CLAUDE_CODE_PROMPT, CONTINUED, (TRANSCRIPT.format(path=path) if path else "") + RESUME, keep.round())

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The instruction files, as they read now, in one message with the summary (unless what
        follows announces them, as a new process does); the date after it; the attribution
        reminder in the next prompt; and after the round kept, in one system message, the files
        read last (but those it shows) as Claude Code re-reads them, then the agent types and
        the environment, announced again; and the harness's other state."""

        *before, summary = head
        opening = next((item for item in state if item.kind is Kind.SYSTEM and item.text.lstrip().startswith(SECTIONS[0])), None)
        announced = next((item for item in state if item.ref is None and item.text.startswith(INSTRUCTIONS[:40])), None)
        attribution = next((item for item in state if item.ref is None and ATTRIBUTION in item.text), None)
        sections = _sections(opening.text if opening else "")
        instructions = "" if any(INSTRUCTIONS[18:60] in item.text for item in tail) else (
            _instructions(local) or (announced.text if announced else ""))
        blocks = [{"type": "text", "text": text} for text in (instructions and f"{instructions}\n", summary.text) if text]
        front = [*before, replace(summary, wire=json.dumps({"role": "user", "content": blocks}))]
        if date := sections.get(SECTIONS[3]):
            front.append(_system(date))
        shown = " ".join(item.text for item in tail if item.kind is Kind.TOOL_CALL)
        files = [_read(doc) for doc in (local.files if local else ()) if json.dumps(doc.path) not in shown][:FILES]
        again = "\n\n".join([*files, *(text for key in (SECTIONS[1], SECTIONS[0]) if (text := sections.get(key)))])
        others = tuple(item for item in state if item not in (opening, announced, attribution))
        prompt = next((n for n, item in enumerate(tail) if item.kind is Kind.USER), None)
        if attribution and not any(ATTRIBUTION in item.text for item in tail):
            if prompt is not None:  # at a turn start, in the new prompt, after the reminders it opens with
                text = tail[prompt].text
                at = sum(len(block) for block in _leading(text))
                tail = (*tail[:prompt], replace(tail[prompt], text=f"{text[:at]}{attribution.text}\n{text[at:]}"),
                        *tail[prompt + 1:])
            else:  # mid-turn, with the results the round kept
                last = max((n for n, item in enumerate(tail) if item.kind is Kind.TOOL_RESULT), default=len(tail) - 1)
                tail = (*tail[:last + 1], Item(Kind.CONTEXT, attribution.text), *tail[last + 1:])
        if again and tail and tail[-1].kind is Kind.SYSTEM and tail[-1].ref is not None:  # one system message, as it folds them
            return tuple(front), (*tail[:-1], replace(tail[-1], text=f"{tail[-1].text}\n\n{again}")), others
        return tuple(front), tail, (*([_system(again)] if again else []), *others)

    def state(self, codec: Codec, items: list[WireItem]) -> State:
        """Besides the system messages it keeps, the instruction files as Claude Code last
        announced each (they open its first user message, and come again when they change)."""

        state = super().state(codec, items)
        latest: dict[str, tuple[str, str]] = {}
        for item in map(codec.classify, items):
            for block in REMINDER.findall(item.text) if item.kind is Kind.USER else ():
                if INSTRUCTIONS[:60] in block or REREAD in block:
                    latest.update((m[1], (m[2], m[3])) for m in INSTRUCTION_FILE.finditer(block))
        files = [(path, description, content) for path, (description, content) in latest.items()]
        attribution = next((block for item in map(codec.classify, items) if item.kind is Kind.USER
                            for block in REMINDER.findall(item.text) if ATTRIBUTION in block), None)
        extra = [*([Item(Kind.CONTEXT, _render(files))] if files else []), *([Item(Kind.CONTEXT, attribution)] if attribution else [])]
        return State((*state.items, *extra))

    def settings(self) -> list[Setting]:
        return [Setting(settings_file(), ("env", "ANTHROPIC_BASE_URL"), endpoint="https://api.anthropic.com"),
                Setting(settings_file(), ("env", "DISABLE_AUTO_COMPACT"), "1")]  # `/compact` still works

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        return ["claude", *args], {"ANTHROPIC_BASE_URL": relay_url}


def _leading(text: str) -> list[str]:
    """The reminders a user message opens with."""

    blocks = []
    while match := re.match(r"\s*<system-reminder>.*?</system-reminder>\n?", text[sum(map(len, blocks)):], re.S):
        blocks.append(match[0])
    return blocks


def _system(text: str) -> Item:
    return Item(Kind.CONTEXT, text, wire=json.dumps({"role": "system", "content": [{"type": "text", "text": text}]}))


def _read(doc) -> str:
    """A file as Claude Code re-reads it after compacting, as its Read tool would show it."""

    lines = "\n".join(f"{n}\t{line}" for n, line in enumerate(doc.content.split("\n"), start=1))
    call = json.dumps({"file_path": doc.path}, separators=(",", ":"))
    return f"Called the Read tool with the following input: {call}\nResult of calling the Read tool:\n{lines}"


def _sections(text: str) -> dict[str, str]:
    """The opening system message by section (each section's text, from its first line on)."""

    starts = sorted((at, key) for key in SECTIONS if (at := text.find(key)) >= 0)
    return {key: text[at:end].strip() for (at, key), end in zip(starts, [*(a for a, _ in starts[1:]), len(text)])}


def _instructions(local: Local | None) -> str:
    """The instruction files as they read now, where Claude Code's side reports them."""

    if not (local and local.instructions):
        return ""
    return _render([(doc.path, DESCRIPTIONS.get(doc.kind, doc.kind), doc.content.rstrip("\n")) for doc in local.instructions])


def _render(files: list[tuple[str, str, str]]) -> str:
    return INSTRUCTIONS.format(files="\n\n".join(f"Contents of {p} ({d}):\n\n{c}" for p, d, c in files))
