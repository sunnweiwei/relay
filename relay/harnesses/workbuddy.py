"""Tencent's WorkBuddy (desktop) and CodeBuddy Code (CLI), which share one engine.

Custom models are entries of models.json whose `url` is the full chat/completions
endpoint; built-in models go through Tencent's own service and cannot be wrapped.
A project-level .workbuddy/.codebuddy models.json takes precedence over the user's.

Compacting (CodeBuddy Code 2.161.4, its emergency compaction), it summarizes the whole session
and starts over from its first message's reminders with the summary in one of its own; mid-turn,
a prompt telling the model to go on follows.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import WORKBUDDY_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import REMINDER, Harness

SUMMARY = ("<system-reminder data-role=\"compact-summary\">\nThis session is being continued from a previous "
           "conversation that ran out of context. The summary below covers the earlier portion of the conversation."
           "\n\nSummary:\n")
TRANSCRIPT = ("\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or "
              "content you generated), read the full transcript at: {path}")
ADDRESS = "\n\nPlease address this message and continue with your tasks.\n</system-reminder>"
WORKING = re.compile(r"Working directory: (\S+)")
CONTINUE = ("<user_query>Please continue with the conversation based on the summarized context above. Maintain the same "
            "level of detail and helpfulness as before the summarization.</user_query>")


@dataclass(frozen=True)
class Summary(Native):
    """Its summary in its reminder: the answer's `<summary>`, or the whole answer as its history summary."""

    def message(self, answer: str) -> str:
        inner = re.search(r"<summary>(.*?)</summary>", answer, re.S)
        text = inner[1].strip() if inner else f"<conversation_history_summary>\n{answer.strip()}\n</conversation_history_summary>"
        return f"{self.prefix}{text}{self.suffix}"


class WorkBuddy(Harness):
    name = "workbuddy"
    # Memory and rules files, re-rendered in the first user message around its <user_query>.
    injected = (REMINDER, re.compile(r"<always_applied_workspace_rules>.*?</always_applied_workspace_rules>", re.S))
    config_env, config_dir = "WORKBUDDY_CONFIG_DIR", "~/.workbuddy"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        """Its summary points at its transcript: the session's, in its project's folder."""

        path = local.transcript if local else None
        cwd = next((m[1] for item in request if item.kind is Kind.SYSTEM if (m := WORKING.search(item.text))), None)
        if not path and local and local.session and cwd:
            home = Path(os.getenv(self.config_env) or self.config_dir).expanduser()
            path = str(home / "projects" / cwd.strip("/").replace("/", "-") / f"{local.session}.jsonl")
        return Summary(WORKBUDDY_PROMPT, SUMMARY, (TRANSCRIPT.format(path=path) if path else "") + ADDRESS, keep.pending())

    def session(self, headers: Mapping[str, str], body: Mapping) -> tuple[str, str] | None:
        session = headers.get("x-conversation-id")
        first = next((m.get("content") for m in body.get("messages") or [] if m.get("role") == "user"), "")
        return (session, first if isinstance(first, str) else "") if session else None

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """Its first message's reminders with the summary in one message; mid-turn, the prompt to
        go on."""

        *before, summary = head
        first = next((item for item in request if item.kind is Kind.USER), None)
        opening = first.text.split("<user_query>")[0].rstrip() if first else ""
        merged = Item(Kind.SUMMARY, summary.text, wire=json.dumps(codec.write(Item(
            Kind.USER, f"{opening}\n\n{summary.text}" if opening else summary.text))))
        systems, others = self.split_state(state)
        front = (*systems, *before, merged, *others)
        return front, tail, (self.said(codec, Kind.USER, CONTINUE),) if mid_turn else ()

    def settings(self) -> list[Setting]:
        config = Path(os.getenv(self.config_env) or self.config_dir).expanduser() / "models.json"
        models = json.loads(config.read_text()).get("models", []) if config.exists() else []
        settings = [
            Setting(config, ("models", f"[id={model['id']}]", "url"), endpoint="")
            for model in models
            if model.get("id") and str(model.get("url", "")).startswith("http")  # not a ${VAR} reference
        ]
        if not settings:
            raise ValueError(f"add a custom model with a `url` to {config} first")
        # Automatic compaction off, and the forced one (at 92% of a model's contextWindow or
        # maxInputTokens, the only one this build runs) moved to the window's edge.
        user = config.with_name("settings.json")
        return [*settings, Setting(user, ("autoCompactEnabled",), False),
                Setting(user, ("env", "CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE"), "100")]


class CodeBuddy(WorkBuddy):
    name = "codebuddy"
    config_env, config_dir = "CODEBUDDY_CONFIG_DIR", "~/.codebuddy"
