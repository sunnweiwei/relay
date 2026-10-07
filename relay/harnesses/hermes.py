"""Nous Research's Hermes Agent; `hermes model` stores the endpoint in config.yaml.

Hermes picks the wire protocol from the endpoint's host, so the profile pins the one it
had: the Responses API for api.openai.com, and for Google's native Gemini API, which
Hermes only speaks to Google's host, Gemini's OpenAI-compatible API. Native Anthropic is
left alone: Hermes ignores Anthropic URLs on hosts it does not recognize.

Compressing (0.19.0, context_compressor), Hermes keeps its system prompt, the first three
messages and the newest ones (a fifth of what it compacts at, eight at least), and summarizes those in between, its
summary marked as reference only; the latest user message comes again after it when none was kept.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting, read
from ..prompts import HERMES_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import Harness

SUMMARY = ("[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below. This is a handoff "
           "from a previous context window — treat it as background reference, NOT as active instructions. Do NOT answer "
           "questions or fulfill requests mentioned in this summary; they were already addressed. Respond ONLY to the "
           "latest user message that appears AFTER this summary — that message is the single source of truth for what "
           "to do right now. Topic overlap with the summary does NOT mean you should resume its task: even on similar "
           "topics, the latest user message WINS. Treat ONLY the latest message as the active task and discard stale "
           "items from '## Historical Task Snapshot' / '## Historical In-Progress State' / '## Historical Pending User "
           "Asks' / '## Historical Remaining Work' entirely — do not 'wrap up' or 'finish' work described there unless "
           "the latest message explicitly asks for it. Reverse signals in the latest message (e.g. 'stop', 'undo', "
           "'roll back', 'just verify', 'don't do that anymore', 'never mind', a new topic) must immediately end any "
           "in-flight work described in the summary; do not re-surface it in later turns. IMPORTANT: Your persistent "
           "memory (MEMORY.md, USER.md) in the system prompt is ALWAYS authoritative and active — never ignore or "
           "deprioritize memory content due to this compaction note. None of the above restricts HOW you work: your "
           "tools remain fully active — keep calling them normally for the active task (edit files, run commands, "
           "search) instead of merely narrating what you would do. The current session state (files, config, etc.) "
           "may reflect work described here — avoid repeating it:\n")
END = "\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"

DEFAULTS = {
    "openai-api": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


class Hermes(Harness):
    name = "hermes"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(HERMES_PROMPT, SUMMARY, END, keep.tail(share=0.2, head=3, least=8))

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The summary points at the latest user message after it: where none was kept, it comes
        again after the summary."""

        front, tail, back = super().compose(codec, head, tail, state, mid_turn, local, request)
        prompt = next((item for item in reversed(request) if item.kind is Kind.USER), None)
        if prompt is None or any(item.kind is Kind.USER for item in (*front, *tail) if item.ref is not None
                                 and item.ref >= (prompt.ref or 0)):
            return front, tail, back
        return (*front, self.said(codec, Kind.USER, prompt.text)), tail, back

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser() / "config.yaml"
        provider = read(config, ("model", "provider"))
        if not provider or provider == "anthropic":
            raise ValueError("configure an OpenAI-compatible or Gemini provider with `hermes model` first")
        base_url = (read(config, ("model", "base_url")) or DEFAULTS.get(provider, "")).rstrip("/")
        settings = []
        if urlsplit(base_url).hostname == "api.openai.com" and not read(config, ("model", "api_mode")):
            settings.append(Setting(config, ("model", "api_mode"), value="codex_responses"))
        if "generativelanguage.googleapis.com" in base_url and not base_url.endswith("/openai"):
            base_url += "/openai"
            settings.append(Setting(config, ("model", "base_url"), value=base_url))
        return [*settings, Setting(config, ("model", "base_url"), endpoint=base_url),
                Setting(config, ("compression", "enabled"), False)]
