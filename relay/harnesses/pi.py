"""pi coding agent (earendil-works/pi). Speaks each provider's own API.

Compacting (1.0.4), pi keeps the newest messages within `keepRecentTokens` (20k by default) and
summarizes what came before; it starts over from its system prompt as it reads now (each section
update merged in), the summary, and the messages kept.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import PI_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import Harness


SECTION = re.compile(r'Updated system prompt section "([^"]+)"')
SUMMARY = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
KEEP = 20_000  # keepRecentTokens
SPLIT = "**Turn Context (split turn):**"


class Pi(Harness):
    name = "pi"

    def state_key(self, item: Item) -> str | None:
        """A resumed session appends its system prompt anew, with instruction files as they are now;
        a changed section alone comes as an update of that section (the prompt it updates stays)."""

        if item.kind is not Kind.SYSTEM:
            return None
        section = SECTION.match(item.text)
        return f"section:{section.group(1)}" if section else "system"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(PI_PROMPT, SUMMARY, "\n</summary>", keep.tail(tokens=KEEP))

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The system prompt with its sections as they read now (updates among what it kept
        included), the summary: when it cuts the first turn, as that turn's context."""

        *before, summary = head
        if tail and tail[0].kind is not Kind.USER and SPLIT not in summary.text:
            start = next((n for n, item in enumerate(request) if item.ref == tail[0].ref), len(request))
            users = [item for item in request[:start] if item.kind is Kind.USER]
            if len(users) == 1:  # (an earlier turn would have a summary of its own)
                inner = summary.text[len(SUMMARY):]
                summary = replace(summary, text=f"{SUMMARY}No prior history.\n\n---\n\n{SPLIT}\n\n{inner}")
        head = (*before, summary)
        updates = [item for item in tail if item.kind is Kind.SYSTEM and SECTION.match(item.text)]
        tail = tuple(item for item in tail if item not in updates)
        systems = [item for item in state if item.kind is Kind.SYSTEM]
        prompt = next((item for item in systems if not SECTION.match(item.text)), None)
        if prompt is None:
            return (*systems, *head), tail, ()
        text = prompt.text
        for update in (item for item in [*systems, *updates] if SECTION.match(item.text)):
            name = SECTION.match(update.text).group(1)
            block = re.search(rf"<{re.escape(name)}>.*?</{re.escape(name)}>", update.text, re.S)
            if block:
                text = re.sub(rf"<{re.escape(name)}>.*?</{re.escape(name)}>", lambda _: block[0], text, flags=re.S)
        others = [item for item in state if item.kind is not Kind.SYSTEM]
        return (replace(prompt, text=text), *others, *head), tail, ()

    def matches(self, headers: Mapping[str, str]) -> bool:
        return headers.get("user-agent", "").startswith(("pi/", "pi ("))

    def settings(self) -> list[Setting]:
        # A provider entry with only `baseUrl` keeps pi's built-in models and logins.
        models = Path(os.getenv("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "models.json"
        return [
            Setting(models.with_name("settings.json"), ("compaction", "enabled"), False),
            Setting(models, ("providers", "openai", "baseUrl"), endpoint="https://api.openai.com/v1"),
            Setting(models, ("providers", "anthropic", "baseUrl"), endpoint="https://api.anthropic.com"),
            Setting(models, ("providers", "google", "baseUrl"),
                    endpoint="https://generativelanguage.googleapis.com/v1beta"),
        ]
