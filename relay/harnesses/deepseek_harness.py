"""DeepSeek Harness (`dsh`); providers are plugin entries in the home-level cordis.patch.yml.

The DeepSeek route takes a `baseURL` (its DEEPSEEK_BASE_URL may only be exported, never
written to a file). Of the pi-ai routes (OpenAI, Google, Anthropic, gateways) only those the
user already declared are wrapped: a patch replaces a plugin's whole config, so adding a
pi-ai entry could hide routes configured in other layers.

Compacting (0.2.0-rc.2, compaction-basic), dsh keeps the turn in progress and the newest turns
before it (a fifth of what it compacts at, by default), and summarizes what came before into a
checkpoint; its workspace
instructions, as they read now, are restated after the messages kept.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting, yaml_keys
from ..prompts import DSH_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import NEVER, Harness

DEEPSEEK = ("[id=llm-deepseek]", "config", "baseURL")
ROUTES = ("[id=llm-pi-ai]", "config", "providers")
DEFAULTS = {  # pi-ai catalog endpoints, for routes without an explicit baseURL
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com",
}


RUNTIME = "Current runtime context."
CHECKPOINT = ("This is an automatically generated checkpoint condensing an earlier span of the conversation to free up "
              "context. Treat the captured context as established background and build on it without restating it. "
              "Continue the task directly from the messages that follow, without acknowledging this checkpoint.\n\n"
              "<compacted-summary>")
WORKSPACE = "The following workspace instructions may be relevant to your work."
FILE = re.compile(r"(Instructions from: (\S+)\n\n)(.*?)(\n*</system-reminder>)", re.S)
CONTENT = re.compile(r"Use the following content instead of the previously loaded instructions from this file\.\n\n(.*?)\n*</system-reminder>", re.S)
UPDATED = re.compile(r"Updated instructions from: (\S+)")
# Sub-agents reporting back (like Codex's notifications): a message, and their closing one.
NOTICE = re.compile(r"Agent \S+ sent a message:|Background subagent \S+ finished")


class DeepSeekHarness(Harness):
    name = "deepseek_harness"

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and (item.text.lstrip().startswith(RUNTIME) or NOTICE.match(item.text.lstrip())):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Native(DSH_PROMPT, CHECKPOINT, "</compacted-summary>", keep.turns(0.2))

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The summary; after the messages kept, the workspace instructions as they read now."""

        workspace = next((item for item in state if WORKSPACE in item.text), None)
        rest = [item for item in state if item is not workspace and not item.text.lstrip().startswith(RUNTIME)
                and not UPDATED.search(item.text)]
        systems, others = self.split_state(tuple(rest))
        front = (*systems, *head, *others)
        if workspace is None:
            return front, tail, ()
        text = workspace.text
        for update in (item for item in request if UPDATED.search(item.text) and CONTENT.search(item.text)):
            path, content = UPDATED.search(update.text).group(1), CONTENT.search(update.text).group(1)
            text = FILE.sub(lambda m: f"{m[1]}{content}{m[4]}" if m[2] == path else m[0], text)
        return front, tail, (Item(Kind.CONTEXT, text),)

    def state_key(self, item: Item) -> str | None:
        """Runtime snapshots ("This snapshot supersedes earlier runtime-context snapshots") and
        instruction files restated after they changed."""

        if item.text.lstrip().startswith(RUNTIME):
            return "runtime"
        if match := UPDATED.search(item.text):
            return f"instructions:{match.group(1)}"
        return None

    def settings(self) -> list[Setting]:
        patch = Path(os.getenv("DSH_HOME", "~/.dsh")).expanduser() / "cordis.patch.yml"
        deepseek = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/anthropic"
        return [Setting(patch, DEEPSEEK, endpoint=deepseek)] + [
            Setting(patch, (*ROUTES, route, "baseURL"), endpoint=DEFAULTS.get(route, ""))
            for route in yaml_keys(patch, ROUTES)
        ] + [
            Setting(patch, ("[id=compaction-basic]", "config", "auto"), False),
            # It trims old tool results in place under token pressure.
            Setting(patch, ("[id=tool-result-pruner]", "config", "thresholdChars"), NEVER),
        ]
