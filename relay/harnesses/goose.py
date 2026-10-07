"""Block's Goose; provider hosts are top-level keys of config.yaml.

Compacting (1.53.0, context_mgmt), Goose summarizes the conversation (as JSON, rendered as its
summary's sections) and starts over from the summary, given as the user's, with a reply of its
own telling the model so, then the latest prompt again. It does so as a turn starts; mid-turn
only when the model rejects a request as too long.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..core.ir import Item, Kind, Native
from ..core.local import Local
from ..install import Setting
from ..prompts import GOOSE_PROMPT
from ..protocols.base import Codec
from . import keep
from .base import Harness

DEFAULTS = {
    "OPENAI_HOST": "https://api.openai.com",
    "ANTHROPIC_HOST": "https://api.anthropic.com",
    "GOOGLE_HOST": "https://generativelanguage.googleapis.com",
}


# Its summary's sections, as compaction_summary.md renders the model's JSON.
LISTS = (("user_intent", "User Intent"), ("technical_concepts", "Technical Concepts"))
LATER = (("errors_and_fixes", "Errors + Fixes"), ("problem_solving", "Problem Solving"), ("user_messages", "User Messages"),
         ("pending_tasks", "Pending Tasks"))


@dataclass(frozen=True)
class Summary(Native):
    """Its summary: the JSON after the model's analysis, rendered as its sections (the raw answer
    where there is none)."""

    def message(self, answer: str) -> str:
        summary = next((found for text in _candidates(answer) if isinstance(found := _json(text), dict) and found), None)
        return _render(summary) if summary else answer


def _candidates(text: str) -> list[str]:
    """Where its JSON may start: after each `</analysis>` (the last first), its ```json fences
    (the last first), then the text itself."""

    ends = [m.end() for m in re.finditer(re.escape("</analysis>"), text)] or [0]
    found = []
    for end in reversed(ends):
        rest = text[end:]
        found += [rest[m.end():] for m in reversed(list(re.finditer(r"```json", rest)))] + [rest]
    return [*found, text]


def _json(text: str) -> object:
    """The object a text opens with, brace-balanced (None if it does not)."""

    text = text.lstrip()
    if not text.startswith("{"):
        return None
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except ValueError:
        return None


def _render(summary: dict) -> str:
    def items(key: str) -> list[str]:
        value = summary.get(key)
        values = value if isinstance(value, list) else [] if value is None else [value]
        return [text for v in values if (text := v if isinstance(v, str) else json.dumps(v)).strip()]

    lines = ["# Conversation Summary", ""]
    for key, title in LISTS:
        if found := items(key):
            lines += [f"## {title}", *(f"- {item}" for item in found), ""]
    files = [f if isinstance(f, dict) else {"path": str(f)} for f in (summary.get("files") or [])]
    if files:
        lines.append("## Files + Code")
        for file in files:
            lines += [*([f"### {file['path']}"] if file.get("path") else []), str(file.get("summary") or "")]
            if code := (file.get("key_code") or "").rstrip("\n"):
                fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", code)), default=0))
                lines += [fence, code, fence]
            lines.append("")
    for key, title in LATER:
        if found := items(key):
            lines += [f"## {title}", *(f"- {item}" for item in found), ""]
    for key, title in (("current_work", "Current Work"), ("next_step", "Next Step")):
        if isinstance(summary.get(key), str) and summary[key].strip():
            lines += [f"## {title}", summary[key], ""]
    return "\n".join(lines).strip()


COMPACTED = ("Your context was compacted. The previous message contains a summary of the conversation so far.\n"
             "Do not mention that you read a summary or that conversation summarization occurred.\n")
GO_ON = {False: "Just continue the conversation naturally based on the summarized context.",
         True: "Continue calling tools as necessary to complete the task."}


class Goose(Harness):
    name = "goose"

    def native(self, local: Local | None, request: tuple[Item, ...] = ()) -> Native:
        return Summary(GOOSE_PROMPT, keep=keep.pending(), at_turns=True)

    def compose(self, codec: Codec, head: tuple[Item, ...], tail: tuple[Item, ...], state: tuple[Item, ...],
                mid_turn: bool, local: Local | None,
                request: tuple[Item, ...] = ()) -> tuple[tuple[Item, ...], tuple[Item, ...], tuple[Item, ...]]:
        """The summary, Goose's reply about it, and (mid-turn) the latest prompt again."""

        prompt = next((item for item in reversed(request) if item.kind is Kind.USER), None)
        systems, others = self.split_state(state)
        front = (*systems, *head, self.said(codec, Kind.ASSISTANT, COMPACTED + GO_ON[mid_turn]), *others)
        return front, tail, (self.said(codec, Kind.USER, prompt.text),) if mid_turn and prompt else ()

    def settings(self) -> list[Setting]:
        config = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser() / "goose/config.yaml"
        return [*(Setting(config, (key,), endpoint=url) for key, url in DEFAULTS.items()),
                # No switch, and values outside (0, 1) may mean the default: the window's edge.
                Setting(config, ("GOOSE_AUTO_COMPACT_THRESHOLD",), 0.99)]
