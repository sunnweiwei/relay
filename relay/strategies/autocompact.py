"""AutoCompact (Zhang et al., autocompact.github.io) as a Relay strategy: the model decides when to
compact its context.

As in AutoCompact, at a transition between phases of the task the agent calls `compact()`, which
answers with an "# Auto Context Summary" of the working state (objective, conclusions, modified
files, verification status, next action), and the agent goes on from the original task, its most
recent turn and the summary; fixed-threshold compaction stays as the fallback. The harness's shell
stands in for the tool: the agent runs `echo "[compact]"` (the guidance says when and how), the same
model, asked after the conversation as it stands (its prompt cache reused), writes the summary, and
the echo's result becomes it, so the agent sees that it compacted.
AutoCompact released no prompts; the guidance and the summary prompt follow its description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.ir import Context, Item, Kind, Request
from .base import Summarizer
from .compaction import Compaction
from .conversation import responses, signal

GUIDANCE = """## Compaction

Compact your context when the state of the task, not its length, says it would help: at a transition between phases of the task (for example, once the issue is localized and before you begin the edit), when the earlier exploration (files inspected, hypotheses tested, failed attempts) is no longer needed. To compact, run the shell command `echo "[compact]"`, alone. Then you write an # Auto Context Summary of your working state, and from then on you see the user's messages, your most recent turn and that summary instead of the full history."""
PROMPT = """You called compact(). Write an # Auto Context Summary that replaces the history above: only the original task, your most recent turn and this summary stay, so keep the working state needed for the next phase of the task, all of it: the objective, the conclusions reached (paths ruled out included), the files modified, the verification status and the next action, with exact names and values. Leave out exploration details that are no longer needed. Start with the line "# Auto Context Summary"."""
HEADER = "# Auto Context Summary"
SIGNAL = re.compile(r"\[(compact)\][ \t]*([^\n]*)")


@dataclass(frozen=True)
class AutoCompact:
    fallback: Compaction = field(default_factory=Compaction)  # the fixed-threshold limit, if the model never compacts
    name: str = "autocompact"

    @classmethod
    def from_env(cls) -> AutoCompact:
        return cls(Compaction.from_env())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "fallback": self.fallback.fingerprint(), "guidance": GUIDANCE, "prompt": PROMPT}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items, guidance = request.current, Item(Kind.SYSTEM, GUIDANCE)
        calls = [n for n in range(len(items) - 1)
                 if not items[n + 1].text.startswith(HEADER) and signal(items[n], items[n + 1], SIGNAL)]
        cut = min((b for b in request.boundaries if calls and b >= calls[-1] + 2), default=None)
        if cut is None:
            context = self.fallback.plan(request, summarizer)
            return Context((guidance, *(context.items if context else items)))
        n = calls[-1]  # the last call counts; the history before it is summarized
        starts = [start for start in responses(items) if start <= n]
        recent = starts[-2] if len(starts) > 1 else starts[-1]  # the turn before the call's, unless a user spoke since
        if any(item.kind in (Kind.USER, Kind.SUMMARY) for item in items[recent:starts[-1]]):
            recent = starts[-1]
        summary = summarizer.summarize(cut, PROMPT).strip()
        summary = summary if summary.startswith(HEADER) else f"{HEADER}\n\n{summary}"
        users = (item for item in items[:recent] if item.kind is Kind.USER)
        return Context((guidance, *users, *items[recent:n + 1], replace(items[n + 1], text=summary), *items[n + 2:]))
