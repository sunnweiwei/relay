"""MEM1 (Zhou et al., github.com/MIT-MI/MEM1) as a Relay strategy: the model carries its memory in an
internal state it rewrites every turn, and everything before it is cleared.

As in MEM1's inference loop, each turn the model consolidates what it knows, its previous internal
state and the latest observation, into a new internal state, then acts; the next turn sees only the
task, that response and its result. Here the task is the user's messages (they always stay), the
internal state is the `<IS>` block MEM1's prompt asks for (MEM1's code tags it `<think>`, which
harnesses and gateways take for hidden reasoning), and the context is cut back to the latest
response holding one, so the model always continues from its newest memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.ir import AGENT_KINDS, Context, Item, Kind, Request
from .base import Summarizer
from .conversation import responses

# MEM1's prompt (gen_data/data_process/websearch.py), worded for an agent's own tools.
GUIDANCE = """## Memory

At each step you see the user's messages, your previous response with its internal state, and the results of its actions; everything before is cleared. In every response, first conduct reasoning, and then update a concise, cumulative summary with essential information inside <IS> </IS>, as visible text before any tool call. This is your persistent memory and should include all important information from your previous <IS> </IS> and the results since (facts found, exact values, files changed, progress on the task, what remains). Then act, or give your final answer. Write it in every response, even one that only calls a tool and even where other instructions ask for brief replies: what it leaves out is gone at the next step."""


@dataclass(frozen=True)
class MEM1:
    name: str = "mem1"

    @classmethod
    def from_env(cls) -> MEM1:
        return cls()

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "guidance": GUIDANCE}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items = request.current
        # The model's text is its own item, or, where a message carries text and calls together (Chat
        # Completions, Anthropic, Gemini), part of the call's.
        states = [n for n, item in enumerate(items) if item.kind in (Kind.ASSISTANT, Kind.TOOL_CALL) and "<IS>" in item.text]
        if states:  # from the response holding the newest internal state on; before it, the user's messages
            start = max(s for s in responses(items) if s <= states[-1])
            items = tuple(item for n, item in enumerate(items) if n >= start or item.kind not in AGENT_KINDS)
        return Context((Item(Kind.SYSTEM, GUIDANCE), *items))
