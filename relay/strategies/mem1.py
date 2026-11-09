"""MEM1 (Zhou et al., github.com/MIT-MI/MEM1) as a Relay strategy: the model carries its memory in an
internal state it rewrites every turn, and everything before it is cleared.

As in MEM1's inference loop (`inference/data_pipelines.py`, `run_llm_loop`), each step sees only the
task, the model's previous response and that response's results: every earlier step is cleared,
whether or not the response wrote its state. The task is the user's messages (MEM1's is one question;
here the first stays whole, as MEM1 keeps its question, and the later ones as Codex keeps a user's
messages through a compaction, newest first within `user_tokens`), and MEM1's prompt (`gen_data/data_process/websearch.py`) goes before
the first of them, as MEM1 puts it before its question, worded for an agent's tools. Its internal
state is the `<IS>` block (MEM1 tags it `<think>`, which harnesses and gateways take for hidden
reasoning).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .. import settings
from ..core.ir import AGENT_KINDS, Context, Item, Kind, Request
from .base import Summarizer
from .compaction import Compaction
from .conversation import responses

# MEM1's prompt (gen_data/data_process/websearch.py, `make_prefix`), worded for an agent's own tools.
GUIDANCE = """You will complete the task through iterative reasoning, summarization, and actions with your tools.

At each step, you can see the user's messages, your previous response, and the results of its tool calls (except the first step where you will be given only the user's messages). Then, you should:

1. Conduct reasoning, and then update a concise, cumulative summary with essential information inside <IS> </IS>. This is your persistent memory and should include all important information from previous <IS> </IS> and the tool results.
2. Then choose one:
   - Call your tools (you may act repeatedly until the task is done). Their results will be returned to you at the next step.
   - Provide your final answer if the task is done.

After the tool results (or the user's message at the beginning), you should always follow the order: <IS> ... </IS> then your tool calls, or <IS> ... </IS> then your final answer.

"""


@dataclass(frozen=True)
class MEM1:
    user_tokens: int = 20_000  # the later user messages kept, as Codex's COMPACT_USER_MESSAGE_MAX_TOKENS
    name: str = "mem1"

    @classmethod
    def from_settings(cls) -> MEM1:
        return cls(user_tokens=int(settings.value("retain_user_tokens", "20000")))  # (compaction's setting)

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "guidance": GUIDANCE, "user_tokens": self.user_tokens}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items = request.current
        if starts := responses(items):  # from the newest response on; before it, the user's messages
            items = tuple(item for n, item in enumerate(items) if n >= starts[-1] or item.kind not in AGENT_KINDS)
        later = [n for n, item in enumerate(items) if item.kind is Kind.USER][1:]
        kept = Compaction(retain_user_tokens=self.user_tokens)._recent_users(tuple(items[n] for n in later))
        keep = dict(zip(later[len(later) - len(kept):], kept))  # (the newest, the oldest of them perhaps cut)
        items = tuple(keep.get(n, item) for n, item in enumerate(items) if n not in later or n in keep)
        first = next((n for n, item in enumerate(items) if item.kind is Kind.USER), None)
        if first is None:
            return Context((Item(Kind.SYSTEM, GUIDANCE.strip()), *items))
        if items[first].text.startswith(GUIDANCE):  # (the context as last sent: the prompt is there already)
            return Context(items)
        return Context((*items[:first], replace(items[first], text=GUIDANCE + items[first].text), *items[first + 1:]))
