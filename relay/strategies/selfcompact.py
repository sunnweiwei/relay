"""Self-Compact (Li et al., arXiv 2606.23525) as a Relay strategy: a rubric decides when the model
summarizes its own trajectory.

As in Self-Compact's Algorithm 1, at a probe the rubric is appended to the conversation (its
prompt cache reused) and the model answers C1 CLOSED-UNIT, C2 SUMMARIZABLE, C3 PROGRESS and N1
STUCK; only C1 = C2 = C3 = Y and N1 = N fires the summarizer, also appended to the conversation,
and the summary then replaces the trajectory: the model goes on from the user's messages and its
summary (introduced as every Relay summary is, where the paper follows it with "continue"). A
CONTINUE verdict leaves the conversation as it was. Probes come every `period` model responses
once the prompt passes `gate` of the budget, as the CLM paper runs Self-Compact (every two turns
past 37%); past `backstop` the summary is made without asking, Self-Compact's token backstop. Both
prompts are the paper's (Appendix B, the search agent's), verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.conversation import pending_cut
from ..core.ir import Context, Item, Kind, Request
from ..prompts import SUMMARY_PREFIX
from .base import Summarizer
from .compaction import DEFAULT_WINDOW, HARD_LIMIT
from .conversation import responses

RUBRIC = """You are about to decide whether to compress your conversation history into a summary that REPLACES the full history above. After compression, you continue research from only [system, original_question, assistant_summary, user_continue]. Compression is irreversible: anything not preserved in the summary is gone.

Compression is safe ONLY when ALL FOUR of the following hold:
(C1) the trajectory has reached a closed unit (not mid-thought),
(C2) the essential information is reducible to 3–5 cite-able facts without loss,
(C3) something has progressed since the last compression,
(N1) you are NOT currently stuck in a way summarization would mask.

Answer C1, C2, C3, N1 honestly. Each Y answer requires verbatim evidence quoted from the trajectory above; answers without evidence default to N.

C1 CLOSED-UNIT: The most recent assistant message is a closed unit — a completed tool call whose result is now visible, or a completed sub-analysis with a clear stopping point. It is NOT mid-sentence reasoning ("Let me now check...", "I should next look at..."), and not a half-formulated query. If Y, quote the closing fragment of the last assistant message. If N, quote the open fragment that shows the trajectory is mid-thought.

C2 SUMMARIZABLE: You can write 3–5 essential facts (with verbatim citations from the trajectory) that future-you needs to continue research after compression. Each fact must be a single concrete statement: a name, date, URL, quoted claim, or resolved sub-question. Answer N if the trajectory's value is dispersed across many small inferences (e.g., a list of dead-end queries needed to avoid retries, negative results that constrain hypothesis space) that would be lost without the dispersal. If Y, list the 3–5 facts numbered, each with a verbatim citation in quotes, separated by "|". If N, name in one sentence the class of information that would be lost.

C3 PROGRESS: Since the most recent compression (or since the start of the conversation if none), you have either obtained a new concrete fact (name, date, URL, or quoted claim) OR refined the sub-question being pursued. If Y, name the new fact or refined sub-question. If N, state that you are returning the same state you compressed from.

N1 STUCK: At least 3 of your last 4 search queries returned no new URL or fact (i.e., were duplicates or returned already-known content). If you have made fewer than 4 searches total, answer N. If Y, name 1 distinct strategy you have NOT yet tried (different tool, different query type, different angle on the question). If N, name one new URL or fact obtained recently.

Output: exactly 4 lines, no preamble or trailing text.
C1: Y/N -- <evidence>
C2: Y/N -- <if Y: 1. fact "citation" | 2. fact "citation" | 3. fact "citation"; if N: <class of info lost>>
C3: Y/N -- <evidence>
N1: Y/N -- <evidence>"""
# The webresummer summarizer of Wu et al., as Self-Compact uses it.
SUMMARIZER = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history above and the user's original question to provide a summary that will REPLACE the full conversation history when you continue.

Task Guidelines
1. Information Analysis:
• Carefully analyze the conversation history to identify truly useful information.
• Focus on information that directly contributes to answering the question.
• Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
• If information is missing or unclear, do NOT include it in your summary.
2. Summary Requirements:
• Extract only the most relevant information that is explicitly present in the conversation.
• Synthesize information from multiple exchanges when relevant.
• Only include information that is certain and clearly stated in the conversation.
• Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation."""
VERDICT = re.compile(r"\b(C1|C2|C3|N1)\b(?:\s+[A-Z-]+)?[^A-Za-z0-9\n]*([YN])\b")  # "C1: Y -- ...", "**C1**: Y"


@dataclass(frozen=True)
class SelfCompact:
    budget: int | None = None  # tokens; None: the model's context window
    gate: float = 0.37  # probe once the prompt passes this share of the budget
    period: int = 2  # every this many model responses since the last summary
    backstop: float = HARD_LIMIT  # summarize without asking past this share
    name: str = "selfcompact"

    @classmethod
    def from_env(cls) -> SelfCompact:
        return cls()

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "budget": self.budget, "gate": self.gate, "period": self.period,
                "backstop": self.backstop, "rubric": RUBRIC, "summarizer": SUMMARIZER}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        budget, items = self.budget or request.window or DEFAULT_WINDOW, request.current
        last = max((n for n, item in enumerate(items) if item.kind is Kind.SUMMARY), default=-1)
        rounds = sum(start > last for start in responses(items))
        forced = request.force or request.tokens >= self.backstop * budget
        due = request.tokens >= self.gate * budget and rounds and rounds % self.period == 0
        if not (forced or due) or (cut := pending_cut(request)) is None:
            return None
        if not forced and not _fires(summarizer.summarize(cut, RUBRIC)):
            return None
        summary = Item(Kind.SUMMARY, f"{SUMMARY_PREFIX}\n{summarizer.summarize(cut, SUMMARIZER)}")
        return Context((*(item for item in items[:cut] if item.kind is Kind.USER), summary, *items[cut:]))


def _fires(answer: str) -> bool:
    """Self-Compact's fire rule: COMPRESS iff C1 = C2 = C3 = Y and N1 = N (a missing answer is N)."""

    verdict: dict[str, str] = {}
    for label, value in VERDICT.findall(answer):
        verdict.setdefault(label, value)
    return all(verdict.get(label) == "Y" for label in ("C1", "C2", "C3")) and verdict.get("N1", "N") == "N"
