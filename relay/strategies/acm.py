"""ACM (agentic context management, github.com/lixiaochuan2020/agentic-context-management) as a Relay
strategy: the model compresses its context when it decides to, and recalls what it compressed.

As in ACM, `manage_context` compresses everything since the previous call (or since the start) up
to the message that issued it: the original messages are saved to disk as `summary_<N>.json`, and
the call's result becomes the summary, "[summary_id: N] ..."; `query_memory(N, query)` asks for
what a summary's original messages say about the query. The system prompt and the user's messages
always stay. After each tool result the model reads its context size, "[CURRENT CONTEXT TOKEN: N]"
(a note at the end of the request). The harness's shell stands in for the tools: the agent runs
`echo "[manage_context]"` and `echo "[query_memory] N :: query"`, and the echo's result is rewritten
in place. ACM's prompts are used as released (`src/prompts.py`), its guidance worded for a shell.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..core.ir import LABELS, Context, Item, Kind, Request
from ..core.tokens import approx_tokens
from .base import Summarizer
from .conversation import responses, signal

GUIDANCE = """## Context memory

Your in-context information serves as short-term memory; previously compressed segments live in long-term memory.{window}

After each tool result, the system appends "[CURRENT CONTEXT TOKEN: N]" at the end of the conversation — this tells you your current context usage. Read it; do NOT emit it yourself.

To manage your context, run the shell command `echo "[manage_context]"`, alone. The system then compresses everything in your conversation since your previous manage_context call (or since the start of the conversation if this is your first call) up to (but not including) the message that issued the call. The system prompt and the user's messages are always preserved. The original messages in the compressed range are saved to disk; the command's result becomes a summary of paths explored, reasoning, and conclusions, prefixed with "[summary_id: N]".

To retrieve detailed information from any prior summary's original messages, run `echo "[query_memory] <summary_id> :: <query>"`, alone; its result becomes the information retrieved.

Strategy:
- Watch the [CURRENT CONTEXT TOKEN: N] marker. When it climbs and your recent rounds contain dead ends or duplicates, call manage_context to compress them into a [summary_id: N] entry, freeing space. Do not write this marker in your own responses; it is an environmental signal you consume, not produce.
- If a prior summary looks relevant to a new direction, call query_memory to pull detailed content out of that summary's original messages.
- Calling manage_context or query_memory does not end the task. After they return, continue — they exist to make room for and surface more evidence, not to wrap up."""
# ACM's summarizer and recall prompts (src/prompts.py _SUMMARY_INSTRUCTION, QUERY_MEMORY_PROMPT), verbatim.
SUMMARY_INSTRUCTION = """Original question: {question}

Conversation to compress:

{conversation}

Compress the conversation above into a working-memory entry. This entry
replaces the archived messages in the agent's working context; future
calls to query_memory(summary_id, query) will search this text.

Coverage:
- Knowledge state — facts established with [docid] citations, candidate
  answers and the evidence supporting or contradicting each, hypotheses
  already ruled out with the [docid] that eliminated them, and remaining
  open sub-questions.
- Thoughts — a concise distillation of the most recent reasoning chain
  from the latest assistant turns (what direction the agent has converged
  on and why), AND the concrete next step the agent should take after
  this compression (specific search vocabulary / document to fetch /
  memory query — not a generic "continue searching").

Output exactly one <memory>...</memory> block with the two sections in
this order:

<memory>
## Knowledge state
<facts, candidates, eliminated hypotheses, open sub-questions, in short prose>

## Thoughts
<a few sentences distilling the latest reasoning thread, followed by 1-2
sentences naming the concrete next step>
</memory>

- ≤ 4096 tokens total inside the block, so try to be concise but still cover all the important information.
- Preserve identifiers verbatim — docid, named entities, dates, numbers.
- Do not enumerate every search query or docid you tried.
- After closing the chat-template-opened `</think>`, your very next token
  must be `<memory>`. Do NOT emit any preamble between `</think>` and
  `<memory>` — no `Thinking Process:` heading, no numbered planning list,
  no `Analysis:` / `Plan:` / `Let me ...` lead-in, no re-stated rules,
  no re-quoted question. Plan inside `<think>` if you need to plan;
  the visible output starts directly with `<memory>` and ends with
  `</memory>`."""
QUERY_MEMORY_PROMPT = """Saved messages under summary_id={summary_id}:

{history}

Recall request — extract content relevant to: {query}

Output format (strict):
- Put any internal reasoning inside <think>...</think> — these will be stripped.
- After </think>, write the recall as compact bullets only. No prose preamble. No restatement of the query. No address to the reader (no "the user", "the agent", "you").
- Each bullet must carry concrete identifiers (docids, URLs, numbers, names) verbatim — never paraphrase numerical evidence.
- Group into up to three sections (omit any that are empty):
  - **Relevant findings:** facts that bear on the query, with supporting docids/URLs.
  - **Dead ends:** queries / docids / hypotheses tried that produced nothing.
  - **Open / unresolved:** most promising direction still to verify.
- If nothing in the saved messages is relevant, output exactly: `(nothing relevant under summary_id={summary_id})`."""
NOTHING = ("Error: nothing to compress. There are no messages between your previous manage_context call and "
           "this one.")
SUMMARY_ID = "[summary_id: "
ANSWERED = (SUMMARY_ID, "[query_memory: summary_id=", "Error: ")  # results already rewritten
SIGNAL = re.compile(r"\[(manage_context|query_memory)\][ \t]*([^\n]*)")
MEMORY = re.compile(r"<memory>(.*?)</memory>", re.S | re.I)
THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass(frozen=True)
class ACM:
    directory: str = "/tmp/.acm"  # the saved originals, a folder per conversation
    name: str = "acm"

    @classmethod
    def from_env(cls) -> ACM:
        return cls(os.getenv("RELAY_ACM_DIR", "/tmp/.acm"))

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "directory": self.directory, "guidance": GUIDANCE,
                "summary": SUMMARY_INSTRUCTION, "query": QUERY_MEMORY_PROMPT}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        items, tokens, n = list(request.current), request.tokens, 0
        while n < len(items) - 1:
            call, result = items[n], items[n + 1]
            found = None if result.text.startswith(ANSWERED) else signal(call, result, SIGNAL)
            if found is None:
                n += 1
                continue
            if found[0] == "query_memory":
                text = self._recall(request, found[1], summarizer)
            else:  # compress from the last summary on, up to the response that called
                start = max((m + 1 for m in range(n) if _summary(items[m])), default=0)
                end = max(s for s in responses(items) if s <= n)
                dropped = [item for item in items[start:end] if item.kind is not Kind.USER]
                text = NOTHING
                if dropped:
                    number = sum(map(_summary, items[:start])) + 1
                    text = f"{SUMMARY_ID}{number}] {self._compress(request, items[:n], dropped, number, summarizer)}"
                    items[start:end] = [item for item in items[start:end] if item.kind is Kind.USER]
                    n -= len(dropped)
                    tokens -= sum(approx_tokens(item.text) for item in dropped)
            tokens += approx_tokens(text) - approx_tokens(items[n + 1].text)
            items[n + 1] = replace(items[n + 1], text=text)
            n += 2
        window = f" Your context window is {request.window} tokens." if request.window else ""
        notes = (f"[CURRENT CONTEXT TOKEN: {tokens}]",) if any(i.kind is Kind.TOOL_RESULT for i in items) else ()
        return Context((Item(Kind.SYSTEM, GUIDANCE.format(window=window)), *items), notes=notes)

    def _compress(self, request: Request, before: list[Item], dropped: list[Item], number: int,
                  summarizer: Summarizer) -> str:
        """ACM's summary of `dropped`, whose original messages it saves first."""

        messages = [{"role": LABELS.get(item.kind, item.kind.value), "content": item.text} for item in dropped]
        path = Path(self.directory) / request.conversation / f"summary_{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        question = "\n\n".join(item.text for item in before if item.kind is Kind.USER)
        conversation = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        text = THINK.sub("", summarizer.summarize(0, SUMMARY_INSTRUCTION.format(question=question,
                                                                                conversation=conversation))).strip()
        memory = MEMORY.search(text)  # ACM's parse: the <memory> block, else the whole answer
        return memory.group(1).strip() if memory and len(memory.group(1).strip()) >= 50 else text

    def _recall(self, request: Request, text: str, summarizer: Summarizer) -> str:
        number, _, query = text.partition("::")
        number = (re.findall(r"\d+", number) or [""])[0]
        if not number or not query.strip():
            return "Error: missing required parameters (summary_id, query)."
        path = Path(self.directory) / request.conversation / f"summary_{number}.json"
        if not path.exists():
            return f"Error: summary_id {number} not found."
        history = "\n".join(f"[{m['role']}] {m['content']}" for m in json.loads(path.read_text(encoding="utf-8")))
        prompt = QUERY_MEMORY_PROMPT.format(summary_id=number, history=history, query=query.strip())
        return f"[query_memory: summary_id={number}]\n\n{THINK.sub('', summarizer.summarize(0, prompt)).strip()}"


def _summary(item: Item) -> bool:
    return item.kind is Kind.TOOL_RESULT and item.text.startswith(SUMMARY_ID)
