"""ACM (agentic context management, github.com/lixiaochuan2020/agentic-context-management) as a Relay
strategy: the model compresses its context when it decides to, and recalls what it compressed.

`manage_context` compresses everything since the previous call (or since the start) through the
response that issued it, as ACM's runner does by default (`mc_drop_call_message`): the original
messages are saved to disk as `summary_<N>.json`, and only the summary, "[summary_id: N] ...", takes
their place, as a user message (the release's client sends that orphaned tool message as user text
after the question). `query_memory(N, query)` asks for what a summary's original messages say about
the query. The system prompt and the original question (the first user message) always stay; as in
ACM's runner, whose compression boundary starts right after them, everything later is compressed
alike, the user's later messages included. Both prompts go to the task's model as ACM's runner sends
them: one user message, no system prompt, no tools, the messages as `[role] content` (no calls, no
reasoning); a summary without a clean `<memory>` block is asked for once more. ACM's output caps
(4,096 tokens for a summary, 2,048 for a recall) are not set: `Summarizer.complete` takes none.
The last tool result (or summary) ends with "[CURRENT CONTEXT TOKEN: N]", N the request's tokens plus
ACM's 16,384-token output reserve (after a message answering several calls it is a note: rewriting
the message would merge their results); from 95% of the window the request ends with ACM's
STOP_PROMPT instead, and the harness, which Relay cannot stop, goes on as the model answers it. The
runner's STOP ends its one question; in a conversation it ends the task at hand, the user's turn: a
new user message is answered without it, so the model can compress first, and gets it again once it
has answered within that turn and is still past 95%. The
harness's shell stands in for the tools: the agent runs `echo "[manage_context]"` and
`echo "[query_memory] N :: query"`, alone. ACM's system prompt and tool descriptions are used as
released, without the search benchmark's parts (its role line, search advice and answer format),
each tool called by its echo.
Unlike the release, which serializes messages as text only, a compressed range's media go with it:
each is marked `[image N]` where it was, attached to the summary request, and saved with the originals,
so query_memory sees them too.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..core.ir import LABELS, Context, Item, Kind, Media, Request
from ..core.media import marks
from ..core.tokens import approx_tokens, item_tokens
from .base import Summarizer
from .conversation import answered, ignored, signalled, step
from .. import settings

# ACM's system prompt (src/prompts.py _HEADER_WITH_MEMORY, _CONTEXT_WINDOW_LINE, _MEMORY_MANAGEMENT_STRATEGY)
# and tool descriptions (src/tools.py MANAGE_CONTEXT_TOOL, QUERY_MEMORY_TOOL), verbatim but for the
# search benchmark's parts; each tool called by its echo.
GUIDANCE = """## Context memory

Your in-context information serves as short-term memory; previously compressed segments live in long-term memory.

After each tool result, the system appends "[CURRENT CONTEXT TOKEN: N]" at the end of the tool response — this tells you your current context usage. Read it; do NOT emit it yourself.

The "manage_context" tool takes no arguments. When you call it, the system compresses everything in your conversation since your previous manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued the call. The system prompt and the original question are always preserved. The original messages in the compressed range are saved to disk; the tool returns a summary of paths explored, reasoning, and conclusions, prefixed with "[summary_id: N]".

Use the "query_memory" tool to retrieve detailed information from any prior summary's original messages by referencing the summary_id.{window}

Strategy:
- Watch the [CURRENT CONTEXT TOKEN: N] marker at the end of each tool result. When it climbs and your recent rounds contain dead ends or duplicates, call "manage_context" to compress them into a [summary_id: N] entry, freeing space. Do not write this marker in your own responses; it is an environmental signal you consume, not produce.
- If a prior summary looks relevant to a new search direction, call the "query_memory" tool to pull detailed content out of that summary's original messages.
- Calling the "manage_context" tool or the "query_memory" tool does not end the investigation. After these tools return, continue searching — they exist to make room for and surface more evidence, not to wrap up. Only commit to a final answer when you have confident evidence.

Both tools are shell commands here: call one by running its command, alone (no other tool call beside it).

### manage_context: `echo "[manage_context]"`

Compress your working memory. Call this when context is filling up with dead ends, duplicates, or detail you no longer need verbatim.

The system automatically picks the range to compress: everything since your last manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued this tool call. The system prompt and the original question are always preserved. The original messages in the compressed range are saved to disk as summary_{summary_id}.json (retrievable later with query_memory(summary_id, query)), and a fresh summary — focused on paths explored, reasoning, and conclusions — is returned as the tool result.

The summary text is prefixed with "[summary_id: N]" so you can refer to it directly in subsequent reasoning and pull the raw content back via query_memory(summary_id=N, ...).

This tool takes no arguments.

### query_memory: `echo "[query_memory] <summary_id> :: <query>"`

Retrieve detailed information from a previously compressed summary. Each manage_context call returns a summary prefixed with [summary_id: N] and saves the original messages to disk as summary_{N}.json. This tool loads summary_{summary_id}.json and uses an LLM to extract information matching your query from the original (uncompressed) content.

DO NOT CALL THIS TOOL until at least one manage_context call has produced a summary_id.

- summary_id (integer, required): The summary_id (as shown in the [summary_id: N] prefix of a prior manage_context summary) whose original content should be searched.
- query (string, required): What specific information to extract from the original messages of that summary."""
WINDOW = """

Your context window is {window} tokens. Plan your searches accordingly — once context is full the conversation ends immediately. You must provide your answer in the required format before reaching the context limit."""
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
# ACM's end-of-window prompt (src/prompts.py STOP_PROMPT), verbatim: the runner sends it as it is, its
# doubled braces too.
STOP_PROMPT = """The investigation must now conclude. You have reached the context window limit.

IMPORTANT: Do NOT call any tools. Do NOT attempt any further searches. You MUST respond with text only.

Please review ALL the search results and reasoning from your previous rounds, then provide your best final answer.

You MUST respond in this exact format (text only, no tool calls):
<explanation>{{your explanation summarizing the key evidence}}</explanation>
<answer>{{your succinct, final answer}}</answer>
<confidence>{{your confidence score between 0 and 100}}</confidence>"""
RESERVE = 16_384  # ACM's output reservation (src/configs/agent.py max_new_tokens), part of the marker's N
STOP = 0.95  # of the window: where ACM's runner sends STOP_PROMPT
SUMMARY_ID = "[summary_id: "
ANSWERED = ("[query_memory: summary_id=", "Error: ")  # a recall already answered
HIDDEN = ("tool_call", "reasoning")  # not a message's content, which is all ACM serializes
SIGNAL = re.compile(r"\[(manage_context|query_memory)\][ \t]*([^\n]*)")
MEMORY = re.compile(r"<memory>(.*?)</memory>", re.S | re.I)
THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
MARKER = re.compile(r"\n\n\[CURRENT CONTEXT TOKEN: \d+\]\Z")


@dataclass(frozen=True)
class ACM:
    directory: str = "/tmp/.acm"  # the saved originals, a folder per conversation
    name: str = "acm"

    @classmethod
    def from_settings(cls) -> ACM:
        return cls(settings.value("acm_dir", "/tmp/.acm"))

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "directory": self.directory, "guidance": GUIDANCE,
                "summary": SUMMARY_INSTRUCTION, "query": QUERY_MEMORY_PROMPT}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools:
            return None
        own = {item.ref: item.text for item in request.history if item.ref is not None}
        items = [_unmarked(item, own) for item in request.current]  # (the last request's marker taken off)
        tokens = request.tokens - sum(item_tokens(a) - item_tokens(b) for a, b in zip(request.current, items) if a is not b)
        notes, n = (), 0
        while (found := step(items, n)) is not None:
            start, first, end = found
            if (signal := signalled(items[start:first], items[first:end], SIGNAL)) is None:
                n = end
                continue
            name, said, k, alone = signal
            if not alone:  # beside other calls
                notes += ignored(items, None if k is None else first + k, end, name)
            if not alone or items[first].text.startswith(ANSWERED):
                n = end
                continue
            if name == "query_memory":
                text = self._recall(request, said, summarizer)
                tokens += approx_tokens(text) - approx_tokens(items[first].text)  # (its only call: the one result)
                items[first], n = replace(items[first], text=text), end
                continue
            # From the last summary on (at first, from after the question) through the response that
            # called: the summary takes their place, the call and its result gone.
            question = next((m + 1 for m, item in enumerate(items) if item.kind is Kind.USER), 0)
            since = max([question, *(m + 1 for m in range(start) if _summary(items[m]))])
            number = sum(map(_summary, items[:since])) + 1
            text = self._compress(request, items[:since], items[since:first], number, summarizer)
            summary = Item(Kind.USER, f"{SUMMARY_ID}{number}] {text}")
            tokens += item_tokens(summary) - sum(map(item_tokens, items[since:end]))
            items[since:end], n = [summary], since + 1
        asked = max((m for m, item in enumerate(items) if item.kind is Kind.USER and not _summary(item)), default=-1)
        answering = any(item.kind in (Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL) for item in items[asked + 1:])
        stop = request.window is not None and tokens >= int(request.window * STOP) and answering  # (within a turn)
        last = next((m for m in reversed(range(len(items))) if items[m].kind is Kind.TOOL_RESULT or _summary(items[m])), None)
        if last is not None and not stop:  # (the runner's request after STOP_PROMPT has no marker)
            marker = f"[CURRENT CONTEXT TOKEN: {tokens + RESERVE}]"
            if items[last].kind is Kind.TOOL_RESULT and len(answered(items, last)) > 1:
                notes += (marker,)  # (one message answering several calls is not rewritten: a note right after it)
            else:
                items[last] = replace(items[last], text=f"{items[last].text}\n\n{marker}")
        guidance = GUIDANCE.replace("{window}", WINDOW.format(window=request.window) if request.window else "")
        return Context((Item(Kind.SYSTEM, guidance), *items), notes=(*notes, *((STOP_PROMPT,) if stop else ())))

    def _compress(self, request: Request, before: list[Item], dropped: list[Item], number: int,
                  summarizer: Summarizer) -> str:
        """ACM's summary of `dropped`, whose original messages it saves first."""

        messages, media = [], []
        for item in dropped:
            message = {"role": LABELS.get(item.kind, item.kind.value), "content": item.text}
            if item.media:
                message["content"] += "".join(f" {mark}" for mark in marks(item.media, len(media)))
                message["media"] = [asdict(m) for m in item.media]
                media += item.media
            messages.append(message)
        path = Path(self.directory) / request.conversation / f"summary_{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        question = next((item.text for item in before if item.kind is Kind.USER), "")
        prompt = SUMMARY_INSTRUCTION.format(question=question, conversation=_serialized(messages, "\n\n"))
        for _ in range(2):  # ACM's parse: a <memory> block of 50 characters or more, nothing nested; else asked once more
            text = _ask(summarizer, prompt, media)
            memory = MEMORY.search(text)
            if memory and len(content := memory.group(1).strip()) >= 50 and "<memory>" not in content \
                    and "</think>" not in content:
                return content
        return text  # (the last answer, whole)

    def _recall(self, request: Request, text: str, summarizer: Summarizer) -> str:
        number, _, query = text.partition("::")
        number = (re.findall(r"\d+", number) or [""])[0]
        if not number or not query.strip():
            return "Error: missing required parameters (summary_id, query)."
        path = Path(self.directory) / request.conversation / f"summary_{number}.json"
        if not path.exists():
            return f"Error: summary_id {number} not found."
        saved = json.loads(path.read_text(encoding="utf-8"))
        media = [Media(**m) for message in saved for m in message.get("media", ())]
        prompt = QUERY_MEMORY_PROMPT.format(summary_id=number, history=_serialized(saved, "\n"), query=query.strip())
        return f"[query_memory: summary_id={number}]\n\n{_ask(summarizer, prompt, media)}"


def _ask(summarizer: Summarizer, prompt: str, media: list[Media]) -> str:
    """ACM's prompt as its runner sends it: one user message (with the range's media), no system prompt,
    no tools; its thinking stripped."""

    return THINK.sub("", summarizer.complete([Item(Kind.USER, prompt, media=tuple(media))])).strip()


def _serialized(messages: list[dict[str, Any]], separator: str) -> str:
    """Messages as ACM's runner writes them for its summarizer and recall: `[role] content` (a model
    response's text and the tools' results; no calls, no reasoning)."""

    return separator.join(f"[{m['role']}] {m['content']}" for m in messages if m["role"] not in HIDDEN)


def _unmarked(item: Item, own: dict[int, str]) -> Item:
    """`item` without the marker a previous request ended it with (a harness's text that ends so is its own)."""

    found = MARKER.search(item.text)
    return item if found is None or own.get(item.ref, "").endswith(found.group()) else replace(
        item, text=item.text[:found.start()])


def _summary(item: Item) -> bool:
    return item.kind is Kind.USER and item.text.startswith(SUMMARY_ID)
