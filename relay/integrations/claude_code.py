"""Claude Code's compaction hook: before every request Claude Code asks, the strategy decides.

`relay install claude_code --via hook` enables a function-hook plugin (early access since
Claude Code 2.1.274) and moves Claude Code's own auto-compaction trigger to the bottom, so
Claude Code compacts, through the hook, before every request; its model endpoint is left alone,
so Remote Control and every login keep working. The plugin (`plugins/claude_code`) posts the
transcript and the size of Claude Code's last request to `/relay/v1/compact`; the strategy
decides on it as it would on a request through the proxy, and either it is "not now" (the
compaction is skipped, leaving no trace) or Claude Code keeps the rewritten transcript and
re-injects its own context (instructions, environment, reminders) after it. Claude Code would
have compacted at the latest 33k tokens below its window (20k output reserve, 13k buffer); past
that the strategy rewrites as if forced, as Claude Code itself blocks a few thousand later.

The hook carries the conversation and the strategy's state (kept here by session), not
instructions for the system prompt, notes for one request, or new content for a message Claude
Code keeps: a strategy that needs those runs through the proxy.

Summaries come from Claude Code's own model. `$.model.fork` continues the conversation as
Claude Code last sent it, its latest reply included, so a summary of exactly that is Codex's
summary request; the messages after the latest reply (tool results not sent yet, mid-turn)
follow in the prompt as text. A sub-agent's history, or one a fork cannot see (a resumed
session before its first request), is summarized from a rendering in a plain completion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..core.engine import Engine
from ..core.ir import CONTEXT_KINDS, Item, Kind, Request
from ..core.tokens import bytes_per_token, item_tokens
from ..providers import context_window
from ..harnesses import HARNESSES
from ..harnesses.claude_code import settings_file
from ..install import Setting
from .base import Installation, Supplied, SummaryNeeded, render

PLUGIN = Path(__file__).parent / "plugins" / "claude_code"  # a local marketplace holding the plugin
ALWAYS = "0.01"  # Claude Code's trigger, in percent of its window: below any request
MARGIN = 33_000  # below Claude Code's window, where it would have compacted at the latest
TAIL = "These messages came after your latest reply above:"
HISTORY = "The conversation so far:"


class ClaudeCodeHook:
    name = "hook"

    def installation(self, relay_url: str) -> Installation:
        path = settings_file()
        return Installation([
            Setting(path, ("env", "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS"), "1"),
            Setting(path, ("env", "RELAY_HOOK_URL"), relay_url),
            Setting(path, ("env", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"), ALWAYS),
            Setting(path, ("extraKnownMarketplaces", "relay"), {"source": {"source": "directory", "path": str(PLUGIN)}}),
            Setting(path, ("enabledPlugins", "relay@relay"), True),
        ])

    def compact(self, engine: Engine, request: dict[str, Any]) -> dict[str, Any]:
        """One request's decision: the plugin's request (Claude Code's transcript, the size of its
        last request, the summaries made so far) answered with "not now", with the transcript's
        replacement, or with the summary to make next."""

        messages, model = request["messages"], request.get("model")
        items, boundaries = conversation(messages)
        # The upstream's count of Claude Code's last request plus what was added since its reply;
        # before the first reply (a new or resumed process), the transcript alone.
        last, window = request.get("tokens"), request.get("window")
        seen = 1 + max((n for n, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        added = [item for item in items if item.ref >= seen] if last else items
        tokens = (last or 0) + sum(item_tokens(item, bytes_per_token(model)) for item in added)
        summaries = Supplied(dict(request.get("summaries") or {}))
        name = f"{request.get('session')}\0{request.get('agent')}"
        asked = Request(tuple(items), tuple(items), boundaries, tokens, engine.window or window or context_window(model),
                        bool(window) and tokens >= window - MARGIN, conversation=hashlib.sha256(name.encode()).hexdigest()[:16])
        try:
            context = engine.answer(asked, summaries)
        except SummaryNeeded as need:
            return {"summarize": ask(need, items, messages, request.get("agent"))}
        if context is None:
            return {"skip": f"{engine.strategy.name}: not now"}
        entries, head = replacement(context.items, context.notes, items, messages)  # what the hook cannot carry fails
        if context.items == asked.current:
            return {"skip": f"{engine.strategy.name}: not now"}
        engine.emit({"strategy": engine.strategy.name, "path": self.name, "harness": "claude_code", "model": model,
                     "trigger": request.get("trigger"), "agent": request.get("agent"), "forced": asked.force,
                     "tokens_before": tokens, "messages_before": len(messages), "messages_after": len(entries),
                     "summary_requests": len(summaries.summaries)}, {"head": head})
        return {"messages": entries}


def conversation(messages: list[dict[str, Any]]) -> tuple[list[Item], frozenset[int]]:
    """Claude Code's transcript as the strategy sees it: each item refers to its message, and a cut
    is legal between messages where no tool call is waiting for its result."""

    profile, items, legal, waiting = HARNESSES["claude_code"], [], set(), set()
    for index, message in enumerate(messages):
        if not waiting:
            legal.add(len(items))
        text, uses, results = message.get("text") or "", message.get("toolUses") or [], message.get("toolResults") or []
        if message["role"] == "assistant":
            if text or not uses:  # an empty reply holds only thinking
                items.append(Item(Kind.ASSISTANT if text else Kind.REASONING, text, ref=index))
            items += [Item(Kind.TOOL_CALL, f"{use.get('tool')} {json.dumps(use.get('input'), ensure_ascii=False)}", ref=index)
                      for use in uses]
            waiting |= {use["tool_use_id"] for use in uses}
        elif results:
            items += [Item(Kind.TOOL_RESULT, result.get("text") or "", ref=index) for result in results]
            waiting -= {result["tool_use_id"] for result in results}
        elif (item := profile.refine(Item(Kind.USER, text, ref=index))).kind not in CONTEXT_KINDS:
            items.append(item)  # Claude Code re-injects its own context after compacting
    if not waiting:
        legal.add(len(items))
    return items, frozenset(legal)


def ask(need: SummaryNeeded, items: list[Item], messages: list[dict[str, Any]], agent: str | None) -> dict[str, Any]:
    """The summary request for the plugin to send: a fork where it sees the history to summarize
    (the main conversation, through its latest reply), and always a rendering for a completion."""

    seen = 1 + max((n for n, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    upto = items[need.cut].ref if need.cut < len(items) else len(messages)
    fork = None
    if agent is None and upto >= seen:
        tail = [item for item in items[: need.cut] if item.ref >= seen]
        fork = f"{TAIL}\n\n{render(tail)}\n\n{need.prompt}" if tail else need.prompt
    return {"key": need.key, "fork": fork, "complete": f"{HISTORY}\n\n{render(items[: need.cut])}\n\n{need.prompt}"}


def replacement(answer: tuple[Item, ...], notes: tuple[str, ...], items: list[Item],
                messages: list[dict[str, Any]]) -> tuple[list, list]:
    """The new transcript: messages kept whole by index, or written as text. Also the context as
    the event log records it."""

    if notes:
        raise ValueError("the hook cannot show notes for one request; run this strategy through the proxy")
    known, entries, head = set(items), [], []
    for item in answer:
        if item.ref is None:
            if item.kind not in (Kind.USER, Kind.SUMMARY, Kind.ASSISTANT) or item.media:
                raise ValueError(f"the hook cannot write a {item.kind.value} item; run this strategy through the proxy")
            entries.append({"role": "assistant" if item.kind is Kind.ASSISTANT else "user", "text": item.text})
            head.append({"kind": item.kind.value, "text": item.text})
        elif item not in known:
            raise ValueError("the hook cannot change a message Claude Code keeps; run this strategy through the proxy")
        elif entries[-1:] != [{"ref": item.ref}]:  # a message's items, kept, keep the message
            if {"ref": item.ref} in entries:
                raise ValueError("the context repeats a message")
            entries.append({"ref": item.ref})
            head.append({"ref": item.ref})
    kept = [messages[e["ref"]] for e in entries if "ref" in e]
    calls = {u["tool_use_id"] for m in kept for u in m.get("toolUses") or []}
    if calls != {r["tool_use_id"] for m in kept for r in m.get("toolResults") or []}:
        raise ValueError("the context separates a tool call from its result")
    return entries, head
