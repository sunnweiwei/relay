"""Claude Code's compaction hook: before every request Claude Code asks, the strategy decides.

`relay install claude_code --via hook` enables a function-hook plugin (early access, Claude Code's
own plugin API); its model endpoint is left alone, so Remote Control and every login keep
working. As a session starts, the plugin (`plugins/claude_code`) moves Claude Code's own
auto-compaction trigger to the bottom, so Claude Code compacts, through the hook, before every
request (a plugin that does not load leaves Claude Code as it is). It posts the transcript and the
size of Claude Code's last request to `/relay/v1/compact`; the strategy decides on it as it would
on a request through the proxy, and either it is "not now" (the compaction is skipped, leaving no
trace) or Claude Code keeps the new transcript and re-injects its own context (instructions,
environment, reminders) after it. Claude Code would have compacted at the latest 33k tokens below
its window (20k output reserve, 13k buffer); past that the strategy rewrites as if forced, as
Claude Code itself blocks a few thousand later.

The new transcript keeps Claude Code's messages whole, or writes them: new text for a message
(a tool's result, say), or a message of the strategy's own. A shell result whose new text is all
the strategy changes (Context Folding's branch) is written as it comes in (`tool.call`, before
Claude Code stores it), no compaction. The strategy's instructions join the
end of Claude Code's system prompt (its last section, `context_management`) as the session starts,
from `/relay/v1/instructions`: Claude Code builds its system prompt once a session, so they are
the same all through it. The history the strategy sees is the transcript, after as many
placeholders as the compactions through Relay took out of it, so it grows as the harness's own
history would. Not carried, so a strategy that needs them runs through the proxy: instructions
that change within a session, notes for one request, a completion of the strategy's own, and
images in a message the strategy writes (Claude Code's messages here are their text).

Summaries come from Claude Code's own model. `$.model.fork` continues the conversation as
Claude Code last sent it, its latest reply included, so a summary of exactly that is Codex's
summary request; the messages after the latest reply (tool results not sent yet, mid-turn)
follow in the prompt as text. A sub-agent's history, or one a fork cannot see (a resumed
session before its first request), is summarized from a rendering in a plain completion.

Each request's size is the upstream's count of the agent's previous one (the plugin reads it from
`turn.step`, a sub-agent's too) plus what came since. Claude Code ends a turn that compacts three
times in a row, each within three requests of the one before (its thrashing breaker); the plugin
counts as it does and has Relay hold a compaction that would be the second such (not when forced).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
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
SINCE = "2.1.294"  # the oldest release of Claude Code the hook was checked with
MARGIN = 33_000  # below Claude Code's window, where it would have compacted at the latest
TAIL = "Since your latest reply above, these came in (the results of its tool calls, then anything the user said):"
HISTORY = "The conversation so far:"
WRITTEN = (Kind.USER, Kind.SUMMARY, Kind.ASSISTANT)  # what a message the strategy writes can be
DROPPED = 5_000  # messages remembered as dropped, per session and agent


class ClaudeCodeHook:
    name = "hook"
    carries = frozenset({"instructions", "rewrites", "history"})
    checked = {"compaction": "2.1.294–2.1.309", "folding": "2.1.294–2.1.309", "autocompact": "2.1.294–2.1.309",
               "prolong": "2.1.294–2.1.309",
               "selfcompact": "2.1.309"}  # (at a 200k window: at 40k it compacts every second response)
    # Before the proxy, checked against it: the hook keeps Remote Control and what Claude Code does
    # only on Anthropic's own endpoint; sub-agents are compacted at their real size (`turn.step`)
    # and no compaction trips Claude Code's thrashing breaker (`check.py --subagent`, low thresholds).
    # Remote Control itself was not started in a check (that enrolls a session with an account).
    preferred = True

    def installation(self, relay_url: str) -> Installation:
        path = settings_file()
        return Installation([
            Setting(path, ("env", "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS"), "1"),
            Setting(path, ("env", "RELAY_HOOK_URL"), relay_url),
            Setting(path, ("extraKnownMarketplaces", "relay"), {"source": {"source": "directory", "path": str(PLUGIN)}}),
            Setting(path, ("enabledPlugins", "relay@relay"), True),
        ])

    def unavailable(self) -> str | None:
        release = installed_release()
        if release is None:
            return "Claude Code was not found here (the hook needs its release)"
        if _release(release) < _release(SINCE):
            return f"Claude Code {release} is older than {SINCE}, the oldest release the hook was checked with"
        return None

    def instructions(self, engine: Engine) -> str:
        """The strategy's instructions, as Claude Code's system prompt gets them once a session."""

        context = engine.strategy.plan(Request((), (), frozenset({0}), 0, engine.window), Supplied())
        return "\n\n".join(item.text for item in context.items if item.kind is Kind.SYSTEM) if context else ""

    def asked(self, engine: Engine, request: dict[str, Any], messages: list[dict[str, Any]], begins: bool = True
              ) -> tuple[Request, list[Item], tuple[bytes, list[bytes]]]:
        """The conversation as the strategy gets it, and where its history is kept (`begins`: a
        request, whose size begins the context window if none has since the latest compaction)."""

        model = request.get("model")
        name = hashlib.sha256(f"{request.get('session')}\0{request.get('agent')}".encode()).hexdigest()[:16]
        partition, key = engine.store.partition("hook-history"), [name.encode()]
        kept = (engine.store.match(partition, key) or (0, {}))[1]
        gone = set(kept.get("dropped", ()))
        items, boundaries = conversation(messages, {n for n, m in enumerate(messages) if digest(m) in gone})
        # The upstream's count of Claude Code's last request plus what was added since its reply;
        # before the first reply (a new or resumed process), the transcript alone.
        last, window = request.get("tokens"), request.get("window")
        seen = 1 + max((n for n, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        added = [item for item in items if item.ref >= seen] if last else items
        tokens = (last or 0) + sum(item_tokens(item, bytes_per_token(model)) for item in added)
        # Kept by session and agent: the placeholders for what compactions took out of the history,
        # the size of the first request of the context window (after the latest compaction), and
        # what they dropped (a resumed session brings it back).
        offset, base = kept.get("offset", 0), kept.get("base")
        if base is None and begins:
            base = tokens
            engine.store.put(partition, key, {**kept, "offset": offset, "base": base})
        asked = Request((Item(Kind.OTHER),) * offset + tuple(items), tuple(items), boundaries, tokens,
                        engine.window or window or context_window(model),
                        bool(window) and tokens >= window - MARGIN, base=base, conversation=name)
        return asked, items, (partition, key, kept)

    def result(self, engine: Engine, request: dict[str, Any]) -> dict[str, Any]:
        """A shell result as it comes in (`tool.call`): its new text, where all the strategy changes
        is that (Context Folding's branch, say), so it is written as the result, no compaction.
        Anything else waits for the next request's compaction."""

        call = request["call"]
        messages = [*request["messages"], {"role": "user", "text": "", "toolUses": [],
                                           "toolResults": [{"tool_use_id": call["tool_use_id"], "text": call["text"]}]}]
        asked, items, _ = self.asked(engine, request, messages, begins=False)
        if not items or items[-1].ref != len(messages) - 1 or len(items) not in asked.boundaries:
            return {}  # (another call of the same reply still runs)
        try:
            context = engine.answer(asked, Supplied())
        except SummaryNeeded:
            return {}
        answer = tuple(item for item in context.items if item.kind is not Kind.SYSTEM) if context else asked.current
        if (len(answer) == len(asked.current) and answer[:-1] == asked.current[:-1] and not answer[-1].media
                and answer[-1].kind is Kind.TOOL_RESULT and answer[-1].ref == asked.current[-1].ref
                and answer[-1].text != asked.current[-1].text):
            engine.emit({"strategy": engine.strategy.name, "path": self.name, "harness": "claude_code",
                         "agent": request.get("agent"), "result": call["tool_use_id"]},
                        {"head": [{"ref": answer[-1].ref, "texts": [answer[-1].text]}]})
            return {"text": answer[-1].text}
        return {}

    def compact(self, engine: Engine, request: dict[str, Any]) -> dict[str, Any]:
        """One request's decision: the plugin's request (Claude Code's transcript, the size of its
        last request, the instructions it placed, the summaries made so far) answered with "not
        now", with the transcript's replacement, or with the summary to make next."""

        messages, model = request["messages"], request.get("model")
        summaries = Supplied(dict(request.get("summaries") or {}))
        asked, items, (partition, key, kept) = self.asked(engine, request, messages)
        tokens = asked.tokens
        if request.get("hold") and not asked.force:  # (Claude Code's thrashing breaker: see the plugin)
            return {"skip": f"{engine.strategy.name}: held, two compactions in a row would end Claude Code's turn"}
        try:
            context = engine.answer(asked, summaries)
        except SummaryNeeded as need:
            return {"summarize": ask(need, items, messages, request.get("agent"))}
        if context is None:
            answer = asked.current
        else:
            if context.notes:
                raise ValueError("the hook cannot show notes for one request; run this strategy through the proxy")
            placed = "\n\n".join(item.text for item in context.items if item.kind is Kind.SYSTEM)
            if placed != (request.get("instructions") or ""):
                raise ValueError("the strategy's instructions are not the ones Claude Code's system prompt got as the "
                                 "session started (it gets them once a session); run this strategy through the proxy")
            answer = tuple(item for item in context.items if item.kind is not Kind.SYSTEM)
        if answer == asked.current and {item.ref for item in items} >= {*range(len(messages))} - context_only(messages):
            return {"skip": f"{engine.strategy.name}: not now"}  # (and nothing a resumed session brought back)
        entries, head = replacement(answer, items, messages)  # what the hook cannot carry fails
        after, _ = conversation([messages[e["ref"]] if "ref" in e else e for e in entries])
        refs = {e["ref"] for e in entries if "ref" in e}
        dropped = [*kept.get("dropped", ()), *(digest(m) for n, m in enumerate(messages)
                                                if n not in refs and n not in context_only(messages))]
        engine.store.put(partition, key, {"offset": len(asked.history) - len(after), "base": None,
                                          "dropped": list(dict.fromkeys(dropped))[-DROPPED:]})
        engine.emit({"strategy": engine.strategy.name, "path": self.name, "harness": "claude_code", "model": model,
                     "trigger": request.get("trigger"), "agent": request.get("agent"), "forced": asked.force,
                     "tokens_before": tokens, "messages_before": len(messages), "messages_after": len(entries),
                     "summary_requests": len(summaries.summaries)}, {"head": head})
        return {"messages": entries}


def installed_release() -> str | None:
    """The release of the `claude` found on PATH, as `claude --version` says it."""

    if (command := shutil.which("claude")) is None:
        return None
    try:
        answer = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return match.group(0) if (match := re.search(r"\d+\.\d+\.\d+", answer)) else None


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version))


def repeated(messages: list[dict[str, Any]]) -> set[int]:
    """The messages a resumed session repeats. Resuming, Claude Code gathers every copy of an
    assistant message that compactions kept (it keeps them by the API's message id) with every
    result its calls had, oldest first: the latest copy and the latest result stand. (It also
    follows a kept result to the call it first answered, which brings back the history before the
    compaction: what Relay remembers as dropped, `digest`.)"""

    last: dict[tuple[str, str], int] = {}
    for index, message in enumerate(messages):
        for use in message.get("toolUses") or []:
            last["use", use["tool_use_id"]] = index
        for result in message.get("toolResults") or []:
            last["result", result["tool_use_id"]] = index
    return {index for index, message in enumerate(messages)
            if any(last["use", u["tool_use_id"]] != index for u in message.get("toolUses") or [])
            or any(last["result", r["tool_use_id"]] != index for r in message.get("toolResults") or [])}


def digest(message: dict[str, Any]) -> str:
    """A message: its row in Claude Code's session (`handle`, the same across a resume), else what it
    holds (its calls and results by id, a result by its text too, else its role and text)."""

    if message.get("handle"):
        return message["handle"]
    uses = [(u.get("tool_use_id"), u.get("tool"), u.get("input")) for u in message.get("toolUses") or []]
    results = [(r.get("tool_use_id"), r.get("text")) for r in message.get("toolResults") or []]
    held = [message.get("role"), message.get("text") or "", uses, results]
    return hashlib.sha256(json.dumps(held, sort_keys=True, default=str).encode()).hexdigest()[:20]


def context_only(messages: list[dict[str, Any]]) -> set[int]:
    """Messages that are Claude Code's own context, which it re-injects after a compaction."""

    profile = HARNESSES["claude_code"]
    return {n for n, m in enumerate(messages) if m["role"] == "user" and not m.get("toolResults")
            and profile.refine(Item(Kind.USER, m.get("text") or "")).kind in CONTEXT_KINDS}


def conversation(messages: list[dict[str, Any]], gone: set[int] = frozenset()) -> tuple[list[Item], frozenset[int]]:
    """Claude Code's transcript as the strategy sees it: each item refers to its message, and a cut
    is legal between messages where no tool call is waiting for its result. Left out: what a
    resumed session repeats (`repeated`), and `gone`, what compactions dropped that it brought back."""

    profile, items, legal, waiting = HARNESSES["claude_code"], [], set(), set()
    skipped = repeated(messages) | gone
    for index, message in enumerate(messages):
        if index in skipped:
            continue
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


def replacement(answer: tuple[Item, ...], items: list[Item], messages: list[dict[str, Any]]) -> tuple[list, list]:
    """The new transcript: messages kept whole by index, written with new text, or written new.
    Also the context as the event log records it."""

    entries: list[dict[str, Any]] = []
    head: list[dict[str, Any]] = []
    used: set[int] = set()
    n = 0
    while n < len(answer):
        item = answer[n]
        if item.ref is None:
            if item.kind not in WRITTEN or item.media:
                raise ValueError(f"the hook cannot write a {item.kind.value} item; run this strategy through the proxy")
            entries.append({"role": "assistant" if item.kind is Kind.ASSISTANT else "user", "text": item.text,
                            "toolUses": []})
            head.append({"kind": item.kind.value, "text": item.text})
            n += 1
            continue
        end = n
        while end < len(answer) and answer[end].ref == item.ref:  # this message's items, as the strategy has them
            end += 1
        group, n = answer[n:end], end
        if item.ref in used:
            raise ValueError("the context repeats a message")
        used.add(item.ref)
        kept = [it for it in items if it.ref == item.ref]
        if list(group) == kept:
            entries.append({"ref": item.ref})
            head.append({"ref": item.ref})
        else:
            entries.append(rewritten(messages[item.ref], kept, group))
            head.append({"ref": item.ref, "texts": [it.text for it in group]})
    written = [messages[e["ref"]] if "ref" in e else e for e in entries]
    calls = {u["tool_use_id"] for m in written for u in m.get("toolUses") or []}
    if calls != {r["tool_use_id"] for m in written for r in m.get("toolResults") or []}:
        raise ValueError("the context separates a tool call from its result")
    return entries, head


def rewritten(message: dict[str, Any], kept: list[Item], group: tuple[Item, ...]) -> dict[str, Any]:
    """A message of Claude Code's with the strategy's new text for its items (a tool's result, its
    text): the same items, the same calls, in the same order."""

    if [it.kind for it in group] != [it.kind for it in kept] or any(it.media for it in group):
        raise ValueError("the hook keeps a message's items as they are, or gives them new text; run this strategy "
                         "through the proxy")
    if any(new.text != old.text for new, old in zip(group, kept) if new.kind is Kind.TOOL_CALL):
        raise ValueError("the hook cannot change a tool call; run this strategy through the proxy")
    texts = iter(it.text for it in group if it.kind is Kind.TOOL_RESULT)
    text = next((it.text for it in group if it.kind in (Kind.USER, Kind.ASSISTANT, Kind.REASONING)), message.get("text") or "")
    written = {"role": message["role"], "text": text, "toolUses": message.get("toolUses") or []}
    if message.get("toolResults"):
        written["toolResults"] = [{**result, "text": next(texts)} for result in message["toolResults"]]
    return written
