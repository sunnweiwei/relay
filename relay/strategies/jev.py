"""fast-jev-compaction (github.com/tamaratran/fast-jev-compaction, MIT) as a Relay strategy:
compaction that deletes and never summarizes. TypeSafe's Jev, a model that answers typed questions
with probabilities and writes no text, decides which tool calls and results the conversation still
needs; everything else stays verbatim, in order.

As its Claude Code plugin does it: when a turn starts with the context at 60% of the window
(`compactAtPercent`, which the plugin checks when a turn has ended), and mid-turn where the harness
would compact (`compaction`'s threshold, whose summary the plugin replaces), every tool call outside the first message and the newest six (`preserveRecentMessages`) gets two
`noul` questions, whether the call should stay and whether its result should stay verbatim. A call
whose result Jev keeps (probability 0.5 or more) stays as it is; one whose call it keeps keeps the
first 300 characters of its result and a note; the others go with their results. Jev's state is the
whole conversation, each result told by its size, shrunk in the library's stages to 25k tokens; the
questions go in as many requests as fit 30k beside it, at once. Where Jev cannot be asked (no key,
an error, a history that cannot be fitted) or frees less than a quarter of the context (in characters
of the messages' text, the calls' inputs and the results', as the library counts), `compaction`
compacts it then, as the plugin falls back to Claude Code's own summary. Messages count as Claude
Code's do: a model response, the results answering it, and each other message.

Where Relay sits makes two differences. A model response's calls and their results are one cut of
the conversation for Relay (an Anthropic message holds all of a response's results), so its calls
are kept or dropped together: each is asked about, and the response keeps what any of them is kept
for. And Relay does not see whether a result was an error: each is `ok`, as the library tells one it
does not know of. A Codex code-mode call (an `exec` script) is told as the one command it runs, as
the plugin is told a Bash call's command. Where the protocol writes the model's text (and its thinking) in the
message of its calls (Anthropic, Chat Completions, Gemini), that message goes with them. Jev is TypeSafe's API (`jev_api_key`, else TYPESAFE_API_KEY), or OpenRouter's
(`jev_url=https://openrouter.ai/api/alpha/decisions`, `jev_model=~typesafe/jev-latest`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from .. import settings
from ..core.ir import Context, Item, Kind, Media, Request
from ..core.media import counted
from ..core.tokens import item_tokens
from .base import Summarizer
from .compaction import Compaction
from ..harnesses.codex import scripted
from .conversation import RESPONSE, calls as _parsed

log = logging.getLogger("relay")

URL, MODEL = "https://api.typesafe.ai/v1/systemone", "jev-latest"
# The library's (src/state.ts, src/compact.ts), as they are.
STATE_CONTEXT = ("A coding assistant conversation is being compacted to free context. `history` is the whole conversation "
                 "so far, oldest first; tool outputs are replaced by a short `result` note and long texts may be abridged. "
                 "Each question asks whether one tool call, or the full output of that call, still needs to stay in the "
                 "history verbatim. Whatever is not kept is deleted permanently, but the assistant can always re-run a "
                 "tool or re-read a file.")
KEEP_CALL = ("Tool call {id} ({tool}) should stay in the history: knowing this call was made, with its input, still "
             "matters for what the assistant does next")
KEEP_RESULT = ("The full output of tool call {id} ({tool}, {chars} chars) should stay in the history verbatim: the "
               "assistant still needs its contents and re-running the tool would not do")
COUNTED = frozenset({Kind.USER, Kind.ASSISTANT, Kind.SUMMARY, Kind.TOOL_CALL, Kind.TOOL_RESULT})
TRUNCATED = "[fast-jev-compaction truncated {what} of this tool result; re-run the tool if needed]"
INPUT_CHARS = (1000, 200, 60)  # successive caps on a call's input in the state
TEXT_HEAD, TEXT_TAIL = 400, 150
OVERHEAD = 20  # tokens the request envelope adds around state and questions
PIECES = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
SPACES = re.compile(r"\s+")


@dataclass
class _Call:
    """A tool call with its result, as the library pairs them."""

    id: str
    tool: str
    input: Any  # its input as JSON, or as text where it is not JSON
    message: int  # the message holding the call
    result: int  # the message holding its result
    chars: int  # its result's size
    pinned: bool
    cut: int  # the cut of the conversation (a model response and its results) it goes with
    media: tuple[Media, ...]  # its result's


@dataclass(frozen=True)
class Jev:
    trigger: Compaction = field(default_factory=lambda: Compaction(ratio=0.6))  # when (the plugin's compactAtPercent)
    fallback: Compaction = field(default_factory=Compaction)  # where Jev cannot do it
    keep: float = 0.5  # keepThreshold
    recent: int = 6  # preserveRecentMessages
    state_tokens: int = 25_000  # maxStateTokens
    request_tokens: int = 30_000  # maxRequestTokens
    head: int = 300  # truncateHeadChars
    min_reduction: float = 0.25  # the plugin's minReductionRatio
    api_key: str = ""
    url: str = URL
    model: str = MODEL
    ask: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = field(default=None, compare=False)
    name: str = "jev"

    @classmethod
    def from_settings(cls) -> Jev:
        base = Compaction.from_settings()  # (compact_ratio: the fallback's own threshold, not the plugin's trigger)
        return cls(trigger=replace(base, ratio=float(settings.value("jev_compact_at", "0.6"))), fallback=base,
                   api_key=settings.value("jev_api_key") or os.getenv("TYPESAFE_API_KEY", ""),
                   url=settings.value("jev_url", URL) or URL, model=settings.value("jev_model", MODEL) or MODEL)

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "trigger": self.trigger.fingerprint(), "fallback": self.fallback.fingerprint(),
                "keep": self.keep, "recent": self.recent, "state_tokens": self.state_tokens,
                "request_tokens": self.request_tokens, "head": self.head, "min_reduction": self.min_reduction,
                "model": self.model, "context": STATE_CONTEXT}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if not request.tools or not (request.force or self._due(request)):
            return None
        # The plugin compacts when a turn has ended (`turn.complete`), before the next prompt exists:
        # at a turn start, the history before it; the new turn follows as it came.
        start = None if request.force else _turn(request.current)
        history = request if start is None else replace(
            request, current=request.current[:start], boundaries=frozenset(b for b in request.boundaries if b <= start),
            tokens=request.tokens - sum(map(item_tokens, request.current[start:])))
        try:
            items, stats = self._compact(history)
        except Exception as error:  # (as the plugin: whatever went wrong, the summary instead)
            log.warning("jev: fallback to compaction (%s)", error)
            return self.fallback.plan(replace(request, force=True), summarizer)
        # (the library's `messageChars`: the messages' text, the calls' inputs and the results' text;
        # not the harness's instructions and tools, which Codex declares among the items, nor thinking or media)
        size = lambda kept: sum(len(item.text) for item in kept if item.kind in COUNTED)  # noqa: E731
        before = size(history.current)
        reduction = (before - size(items)) / before if before else 0.0
        log.info("jev: %d%% reduction; %s", round(reduction * 100), stats)
        if reduction < self.min_reduction:
            log.info("jev: fallback to compaction (below %d%% minimum)", round(self.min_reduction * 100))
            return self.fallback.plan(replace(request, force=True), summarizer)
        return Context(items if start is None else (*items, *request.current[start:]))

    def _due(self, request: Request) -> bool:
        """At the trigger when a user turn starts (its history before the turn), as the plugin checks
        when a turn has ended; mid-turn, where the harness compacts."""

        items = request.current
        if (start := _turn(items)) is not None:
            before = request.tokens - sum(map(item_tokens, items[start:]))
            return self.trigger.due(replace(request, tokens=before))[0]
        return self.fallback.due(request)[0]

    def _compact(self, request: Request) -> tuple[tuple[Item, ...], str]:
        items = request.current
        messages = _messages(items)
        calls = self._calls(items, messages, request.boundaries)
        candidates = [call for call in calls if not call.pinned]
        answers: dict[str, tuple[float, float]] = {}
        stage, tokens, batches = "", 0, []
        if candidates:
            state, tokens, stage = self._state(items, messages, calls)
            batches = self._batches(candidates, tokens)
            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                for answered in pool.map(lambda batch: self._answers(state, batch), batches):
                    answers.update(answered)
        decided: dict[int, str] = {}  # each cut's action, from its calls' answers
        for cut in sorted({call.cut for call in calls}):
            own = [call for call in calls if call.cut == cut]
            if any(call.pinned for call in own):
                continue
            keep_call, keep_result = (max(answers[call.id][k] for call in own) for k in (0, 1))
            decided[cut] = "keep" if keep_result >= self.keep else "drop_result" if keep_call >= self.keep else "drop_call"
        actions = [decided.get(call.cut, "keep") for call in calls if not call.pinned]
        log.info("jev: decisions: %s", " ".join(f"{call.id}:{call.tool}:{decided.get(call.cut, 'keep')}/call="
                                                 f"{answers[call.id][0]:.2f}/result={answers[call.id][1]:.2f}"
                                                 for call in calls if not call.pinned) or "(none)")
        stats = (f"{actions.count('keep')} kept, {actions.count('drop_result')} results truncated, "
                 f"{actions.count('drop_call')} call_dropped, {len(calls) - len(actions)} pinned; state ~{tokens} tokens "
                 f"({stage}) in {len(batches)} request(s)")
        return self._apply(items, request.boundaries, decided), stats

    def _calls(self, items: tuple[Item, ...], messages: list[list[int]], boundaries: frozenset[int]) -> list[_Call]:
        at = {n: m for m, members in enumerate(messages) for n in members}
        pinned = lambda m: m == 0 or m >= len(messages) - self.recent  # noqa: E731
        calls: list[_Call] = []
        for cut, (start, end) in enumerate(_cuts(items, boundaries)):
            made = [(n, *_scripted(tool, value)) for n in range(start, end) if items[n].kind is Kind.TOOL_CALL
                    for tool, value in _parsed(items[n].text)[0]]
            results = [n for n in range(start, end) if items[n].kind is Kind.TOOL_RESULT]
            if not made or not results:  # (a call without its result is not a candidate)
                continue
            for k, (n, tool, value) in enumerate(made):
                answered = [results[k]] if len(results) == len(made) else results  # (one message of all: together)
                calls.append(_Call(f"t{len(calls) + 1}", tool, value, at[n], at[answered[-1]],
                                   sum(len(items[r].text) for r in answered), False, cut,
                                   tuple(m for r in answered for m in items[r].media)))
        for call in calls:  # (a cut is pinned where one of its calls is)
            call.pinned = any(pinned(c.message) or pinned(c.result) for c in calls if c.cut == call.cut)
        return calls

    def _state(self, items: tuple[Item, ...], messages: list[list[int]], calls: list[_Call]) -> tuple[dict, int, str]:
        """The library's `fitState`: the whole history, shrunk in stages until it fits."""

        users = [items[m[0]].text for m in messages if items[m[0]].kind is Kind.USER and items[m[0]].text.strip()]
        goal = "\n".join(_truncate(text, 500) for text in users[-3:])
        state = lambda history: {"context": STATE_CONTEXT, "goal": goal, "history": history}  # noqa: E731
        size = lambda entry: _tokens(_json(entry)) + 1  # noqa: E731
        base = _tokens(_json(state([])))
        texts = [_text(items, members) for members in messages]
        by_message: dict[int, list[_Call]] = {}
        for call in calls:
            by_message.setdefault(call.message, []).append(call)
        pinned = lambda entry: entry["i"] == 0 or entry["i"] >= len(messages) - self.recent  # noqa: E731

        def entries(limit: int) -> list[dict[str, Any]]:
            history = []
            for i, members in enumerate(messages):
                own = [{"id": c.id, "tool": c.tool, "input": _truncate(_input(c.input), limit),
                        "result": f"ok, {_described(c.chars, c.media)} (omitted)"} for c in by_message.get(i, [])]
                if texts[i].strip() or own:
                    history.append({"i": i, "role": _role(items, members), "text": texts[i], **({"tool_calls": own} if own else {})})
            return history

        for limit in INPUT_CHARS:
            history = entries(limit)
            sizes = [size(entry) for entry in history]
            tokens = base + sum(sizes)
            if tokens <= self.state_tokens:
                return state(history), tokens, "full" if limit == INPUT_CHARS[0] else f"inputs<={limit}"

        order = [n for n, e in enumerate(history) if not pinned(e)] + [n for n, e in enumerate(history) if pinned(e)]

        def shrink(n: int, **changes: Any) -> bool:
            nonlocal tokens
            history[n].update(changes)
            tokens += size(history[n]) - sizes[n]
            sizes[n] = size(history[n])
            return tokens <= self.state_tokens

        for n in order:
            if len(history[n]["text"]) > TEXT_HEAD + TEXT_TAIL + 40 and shrink(n, text=_abridge(history[n]["text"])):
                return state(history), tokens, "texts abridged"
        for n in order:
            if not pinned(history[n]) and history[n]["text"] and shrink(n, text=f"[… {len(texts[history[n]['i']])} chars omitted …]"):
                return state(history), tokens, "old messages collapsed"
        for n in order:
            own = by_message.get(history[n]["i"])
            if not pinned(history[n]) and own and shrink(n, tool_calls=[_line(c) for c in own]):
                return state(history), tokens, "old calls compacted"
        left: set[int] = set()
        for n in order:
            if not pinned(history[n]) and "tool_calls" not in history[n]:
                left.add(n)
                tokens -= sizes[n]
                if tokens <= self.state_tokens:
                    return state([e for k, e in enumerate(history) if k not in left]), tokens, "old messages left out"
        merged: list[dict[str, Any]] = []
        foldable = lambda e: not pinned(e) and not e["text"] and isinstance(e.get("tool_calls", [None])[0], str)  # noqa: E731
        for entry in (e for k, e in enumerate(history) if k not in left):  # runs of old call-only messages, folded
            if merged and foldable(merged[-1]) and foldable(entry) and merged[-1]["role"] == entry["role"]:
                merged[-1]["tool_calls"] = [*merged[-1]["tool_calls"], *entry["tool_calls"]]
            else:
                merged.append(dict(entry))
        tokens = base + sum(size(entry) for entry in merged)
        if tokens <= self.state_tokens:
            return state(merged), tokens, "old calls merged"
        raise ValueError(f"history too large for Jev (~{tokens} tokens after truncation, limit {self.state_tokens})")

    def _batches(self, calls: list[_Call], state_tokens: int) -> list[list[_Call]]:
        budget = self.request_tokens - state_tokens - OVERHEAD
        batches: list[list[_Call]] = []
        current, used = [], 0
        for call in calls:
            tokens = _tokens(_json(_questions(call)))
            if current and used + tokens > budget:
                batches.append(current)
                current, used = [], 0
            if not current and tokens > budget:
                raise ValueError(f"state leaves no room for questions (~{state_tokens} of {self.request_tokens} tokens)")
            current.append(call)
            used += tokens
        return [*batches, current] if current else batches

    def _answers(self, state: dict[str, Any], batch: list[_Call]) -> dict[str, tuple[float, float]]:
        questions = {name: question for call in batch for name, question in _questions(call).items()}
        answers = (self.ask or self._post)(state, questions).get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Jev response is missing answers")
        return {call.id: (_noul(answers, f"call_{call.id}"), _noul(answers, f"result_{call.id}")) for call in batch}

    def _post(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("no Jev key: set jev_api_key (or TYPESAFE_API_KEY)")
        from ..transport import outbound  # (Relay's trust and keepalive; here: that package imports the strategies)

        with outbound.client(timeout=120) as client:
            response = client.post(self.url, json={"model": self.model, "state": state, "questions": questions},
                                   headers={"authorization": f"Bearer {self.api_key}"})
        if response.status_code >= 300:
            raise ValueError(f"Jev request failed ({response.status_code}): {response.text[:200]}")
        try:
            return response.json()
        except ValueError:
            raise ValueError("Jev returned malformed JSON") from None

    def _apply(self, items: tuple[Item, ...], boundaries: frozenset[int], decided: dict[int, str]) -> tuple[Item, ...]:
        """The library's `applyDecisions`: a dropped call goes with its result (and the reasoning
        before it), a dropped result keeps its head and a note."""

        out: list[Item] = []
        cuts = {n: cut for cut, (start, end) in enumerate(_cuts(items, boundaries)) for n in range(start, end)}
        for n, item in enumerate(items):
            action = decided.get(cuts.get(n, -1), "keep")
            if action == "drop_result" and item.kind is Kind.TOOL_RESULT:
                out.append(replace(item, text=self._truncated(item.text, item.media), media=()))
            elif not (action == "drop_call" and item.kind in (Kind.TOOL_CALL, Kind.TOOL_RESULT, Kind.REASONING)):
                out.append(item)
        return tuple(out)

    def _truncated(self, text: str, media: tuple[Media, ...]) -> str:
        if len(text) <= self.head + 120:
            return f"{text}\n{TRUNCATED.format(what=_described(0, media))}".strip() if media else text
        return (f"{text[: self.head]}\n" if self.head > 0 else "") + TRUNCATED.format(
            what=_described(len(text) - self.head, media))


def _scripted(tool: str, value: Any) -> tuple[str, Any]:
    """A call as Jev is told it: a Codex code-mode script (`exec`, JavaScript) as the one command it
    runs (`exec_command` {"cmd": ...}) or patch it applies, as the plugin sees a Bash call's command."""

    if tool == "exec" and isinstance(value, str) and (ran := scripted(value)) is not None:
        return ran[0], ({"cmd": ran[1]} if ran[0] == "exec_command" else ran[1])
    return tool, value


def _described(chars: int, media: tuple[Media, ...]) -> str:
    """A result's size as Jev and the model are told it: "N chars", and its media."""

    shown = [f"{chars} chars"] if chars or not media else []
    return " and ".join([*shown, *([counted(media)] if media else [])])



def _turn(items: tuple[Item, ...]) -> int | None:
    """Where a new user turn starts (the items after the model's last response, a user message among
    them), or None mid-turn."""

    last = max((n for n, item in enumerate(items) if item.kind in RESPONSE), default=None)
    return last + 1 if last is not None and any(item.kind is Kind.USER for item in items[last + 1:]) else None

def _messages(items: tuple[Item, ...]) -> list[list[int]]:
    """The conversation as Claude Code's messages: a model response, the results answering it,
    each other item a message of its own."""

    messages: list[list[int]] = []
    for n, item in enumerate(items):
        last = items[messages[-1][-1]].kind if messages else None
        if last is not None and ((item.kind in RESPONSE and last in RESPONSE) or item.kind is last is Kind.TOOL_RESULT):
            messages[-1].append(n)
        else:
            messages.append([n])
    return messages


def _cuts(items: tuple[Item, ...], boundaries: frozenset[int]) -> list[tuple[int, int]]:
    """The smallest cuts of the conversation that keep each call with its result."""

    points = sorted(b for b in boundaries if b <= len(items))
    return list(zip(points, points[1:]))


def _role(items: tuple[Item, ...], members: list[int]) -> str:
    return "assistant" if items[members[0]].kind in RESPONSE else "user"


def _text(items: tuple[Item, ...], members: list[int]) -> str:
    """A message's text: what the model or the user wrote (not reasoning, calls or results)."""

    parts = []
    for n in members:
        item = items[n]
        if item.kind is Kind.TOOL_CALL:
            parts.append(_parsed(item.text)[1])
        elif item.kind not in (Kind.REASONING, Kind.TOOL_RESULT):
            parts.append(item.text)
    return "\n".join(part for part in parts if part)


def _questions(call: _Call) -> dict[str, Any]:
    return {f"call_{call.id}": {"type": "noul", "instructions": KEEP_CALL.format(id=call.id, tool=call.tool)},
            f"result_{call.id}": {"type": "noul",
                                  "instructions": KEEP_RESULT.format(id=call.id, tool=call.tool, chars=call.chars)}}


def _noul(answers: dict[str, Any], name: str) -> float:
    answer = answers.get(name)
    value = answer.get("noul") if isinstance(answer, dict) else None
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Invalid Jev answer for {name}")
    return float(value)


def _line(call: _Call) -> str:
    """A call in one line, for when the structured form is too costly."""

    if isinstance(call.input, dict):
        text = " ".join(f"{key}={(value if isinstance(value, str) else _truncate(_json({key: value}), 200))}"
                        for key, value in call.input.items())
    else:
        text = _input(call.input)
    return f"{call.id} {call.tool} {_truncate(SPACES.sub(' ', text), INPUT_CHARS[-1])} → ok {call.chars}ch"


def _input(value: Any) -> str:
    return value if isinstance(value, str) else _json(value)


def _json(value: Any) -> str:  # (as JavaScript's JSON.stringify writes it)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tokens(text: str) -> int:
    """The library's estimate: a word one token per six letters, a digit half a token, any other
    symbol nine tenths (a little above what Jev counts)."""

    tokens = 0.0
    for piece in PIECES.findall(text):
        if piece[0].isdigit():
            tokens += len(piece) / 2
        elif piece[0].isascii() and piece[0].isalpha():
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return math.ceil(tokens)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: max(0, limit - 1)]}…"


def _abridge(text: str) -> str:
    if len(text) <= TEXT_HEAD + TEXT_TAIL + 40:
        return text
    return f"{text[:TEXT_HEAD]}\n[… {len(text) - TEXT_HEAD - TEXT_TAIL} chars omitted …]\n{text[-TEXT_TAIL:]}"
