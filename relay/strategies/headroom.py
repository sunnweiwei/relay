"""Headroom (github.com/headroomlabs-ai/headroom, Apache-2.0) as a Relay strategy: text compressed in
place by Headroom itself, as its proxy compresses Codex's requests (`/v1/responses` in its `coding`
profile, `cache` mode: `proxy/handlers/openai.py:2208-2878`). Nothing is dropped, summarized or
reordered; calls and reasoning stay as they were.

Each text is a unit, compressed on its own: a tool output given as a string, or each text part of one
given in parts (Codex's code mode: its header, then each text the script printed), from 512 bytes; a
user, assistant or summary message from 8192 bytes, but the newest user message (a summary is one),
which stays as it is until a newer one comes and is compressed on that request. Headroom's
ContentRouter picks one compressor per unit: JSON arrays become tables (SmartCrusher), logs keep their
errors and the lines around them, search results and diffs their hits, and prose loses function words
(Kompress), the instructions in a message included. A file read stays verbatim unless it is data
(`protect_reads`, `cat` included): Headroom is given the command of the call an output answers, of a
Codex code-mode script each `exec_command` as its proxy reads them, one not written as a string taken
for a read. An output given as a string that repeats an earlier one's lines points at it ("same as
msg 3"); parts never do. A unit goes as Headroom first compressed it and stays so, so the provider's
prompt cache holds but where an earlier user message is compressed, and Headroom counts tokens with
the request's own model.

Relay asks a running Headroom (`headroom proxy --port 8790`) through the endpoint it offers gateways,
`POST /v1/compress`, in OpenAI Chat messages: each part and each message alone, an output after the
call it answers, and the outputs given as strings together, after those forwarded before, frozen
(`frozen_message_count`): the only way that endpoint compares them. Where it does not answer, those
units go as they are, and stay so.

That endpoint does not do all the proxy does. It is marker-free by default: the proxy also keeps the
originals and gives the model a `headroom_retrieve` tool to get them back (CCR), which takes
answering the model's own tool calls, past what a strategy does; `headroom_mode=ccr` asks for the
markers, for a harness given Headroom's MCP server (`headroom mcp`), whose tool reads the same store.
It runs Headroom's chat pipeline, which keeps an error output of up to 8,000 characters as it is,
folds a `bash` or `shell` tool's search output losslessly, judges a read on each part rather than on
the whole output, lets a repeated read point at the first, and reads one command per call: a script's
commands go joined by `;` (one with a heredoc left out, which would make the whole line no read).
Its pointers count Relay's messages, and an output under 512 bytes is not folded. Relay never
compresses the harness's own context (Codex's AGENTS.md and environment messages), takes a message
whole rather than part by part, and leaves an Anthropic or Gemini message answering calls of several
tools at once as it is.

Headroom never shrinks a conversation: behind its proxy, the harness's own compaction keeps it within
the window. Relay turns that off, so `compaction` does the same, at the same point (90% of the window,
or when the upstream finds the request too long), and Headroom goes on from there.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from .. import settings
from ..core.ir import Context, Item, Kind, Request
from .base import Summarizer
from .compaction import Compaction
from .conversation import RESPONSE, calls

log = logging.getLogger("relay")

URL = "http://127.0.0.1:8790"  # (Headroom's own default, 8787, is Relay's)
MODEL = "claude-sonnet-4-5-20250929"  # what it counts with where the request names no model (compress()'s default)
OUTPUT_FLOOR, MESSAGE_FLOOR = 512, 8192  # bytes below which a unit stays as it is (openai.py:1925-1926)
USERS, MESSAGES = frozenset({Kind.USER, Kind.SUMMARY}), frozenset({Kind.USER, Kind.SUMMARY, Kind.ASSISTANT})
# How Headroom reads a Codex code-mode script's commands (content_router.py:630-726).
EXEC_COMMAND = re.compile(r"\bexec_command\s*\(")
JS_CMD = re.compile(r"""\{[^{}]*?(?<![\w$])(?:cmd|"cmd"|'cmd')\s*:\s*"""
                    r"""(?:"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\$]|\\.|\$(?!\{))*)`)(?=\s*[,}])""")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v"}
HEREDOC = re.compile(r"(^|\s)<<")
Unit = tuple[int, int | None]  # an item, and which of its parts (None: all of it)


@dataclass(frozen=True)
class Headroom:
    url: str = URL
    model: str = ""  # the model it counts tokens with; unset: the request's own
    mode: str = ""  # the endpoint's `config.mode`: "" (marker-free), "ccr" or "lossy_inline"
    token: str = ""  # HEADROOM_PROXY_TOKEN, where Headroom is given one
    fallback: Compaction = field(default_factory=Compaction)  # what keeps the context within the window
    post: Callable[[dict[str, Any]], dict[str, Any]] | None = field(default=None, compare=False, repr=False)
    name: str = "headroom"

    @classmethod
    def from_settings(cls) -> Headroom:
        return cls(url=settings.value("headroom_url", URL) or URL, model=settings.value("headroom_model", ""),
                   mode=settings.value("headroom_mode", ""), token=settings.value("headroom_token", ""),
                   fallback=Compaction.from_settings())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "mode": self.mode, "fallback": self.fallback.fingerprint()}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        state = request.state or {}
        if (compacted := self.fallback.plan(request, summarizer)) is not None:
            items, sent = compacted.items, len(compacted.items)  # (as forwarded, but a user message held back)
        else:
            items, sent = request.current, state.get("sent", 0)
            if sent > len(items) or (sent and state.get("last") != _digest(items[sent - 1])):
                sent = 0  # (not the history forwarded: the harness compacted it, or resumed another)
            if sent == len(items):
                return None  # (nothing new: a retry)
        newest = max((n for n, item in enumerate(items) if item.kind in USERS), default=-1)
        answers, joint = _answers(items)
        asks: list[tuple[list[dict[str, Any]], int, dict[int, Unit]]] = []  # messages, how many frozen, units among them

        def output(messages: list[dict[str, Any]], n: int, text: str) -> int:  # after the call it answers; where it is
            if call := answers.get(n):
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call["id"] if call else "call_none", "content": text})
            return len(messages) - 1

        strings = [n for n, item in enumerate(items) if item.kind is Kind.TOOL_RESULT and not item.parts]
        if whole := [n for n in strings if n >= sent and n not in joint and _size(items[n].text) >= OUTPUT_FLOOR]:
            messages: list[dict[str, Any]] = []  # (after those forwarded, which they may point at)
            for n in strings:
                if n < sent:
                    output(messages, n, items[n].text)
            frozen = len(messages)
            units = {output(messages, n, items[n].text): (n, None) for n in whole}
            asks.append((messages, frozen, units))
        for n, item in enumerate(items):
            if item.kind is Kind.TOOL_RESULT and item.parts and n >= sent and n not in joint:
                for k, part in enumerate(_parts(item)):
                    if _size(part) >= OUTPUT_FLOOR:
                        messages = []
                        units = {output(messages, n, part): (n, k)}
                        asks.append((messages, 0, units))
            elif (item.kind in MESSAGES and n != newest and _size(item.text) >= MESSAGE_FLOOR
                  and (n >= sent or item.kind in USERS and _digest(item) == state.get("held"))):
                role = "assistant" if item.kind is Kind.ASSISTANT else "user"  # (a summary goes as the user's)
                asks.append(([{"role": role, "content": item.text}], 0, {0: (n, None)}))

        def ask(asked: tuple[list[dict[str, Any]], int, dict[int, Unit]]) -> dict[str, Any] | None:
            messages, frozen, _ = asked
            config = {"frozen_message_count": frozen, **({"mode": self.mode} if self.mode else {})}
            body = {"messages": messages, "model": self.model or request.model or MODEL, "config": config,
                    **({"token_budget": request.window} if request.window else {})}
            try:
                answer = (self.post or self._post)(body)
                if len(answer["messages"]) != len(messages):
                    raise ValueError(f"{len(answer['messages'])} messages back for {len(messages)}")
                return answer
            except Exception as error:  # (as Headroom fails open: these as they are, from now on)
                log.warning("headroom: %s; the results go as they are", error)
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:  # (four at once, as its proxy compresses units: openai.py:139)
            answered = list(pool.map(ask, asks))
        texts: dict[Unit, str] = {}
        saved, before_tokens, transforms = 0, 0, []
        for (_, _, units), answer in zip(asks, answered):
            if answer is None:
                continue
            compressed = answer["messages"]
            texts |= {unit: compressed[at]["content"] for at, unit in units.items()
                      if isinstance(compressed[at].get("content"), str)}
            saved += answer.get("tokens_saved") or 0
            before_tokens += answer.get("tokens_before") or 0
            transforms += [t for t in answer.get("transforms_applied") or [] if t not in transforms]
        out = list(items)
        for n in {n for n, _ in texts}:
            text = texts[n, None] if (n, None) in texts else "\n".join(
                texts.get((n, k), part) for k, part in enumerate(_parts(items[n])))
            if text != items[n].text:
                out[n] = replace(items[n], text=text)
        if asks:
            log.info("headroom: %s of %s tokens saved; %s", saved, before_tokens,
                     ", ".join(transforms) or "nothing compressed")
        state = {"sent": len(out), "last": _digest(out[-1]), "held": _digest(out[newest]) if newest >= 0 else None}
        return replace(compacted, items=tuple(out), state=state) if compacted else Context(tuple(out), state=state)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        from ..transport import outbound  # (Relay's trust and keepalive; here: that package imports the strategies)

        with outbound.client(timeout=120) as client:
            response = client.post(f"{self.url.rstrip('/')}/v1/compress", json=body,
                                   headers={"authorization": f"Bearer {self.token}"} if self.token else {})
        if response.status_code != 200:
            raise ValueError(f"Headroom answered {response.status_code}: {response.text[:200]}")
        return response.json()


def _answers(items: tuple[Item, ...]) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """The call each tool result answers, as an OpenAI Chat tool call, and the results that answer
    calls of several tools at once. A result answers the oldest of its response's calls not yet
    answered, the last also the calls left (where a message holds all of a response's results:
    Anthropic, Gemini)."""

    made, answered, waiting, last = {}, {}, [], None

    def close() -> None:  # (a response's results are in: calls left answered by its last)
        if waiting and last is not None:
            answered[last] += waiting
        waiting.clear()

    for n, item in enumerate(items):
        if item.kind is Kind.TOOL_CALL:
            if n and items[n - 1].kind not in RESPONSE:  # (a new response)
                close()
            for k, (tool, arguments) in enumerate(_calls(item.text)):
                key = f"call_{n}_{k}"
                made[key] = {"id": key, "type": "function", "function": {"name": tool, "arguments": arguments}}
                waiting.append(key)
        elif item.kind is Kind.TOOL_RESULT:
            answered[n], last = [waiting.pop(0) if waiting else answered[last][-1] if last is not None else ""], n
        elif item.kind in MESSAGES:
            close()
    close()
    return ({n: made[ids[0]] for n, ids in answered.items() if ids[0] in made},
            {n for n, ids in answered.items() if len({made[key]["function"]["name"] for key in ids if key in made}) > 1})


def _calls(text: str) -> list[tuple[str, str]]:
    """The calls a call item's text holds, each its tool and its arguments: as the model gave them
    (JSON), or, of a Codex code-mode script (`exec`, JavaScript), its commands as JSON arguments."""

    tool, _, script = text.strip().partition(" ")
    if re.fullmatch(r"[\w.:-]+", tool) and script[:1] not in "{[" and (commands := _commands(script)):
        return [(tool, json.dumps({"cmd": _command(commands)}))]
    return [(tool, value if isinstance(value, str) else json.dumps(value)) for tool, value in calls(text)[0]]


def _commands(script: str) -> list[str | None]:
    """The command of each `exec_command` call of a Codex code-mode script, as Headroom reads them
    (`_custom_tool_call_commands`, content_router.py:681-726): a JSON argument's `cmd` or `command`,
    else its `cmd` written as a string literal; None for one neither (a variable, an expression)."""

    commands, decoder = [], json.JSONDecoder()
    for match in EXEC_COMMAND.finditer(script):
        start = script.find("{", match.end())
        if start < 0 or script[match.end():start].strip():
            commands.append(None)
            continue
        try:
            arguments = decoder.raw_decode(script, start)[0]
        except ValueError:
            literal = JS_CMD.match(script, start)
            commands.append(_unescaped(next(g for g in literal.groups() if g is not None)) or None if literal else None)
            continue
        command = arguments.get("command", arguments.get("cmd", "")) if isinstance(arguments, dict) else ""
        command = " ".join(map(str, command)) if isinstance(command, list) else command
        if command and isinstance(command, str):
            commands.append(command)
    return commands


def _unescaped(body: str) -> str | None:
    """A JavaScript string literal's text; None for an escape Headroom does not decode (`\\x61`)."""

    if any(c.isalnum() and c not in ESCAPES for c in re.findall(r"\\(.)", body, re.S)):
        return None
    return re.sub(r"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)), body, flags=re.S)


def _command(commands: list[str | None]) -> str:
    """A script's commands as the one command line Headroom reads a read from: a read where its proxy
    reads any of them as one (openai.py:2436-2447). Joined by `;`, each side of which it reads on its
    own; one with a heredoc left out (never a read, it would make the whole line none:
    content_router.py:837-839); `cat` for one not written as a string, which the proxy takes for a read."""

    return "; ".join([c for c in commands if c is not None and not HEREDOC.search(c)] + ["cat"] * (None in commands))


def _parts(item: Item) -> tuple[str, ...]:
    return item.parts if "\n".join(item.parts) == item.text else (item.text,)


def _size(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def _digest(item: Item) -> str:
    return hashlib.sha256(f"{item.kind.value}\0{item.text}".encode()).hexdigest()[:16]
