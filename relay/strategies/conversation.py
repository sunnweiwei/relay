"""What strategies read off a conversation: where the model's responses start, and the signals the
agent gives with the harness's own shell (`echo "[name] text"`), which stand in for a method's own
tools (FoldAgent's `branch`, AutoCompact's `compact()`, ACM's `manage_context`) in any harness."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from dataclasses import replace

from ..core.ir import Item, Kind, joined

RESPONSE = frozenset({Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL})  # what one model response holds


def responses(items: Sequence[Item]) -> list[int]:
    """Where each model response starts: a run of the model's own items, ended by what answers it."""

    return [n for n, item in enumerate(items) if item.kind in RESPONSE and (n == 0 or items[n - 1].kind not in RESPONSE)]


def signal(call: Item, result: Item, pattern: re.Pattern[str], whole: bool = False) -> tuple[str, str] | None:
    """A shell call echoing a signal (`pattern`'s groups: its name, then its text to the end of the
    line), and what it says: read from the echo's output (in whatever form the harness reports
    it), else from the call. `whole`: an output that is the echo alone says all of its lines (a
    message of several lines, as the echo printed it)."""

    found = signalled(call, result, pattern, whole)
    return found and found[:2]


def signalled(call: Item, result: Item, pattern: re.Pattern[str], whole: bool = False) -> tuple[str, str, str] | None:
    """`signal`, and the id of the call whose result is the echo's output where the result holds
    others beside it ("" otherwise: the result is the echo's alone). A signal is a command that is
    a lone `echo` of it (as the guidance asks: "alone"), in any tool's input, and an echo prints
    its argument: where the call reports output, that output says it. Text that only mentions a
    signal (the model's, a task handed to a sub-agent, a file written, a grep for it) is none."""

    if call.kind is not Kind.TOOL_CALL or result.kind is not Kind.TOOL_RESULT:
        return None
    uses = [(i, text) for i, text in call.parts if i] if call.parts else [("", call.text)]
    echo, argument = next(((i, argument) for i, text in reversed(uses)
                           if (argument := _echoed(text, pattern)) is not None), (None, ""))
    if echo is None:
        return None
    own = next((text for i, text in result.parts if echo and i == echo), None)
    output, alone = (own, True) if own is not None else (result.text, not result.parts)
    call_id = echo if own is not None else ""
    if not output.strip():  # reported without output: the command says it
        name, said = pattern.match(argument).groups()
        return name, (argument[len(name) + 2:] if whole else said).strip(), call_id
    for text in _strings(output):
        if found := pattern.findall(text):  # the last: the output's, after any command the harness repeats
            name, said = found[-1]
            if whole and alone and len(found) == 1 and text.lstrip().startswith(f"[{name}]"):
                said = text.lstrip()[len(name) + 2:]
            return name, said.strip(), call_id
    return None  # output, but not the echo's


# A lone echo: `echo`, flags, then one argument (quoted, or bare words), and nothing after it.
LONE_ECHO = re.compile(r"""\s*echo(?:\s+-[neE]+)*\s+(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|([^"'`;&|<>$\n]+?))\s*""", re.S)


def _echoed(use: str, pattern: re.Pattern[str]) -> str | None:
    """The argument of a lone echo of a signal among a tool use's input values, if one is."""

    start = min((i for i in (use.find("{"), use.find("[")) if i >= 0), default=-1)
    try:  # the input: JSON after the tool's name (`Bash({...})`, `Bash {...}`, `shell {...}`)
        values = _values(json.JSONDecoder().raw_decode(use[start:])[0]) if start >= 0 else [use]
    except ValueError:
        values = [use]
    for value in values + [re.sub(r'\\(["\'\\])', r"\1", v) for v in values]:  # (or quoted once more, as some report it)
        if (lone := LONE_ECHO.fullmatch(value)) is not None:
            double, single, bare = lone.groups()
            argument = re.sub(r'\\(["\\$`])', r"\1", double) if double is not None else single if single is not None else bare
            if pattern.match(argument):
                return argument
    return None


def own(result: Item, call: str) -> str:
    """The answer to `call` in `result` (all of it where `call` is "", or not among its parts)."""

    return next((text for i, text in result.parts if call and i == call), result.text)


def answered(result: Item, call: str, text: str) -> Item:
    """`result` with `text` as the answer to `call` alone, the other calls' answers as they were
    (its parts say where each goes, for a protocol that answers them in one message); all of it
    where `call` is "" (the result is that call's alone)."""

    if not call or not result.parts:
        return replace(result, text=text, parts=())
    parts = tuple((i, text if i == call else part) for i, part in result.parts)
    return replace(result, text=joined(parts), parts=parts)


def _values(value: Any) -> list[str]:
    """The strings in a JSON value (a tool's input), JSON strings among them read as JSON too
    (Chat Completions and Responses send a call's arguments as a string)."""

    strings, stack = [], [value]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            try:
                inner = json.loads(value)
            except ValueError:
                inner = None
            if isinstance(inner, dict | list):
                stack.append(inner)
            else:
                strings.append(value)
        elif isinstance(value, dict | list):
            stack.extend(value.values() if isinstance(value, dict) else value)
    return strings


def _strings(text: str) -> list[str]:
    """`text`, or, where it is JSON (some harnesses report a command's output so), its strings."""

    try:
        value = json.loads(text)
    except ValueError:
        return [text]
    strings, stack = [], [value]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict | list):
            stack.extend(value.values() if isinstance(value, dict) else value)
    return strings or [text]
