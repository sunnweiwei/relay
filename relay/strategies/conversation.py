"""What strategies read off a conversation: where the model's responses start, and the signals the
agent gives with the harness's own shell (`echo "[name] text"`), which stand in for a method's own
tools (FoldAgent's `branch`, AutoCompact's `compact()`, ACM's `manage_context`) in any harness."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..core.ir import Item, Kind

RESPONSE = frozenset({Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL})  # what one model response holds


def responses(items: Sequence[Item]) -> list[int]:
    """Where each model response starts: a run of the model's own items, ended by what answers it."""

    return [n for n, item in enumerate(items) if item.kind in RESPONSE and (n == 0 or items[n - 1].kind not in RESPONSE)]


def step(items: Sequence[Item], n: int) -> tuple[int, int, int] | None:
    """The first model response from `n` on that calls tools, and the results answering it: where
    it starts, where its results start and where they end."""

    while n < len(items):
        if items[n].kind not in RESPONSE:
            n += 1
            continue
        start = n
        while n < len(items) and items[n].kind in RESPONSE:
            n += 1
        end = n
        while end < len(items) and items[end].kind is Kind.TOOL_RESULT:
            end += 1
        if end > n and any(item.kind is Kind.TOOL_CALL for item in items[start:n]):
            return start, n, end
    return None


def signalled(response: Sequence[Item], results: Sequence[Item], pattern: re.Pattern[str],
              whole: bool = False) -> tuple[str, str, int | None, bool] | None:
    """A shell call of a model response echoing a signal (`pattern`'s groups: its name, then its
    text to the end of the line), what it says, which of `results` is the echo's (None where one
    result answers all the response's calls: Anthropic Messages, Gemini), and whether it is the
    response's only call (as the guidance asks: "alone"). A signal is a command that is a lone
    `echo` of it, in any tool's input, and an echo prints its argument: the output says it (in
    whatever form the harness reports it; of several results, the one saying what the echo
    printed), else, reported without output, the command. Text that only mentions a signal (the
    model's, a task handed to a sub-agent, a file written, a grep for it) is none. `whole`: an
    output that is the echo alone says all of its lines (a message of several lines, as printed)."""

    calls = [values for item in response if item.kind is Kind.TOOL_CALL for values in _inputs(item.text)]
    echoed = [argument for values in calls if (argument := _echoed(values, pattern)) is not None]
    if not echoed or not results:
        return None
    name, said = pattern.match(echoed[-1]).groups()
    own = len(results) == len(calls)  # (a result per call)
    if own and len(results) == 1 and not results[0].text.strip():  # reported without output: the command says it
        return name, (echoed[-1][len(name) + 2:] if whole else said).strip(), 0, True
    for k in reversed(range(len(results))):
        for text in _strings(results[k].text):
            found = pattern.findall(text)  # the last: the output's, after any command the harness repeats
            if found and found[-1][0] == name and (len(results) == 1 or found[-1][1].strip() == said.strip()):
                if whole and own and len(found) == 1 and text.lstrip().startswith(f"[{name}]"):
                    said = text.lstrip()[len(name) + 2:]
                else:
                    said = found[-1][1]
                return name, said.strip(), k if own else None, len(calls) == 1
    return None  # output, but not the echo's


IGNORED = ("Your {name} echo ran beside other tool calls, so it was ignored: run it again alone, with no other "
           "tool call beside it.")  # what the model is told of a signal beside other calls


def ignored(items: list[Item], echo: int | None, end: int, name: str) -> tuple[str, ...]:
    """A signal beside other calls is ignored, and the model told so: in the echo's own result
    (`echo`, its place in `items`) where it has one, else in a note on the request its results are
    the last of (where they end: `end`). The notes to show."""

    if echo is not None:
        items[echo] = replace(items[echo], text=IGNORED.format(name=name))
        return ()
    return (IGNORED.format(name=name),) if end == len(items) else ()
# A lone echo: `echo`, flags, then one argument (quoted, or bare words), and nothing after it.
LONE_ECHO = re.compile(r"""\s*echo(?:\s+-[neE]+)*\s+(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|([^"'`;&|<>$\n]+?))\s*""", re.S)
# Where a tool call's input starts in a call's text: a line's JSON, after the tool's name (`Bash({...})`,
# `Bash {...}`, `exec_command {...}`) or not (Gemini's `{"name": ..., "args": {...}}`).
INPUT = re.compile(r"^(?:[\w.:-]+[( ])?(?=[{\[])", re.M)


def _inputs(text: str) -> list[list[str]]:
    """The string values of the input of each tool call in a call's text (the text itself, where
    none is found)."""

    inputs, end, decoder = [], 0, json.JSONDecoder()
    for match in INPUT.finditer(text):
        if match.end() >= end:
            try:
                value, end = decoder.raw_decode(text, match.end())
            except ValueError:
                continue
            inputs.append(_values(value))
    return inputs or [[text]]


def _echoed(values: list[str], pattern: re.Pattern[str]) -> str | None:
    """The argument of a lone echo of a signal among a tool call's input values, if one is."""

    for value in values + [re.sub(r'\\(["\'\\])', r"\1", v) for v in values]:  # (or quoted once more, as some report it)
        if (lone := LONE_ECHO.fullmatch(value)) is not None:
            double, single, bare = lone.groups()
            argument = re.sub(r'\\(["\\$`])', r"\1", double) if double is not None else single if single is not None else bare
            if pattern.match(argument):
                return argument
    return None


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
