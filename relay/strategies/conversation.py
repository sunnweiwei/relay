"""What strategies read off a conversation: where the model's responses start, and the signals the
agent gives with the harness's own shell (`echo "[name] text"`), which stand in for a method's own
tools (FoldAgent's `branch`, AutoCompact's `compact()`, ACM's `manage_context`) in any harness."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..core.ir import Item, Kind

RESPONSE = frozenset({Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL})  # what one model response holds


def responses(items: Sequence[Item]) -> list[int]:
    """Where each model response starts: a run of the model's own items, ended by what answers it."""

    return [n for n, item in enumerate(items) if item.kind in RESPONSE and (n == 0 or items[n - 1].kind not in RESPONSE)]


def signal(call: Item, result: Item, pattern: re.Pattern[str]) -> tuple[str, str] | None:
    """A shell call echoing a signal (`pattern`'s groups: its name, then its text to the end of the
    line), and what it says: read from the echo's output (in whatever form the harness reports
    it), else from the call."""

    if call.kind is not Kind.TOOL_CALL or result.kind is not Kind.TOOL_RESULT or not pattern.search(call.text):
        return None
    for text in _strings(result.text):
        if found := pattern.findall(text):  # the last: the output's, after any command the harness repeats
            return found[-1][0], found[-1][1].strip()
    name, text = pattern.findall(call.text)[-1]  # no output: the echo's argument, to its closing quote
    return name, re.split(r'\\?["\']', text)[0].strip()


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
