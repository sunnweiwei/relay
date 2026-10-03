"""Protocol-neutral view of a conversation.

A codec maps the wire items of one API onto `Item`s; strategies only ever see a
`View` of the conversation and answer with a `Rewrite` of it. What the harness wrote
itself (system and context items) is left to the harness profile, which puts its current
state into every rewrite. Items remember which wire item they came from, so anything a
strategy keeps is forwarded byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    SYSTEM = "system"  # system / developer instructions inside the item list
    CONTEXT = "context"  # context the harness injected as a user-role message
    USER = "user"  # input from the user
    SUMMARY = "summary"  # a compaction summary (Relay's or the harness's own)
    ASSISTANT = "assistant"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OTHER = "other"


AGENT_KINDS = frozenset({Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL, Kind.TOOL_RESULT})
CONTEXT_KINDS = frozenset({Kind.SYSTEM, Kind.CONTEXT})


@dataclass(frozen=True)
class Item:
    kind: Kind
    text: str = ""
    ref: int | None = None  # index of the wire item in the request; None if Relay wrote it
    media: bool = False  # carries non-text content such as images or files
    wire: str | None = None  # JSON of the wire item Relay wrote, when not a plain user message


@dataclass(frozen=True)
class View:
    items: tuple[Item, ...]  # the conversation: no system or context items
    boundaries: frozenset[int]  # i is legal if items[:i] can be replaced as a unit
    tokens: int  # estimated prompt tokens of the whole request
    window: int | None  # context window of the requested model, when known
    force: bool = False  # the upstream already rejected this request as too long
    base: int | None = None  # prompt tokens when the current context window began


@dataclass(frozen=True)
class Rewrite:
    cut: int  # items[:cut] are replaced by `head`
    head: tuple[Item, ...]  # kept view items and/or new items written by Relay (no harness state)
