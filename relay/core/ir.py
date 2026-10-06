"""The strategy contract, protocol-neutral.

A codec maps the wire items of one API onto `Item`s. Before every request a strategy gets the
`Request`, the conversation as the harness sent it and as the model currently sees it, and
answers with the `Context` the model should see: items taken from the request (kept as they
are, or with new text or media), new items, anywhere. What the harness wrote itself (system and
context items) never reaches the strategy; the harness profile places it. Items remember which
wire item they came from, so anything a strategy keeps is forwarded byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    SYSTEM = "system"  # system / developer instructions
    CONTEXT = "context"  # context the harness injected as a user-role message
    USER = "user"  # input from the user
    SUMMARY = "summary"  # stands in for the history before it (Relay's or the harness's own compaction)
    ASSISTANT = "assistant"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OTHER = "other"


AGENT_KINDS = frozenset({Kind.ASSISTANT, Kind.REASONING, Kind.TOOL_CALL, Kind.TOOL_RESULT})
CONTEXT_KINDS = frozenset({Kind.SYSTEM, Kind.CONTEXT})
LABELS = {Kind.USER: "user", Kind.ASSISTANT: "assistant", Kind.REASONING: "reasoning", Kind.TOOL_CALL: "tool_call",
          Kind.TOOL_RESULT: "tool", Kind.SUMMARY: "summary"}  # an item's role, as a note names it


@dataclass(frozen=True)
class Media:
    """Non-text content of an item: an image, a file, audio."""

    type: str  # "image", "file", "audio"
    mime: str = ""  # e.g. "image/png", when known
    data: str = ""  # base64 content, when inline
    url: str = ""  # or where it is: a URL, a provider's file id


@dataclass(frozen=True)
class Item:
    kind: Kind
    text: str = ""
    ref: int | None = None  # the wire item it came from; None for an item Relay writes
    media: tuple[Media, ...] = ()
    wire: str | None = None  # JSON of a wire item Relay writes, when not a plain message
    opaque: int = 0  # bytes of opaque content the model reads (encrypted reasoning, ...), decoded


@dataclass(frozen=True)
class Request:
    """One request as the strategy sees it: the conversation only, no system or context items."""

    history: tuple[Item, ...]  # every item the harness sent, as it sent them
    current: tuple[Item, ...]  # what the model sees if nothing changes: the last context, then what came since
    boundaries: frozenset[int]  # cuts of `current` that keep each tool call with its result
    tokens: int  # estimated prompt tokens of `current` sent (the upstream's count where known)
    window: int | None  # context window of the requested model, when known
    force: bool = False  # the upstream rejected `current` as too long
    base: int | None = None  # prompt tokens when the current context window began
    state: Any = None  # what the strategy saved on this conversation's previous request
    conversation: str = ""  # names the conversation, the same on all of its requests
    tools: bool = True  # the request offers the model tools: an agent's turn, not a side call (a title)
    # The items as they will be sent: what the protocol cannot keep where it stands (a tool call whose
    # result is gone, new text for a call) told as a note. Relay applies it to every answer.
    sendable: Callable[[tuple[Item, ...]], tuple[Item, ...]] = field(default=lambda items: items, repr=False,
                                                                     compare=False)


def note(role: str, text: str) -> Item:
    """A user-role note telling `text` as `role` (an item the protocol cannot keep, a strategy's own)."""

    return Item(Kind.USER, text if role == "user" else f"[context role={role}]\n{text}")


@dataclass(frozen=True)
class Context:
    """A strategy's answer: what the model sees from now on, until the next answer.

    `items` come from the request (as they are, or `dataclasses.replace`d with new text or
    media) or are new; a new SYSTEM item joins the system prompt. A SUMMARY stands in for the
    history before it, so the harness's context goes where the harness puts it after its own
    compaction; otherwise it stays where the harness sent it."""

    items: tuple[Item, ...]
    state: Any = None  # given back as `Request.state` on the conversation's next request (JSON)
    notes: tuple[str, ...] = field(default=())  # shown at the end of this request only
