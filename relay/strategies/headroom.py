"""Headroom (github.com/headroomlabs-ai/headroom, Apache-2.0) as a Relay strategy: every tool result
compressed in place as it comes in, by Headroom itself, and the history before it forwarded as it
was. Nothing is dropped, summarized or reordered.

As Headroom's proxy runs by default (its `coding` profile in `cache` mode), only the newest delta is
compressed and what was forwarded stays byte for byte, so the provider's prompt cache holds. Its
ContentRouter picks one compressor for each result by what it holds: JSON arrays become tables
(SmartCrusher), logs keep their errors and the lines around them, search results and diffs their
hits, a file read stays verbatim (`protect_reads`, `cat` included) and the same output again points
at the first. The profile compresses the user's messages too: a long one loses function words, the
instructions in it included. Relay asks a running Headroom (`headroom proxy --port 8790`) through the endpoint it
offers gateways, `POST /v1/compress`: the conversation as OpenAI Chat messages, with every message
forwarded before frozen (`frozen_message_count`).

Two differences from Headroom's proxy. The endpoint's default is marker-free: Headroom's proxy also
keeps the originals and injects a `headroom_retrieve` tool the model can get them back with (CCR),
which takes answering the model's own tool calls, past what a strategy does. `headroom_mode=ccr` asks
for the markers, for a harness given Headroom's MCP server (`headroom mcp`), whose tool reads the
same store. And the messages are Relay's view of the conversation (a call and its result each a
message), so where a result points at an earlier one ("same as msg 2"), it counts those. Where
Headroom does not answer, the results go as they are, and stay so.

Headroom never shrinks a conversation: behind its proxy, the harness's own compaction keeps it within
the window. Relay turns that off, so `compaction` does the same, at the same point (90% of the window,
or when the upstream finds the request too long), and Headroom goes on from there.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .. import settings
from ..core.ir import Context, Item, Kind, Request
from .base import Summarizer
from .compaction import Compaction
from .conversation import RESPONSE, calls

log = logging.getLogger("relay")

URL = "http://127.0.0.1:8790"  # (Headroom's own default, 8787, is Relay's)
MODEL = "claude-sonnet-4-5-20250929"  # its tokenizer and window, where Relay has none (compress()'s default)
EDITABLE = frozenset({Kind.USER, Kind.SUMMARY, Kind.ASSISTANT, Kind.TOOL_RESULT})  # (a call goes as it was made)


@dataclass(frozen=True)
class Headroom:
    url: str = URL
    model: str = MODEL
    mode: str = ""  # the endpoint's `config.mode`: "" (marker-free), "ccr" or "lossy_inline"
    token: str = ""  # HEADROOM_PROXY_TOKEN, where Headroom is given one
    fallback: Compaction = field(default_factory=Compaction)  # what keeps the context within the window
    post: Callable[[dict[str, Any]], dict[str, Any]] | None = field(default=None, compare=False, repr=False)
    name: str = "headroom"

    @classmethod
    def from_settings(cls) -> Headroom:
        return cls(url=settings.value("headroom_url", URL) or URL, model=settings.value("headroom_model", MODEL) or MODEL,
                   mode=settings.value("headroom_mode", ""), token=settings.value("headroom_token", ""),
                   fallback=Compaction.from_settings())

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "mode": self.mode, "fallback": self.fallback.fingerprint()}

    def plan(self, request: Request, summarizer: Summarizer) -> Context | None:
        if (compacted := self.fallback.plan(request, summarizer)) is not None:
            return replace(compacted, state=_forwarded(compacted.items))
        items, state = request.current, request.state or {}
        sent = state.get("sent", 0)
        if sent > len(items) or (sent and state.get("last") != _digest(items[sent - 1])):
            sent = 0  # (not the history forwarded: the harness compacted it, or resumed another)
        if sent == len(items):
            return None  # (nothing new: a retry)
        messages, at, joint = _messages(items)
        config = {"frozen_message_count": sum(n < sent for n in at), **({"mode": self.mode} if self.mode else {})}
        body = {"messages": messages, "model": self.model, "config": config,
                **({"token_budget": request.window} if request.window else {})}
        try:
            answer = (self.post or self._post)(body)
            compressed = answer["messages"]
            if len(compressed) != len(messages):
                raise ValueError(f"{len(compressed)} messages back for {len(messages)}")
        except Exception as error:  # (as Headroom fails open: the results as they are, from now on)
            log.warning("headroom: %s; the results go as they are", error)
            return Context(items, state=_forwarded(items))
        out = list(items)
        for message, n in zip(compressed, at):
            text = message.get("content")
            if (n >= sent and n not in joint and items[n].kind in EDITABLE and isinstance(text, str)
                    and text != items[n].text):
                out[n] = replace(items[n], text=text)
        log.info("headroom: %s of %s tokens saved; %s", answer.get("tokens_saved"), answer.get("tokens_before"),
                 ", ".join(answer.get("transforms_applied") or []) or "nothing compressed")
        return Context(tuple(out), state=_forwarded(out))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        from ..transport import outbound  # (Relay's trust and keepalive; here: that package imports the strategies)

        with outbound.client(timeout=120) as client:
            response = client.post(f"{self.url.rstrip('/')}/v1/compress", json=body,
                                   headers={"authorization": f"Bearer {self.token}"} if self.token else {})
        if response.status_code != 200:
            raise ValueError(f"Headroom answered {response.status_code}: {response.text[:200]}")
        return response.json()


def _messages(items: tuple[Item, ...]) -> tuple[list[dict[str, Any]], list[int], set[int]]:
    """The conversation as OpenAI Chat messages, the item each came from, and the results that answer
    calls of several tools at once. A call each, with the model's text around it; a result answering
    the oldest of its response's calls not yet answered, the last also the calls left (where a
    message holds all of a response's results: Anthropic, Gemini)."""

    messages, at, tools, answered, waiting, last = [], [], {}, {}, [], None

    def close() -> None:  # (a response's results are in: calls left answered by its last)
        if waiting and last is not None:
            answered[last] += waiting
        waiting.clear()

    for n, item in enumerate(items):
        if item.kind is Kind.TOOL_CALL:
            if n and items[n - 1].kind not in RESPONSE:  # (a new response)
                close()
            made, text = calls(item.text)
            ids = [f"call_{n}_{k}" for k in range(len(made))]
            tools |= {id: tool for id, (tool, _) in zip(ids, made)}
            waiting += ids
            message: dict[str, Any] = {"role": "assistant", "content": text or None, "tool_calls": [
                {"id": id, "type": "function", "function": {"name": tool, "arguments": value if isinstance(value, str)
                                                            else json.dumps(value)}} for id, (tool, value) in zip(ids, made)]}
        elif item.kind is Kind.TOOL_RESULT:
            id = waiting.pop(0) if waiting else answered[last][-1] if last is not None else "call_none"
            answered[n], last = [id], n
            message = {"role": "tool", "tool_call_id": id, "content": item.text}
        elif item.kind in (Kind.USER, Kind.SUMMARY, Kind.ASSISTANT):
            close()
            message = {"role": "assistant" if item.kind is Kind.ASSISTANT else "user", "content": item.text}
        else:
            continue  # (reasoning, and what the protocol alone reads)
        messages.append(message)
        at.append(n)
    close()
    return messages, at, {n for n, ids in answered.items() if len({tools.get(id) for id in ids}) > 1}


def _forwarded(items: tuple[Item, ...] | list[Item]) -> dict[str, Any]:
    return {"sent": len(items), "last": _digest(items[-1])}


def _digest(item: Item) -> str:
    return hashlib.sha256(f"{item.kind.value}\0{item.text}".encode()).hexdigest()[:16]
