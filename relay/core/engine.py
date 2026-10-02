"""Per-request orchestration.

For each request the engine restores the stored rewrite for the longest known
prefix, builds the protocol-neutral view the strategy sees, applies the strategy's
rewrite to the wire items, and remembers the result (and later the prompt tokens the
upstream reported) under the request's prefix. Strategy failures never reach the
harness: the request is forwarded with the last good rewrite instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..harnesses import Harness
from ..protocols.base import Body, Codec, WireItem
from ..providers import context_window
from ..strategies.base import Strategy
from .ir import AGENT_KINDS, CONTEXT_KINDS, Item, Kind, Rewrite, View
from .store import PrefixStore
from .tokens import approx_tokens

log = logging.getLogger("relay")

Post = Callable[[Body], tuple[int, Any]]  # send a body upstream -> (status, JSON payload)
TRIM_NOTE = "(Earlier conversation history was omitted here to fit the context window.)"
TRANSIENT = {429, 500, 502, 503, 504}  # summary request errors worth retrying
RETRIES = 5  # Codex's default stream_max_retries


@dataclass(frozen=True)
class Exchange:
    """A prepared request: what to forward, and where to record its usage."""

    body: Body
    partition: bytes
    keys: list[bytes]  # canonical identities of the request's wire items
    state: dict[str, Any]  # stored rewrite applied to this request
    compacted: bool = False  # this request produced a new rewrite
    estimate: int = 0  # Relay's own estimate of the forwarded prompt, if usage is never reported


class Engine:
    def __init__(
        self,
        strategy: Strategy,
        store: PrefixStore | None = None,
        *,
        window: int | None = None,
        event_log: str | None = None,
        retry_after: float = 60.0,
    ) -> None:
        self.strategy = strategy
        self.store = store or PrefixStore()
        self.window = window
        self.event_log = event_log
        self.retry_after = retry_after
        self._retry_at: dict[bytes, float] = {}  # per conversation, after a failure

    def prepare(
        self,
        codec: Codec,
        harness: Harness,
        body: Body,
        *,
        tenant: str,
        post: Post,
        force: bool = False,
    ) -> Exchange:
        raw = codec.items(body)
        # Trailing items the harness regenerates on every request (live status, say) are
        # forwarded as they are, but kept out of prefix matching and compaction.
        end = len(raw)
        while end and harness.volatile(harness.refine(codec.classify(raw[end - 1]))):
            end -= 1
        raw, volatile = raw[:end], raw[end:]
        keys = [codec.canonical(item) for item in raw]
        fingerprint = json.dumps(self.strategy.fingerprint(), sort_keys=True)
        partition = self.store.partition(tenant, codec.name, harness.name, fingerprint)
        depth, state = self.store.match(partition, keys) or (0, {})

        def view_item(index: int) -> Item:
            return replace(harness.refine(codec.classify(raw[index])), ref=index)

        def materialize(state: dict[str, Any]) -> tuple[list[Item], list[WireItem]]:
            head = [view_item(h["ref"]) if "ref" in h else Item(Kind(h["kind"]), h["text"])
                    for h in state.get("head", [])]
            items = head + [view_item(i) for i in range(state.get("covered", 0), len(raw))]
            wire = [raw[i.ref] if i.ref is not None else codec.user_message(i.text) for i in items]
            return items, wire

        items, wire = materialize(state)
        if isinstance(state.get("tokens"), int):
            tokens = state["tokens"] + sum(approx_tokens(view_item(i).text) for i in range(depth, len(raw)))
        else:
            tokens = approx_tokens(codec.preamble(body)) + sum(approx_tokens(i.text) for i in items)
            tokens += sum(approx_tokens(codec.classify(item).text) for item in volatile)
        # The harness resends its history from the start, so the system and context items before
        # the model's first action are always at hand as the conversation's initial context, which
        # compaction re-injects like Codex does, whatever an earlier rewrite kept of them.
        first_action = next((i for i in range(len(raw)) if view_item(i).kind in AGENT_KINDS), len(raw))
        view = View(
            tuple(items),
            codec.boundaries(wire),
            tokens,
            self.window or context_window(body.get("model")),
            force,
            state.get("base"),
            tuple(item for item in map(view_item, range(first_action)) if item.kind in CONTEXT_KINDS),
        )

        compacted = False
        conversation = partition + hashlib.sha256(b"\0".join(keys[:3])).digest()
        if force or time.monotonic() >= self._retry_at.get(conversation, 0.0):
            started = time.monotonic()
            try:
                rewrite = self.strategy.plan(view, _Summarizer(codec, body, view, wire, post))
                if rewrite is not None:
                    rewrite = replace(rewrite, head=codec.arrange(rewrite.head, rewrite.cut == len(view.items)))
                    state = _to_state(rewrite, view, len(raw))
                    self.store.put(partition, keys, state)
                    items, wire = materialize(state)
                    compacted = True
                    self._emit(
                        {
                            "strategy": self.strategy.name,
                            "protocol": codec.name,
                            "harness": harness.name,
                            "model": body.get("model"),
                            "forced": force,
                            "tokens_before": tokens,
                            "items_before": len(view.items),
                            "items_after": len(items),
                            "seconds": round(time.monotonic() - started, 3),
                        },
                        state,
                    )
            except Exception:
                log.warning("%s failed; forwarding the request unchanged", self.strategy.name, exc_info=True)
                self._retry_at[conversation] = time.monotonic() + self.retry_after

        unchanged = not state.get("head") and not state.get("covered")
        forwarded = body if unchanged else codec.with_items(body, [*wire, *volatile])
        estimate = approx_tokens(codec.preamble(body)) + sum(approx_tokens(i.text) for i in items)
        estimate += sum(approx_tokens(codec.classify(item).text) for item in volatile)
        return Exchange(forwarded, partition, keys, state, compacted, estimate)

    def record(self, exchange: Exchange, input_tokens: int | None) -> None:
        """Anchor future token estimates to the prompt tokens the upstream reported.

        The first size after a rewrite (or ever) starts a new context window. Upstreams that
        report no usage (some gateways send zeros) fall back to Relay's own estimate."""

        state = {k: v for k, v in exchange.state.items() if k != "tokens"}
        if "base" not in state or (input_tokens and state.get("estimated")):
            state["base"], state["estimated"] = input_tokens or exchange.estimate, not input_tokens
        if input_tokens:
            state["tokens"] = input_tokens
        self.store.put(exchange.partition, exchange.keys, state)

    def _emit(self, event: dict[str, Any], state: dict[str, Any]) -> None:
        log.info("rewrote context: %s", event)
        if self.event_log:
            record = {"time": time.time(), **event, **state}
            try:
                with open(self.event_log, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                log.warning("could not write the event log %s", self.event_log, exc_info=True)


def _to_state(rewrite: Rewrite, view: View, raw_length: int) -> dict[str, Any]:
    """Express a rewrite of the view in terms of the request's wire items."""

    if rewrite.cut not in view.boundaries:
        raise ValueError(f"rewrite cut {rewrite.cut} splits an atomic group")
    if rewrite.cut < len(view.items):
        covered = view.items[rewrite.cut].ref
        if covered is None:
            raise ValueError("rewrite cut falls inside Relay's own items")
    else:
        covered = raw_length
    head = []
    for item in rewrite.head:
        if item.ref is None:
            head.append({"kind": item.kind.value, "text": item.text})
        elif item.ref < covered:
            head.append({"ref": item.ref})
        else:
            raise ValueError("rewrite keeps an item that it does not replace")
    return {"covered": covered, "head": head}


class _Summarizer:
    """Runs the summary request through the task's own upstream ("native continuation")."""

    def __init__(self, codec: Codec, body: Body, view: View, wire: list[WireItem], post: Post) -> None:
        self.codec, self.body, self.view, self.wire, self.post = codec, body, view, wire, post

    def summarize(self, cut: int, prompt: str) -> str:
        items = self.wire[:cut]
        # Like Codex, drop the oldest history on overflow, but keep the initial context.
        keep = next(
            (i for i, item in enumerate(self.view.items) if item.kind not in CONTEXT_KINDS),
            len(self.view.items),
        )
        attempt = 0
        while True:
            status, payload = self.post(self.codec.summary_request(self.body, items, prompt))
            complete = status < 300 and isinstance(payload, dict) and self.codec.finished(payload)
            if complete:
                text = self.codec.output_text(payload)
                if not text:
                    raise RuntimeError("the summary response contained no text")
                return text
            # Like Codex, a summary that did not complete (a stream cut short, say) is never used.
            if (status < 300 or status in TRANSIENT) and attempt < RETRIES:
                attempt += 1
                time.sleep(2**attempt)  # incomplete, rate limited or overloaded: back off and retry
                continue
            if status < 300:
                raise RuntimeError(f"the summary response did not complete: {payload!r:.300}")
            trimmed = _trim(self.codec, items, keep) if self.codec.is_overflow(status, payload) else None
            if trimmed is None:
                raise RuntimeError(f"summary request failed with HTTP {status}: {payload!r:.300}")
            items = trimmed


def _trim(codec: Codec, items: list[WireItem], keep: int) -> list[WireItem] | None:
    """Replace at least a tenth of the history after `keep` with a short note.

    The note (a user message) keeps the request well-formed wherever the cut falls,
    e.g. Anthropic requires the first message to come from the user.
    """

    sizes = [approx_tokens(codec.classify(item).text) for item in items]
    candidates = sorted(b for b in codec.boundaries(items) if keep < b < len(items))
    if not candidates:
        return None
    boundary = next((b for b in candidates if sum(sizes[keep:b]) * 10 >= sum(sizes)), candidates[-1])
    return [*items[:keep], codec.user_message(TRIM_NOTE), *items[boundary:]]
