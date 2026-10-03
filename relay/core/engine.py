"""Per-request orchestration.

For each request the engine restores the stored rewrite for the longest known
prefix (compared by the harness's `identity`), builds the view of the conversation the
strategy sees, puts the harness's current `state` into the strategy's rewrite where the
harness `place`s it, lets the codec make it legal, and remembers the result (and later the
prompt tokens the upstream reported) under the request's prefix. Strategy failures never
reach the harness: the request is forwarded with the last good rewrite instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Any

from ..harnesses import Harness
from ..protocols.base import Body, Codec, WireItem
from ..prompts import SUMMARY_PREFIX
from ..providers import context_window
from ..strategies.base import Strategy
from .ir import CONTEXT_KINDS, Item, Kind, View
from .store import PrefixStore
from .tokens import approx_tokens, bytes_per_token, item_tokens

log = logging.getLogger("relay")

Post = Callable[[Body], tuple[int, Any]]  # send a body upstream -> (status, JSON payload)
TRANSIENT = {429, 500, 502, 503, 504}  # summary request errors worth retrying
RETRIES = 5  # Codex's default stream_max_retries
SUMMARY_SHARE = 0.95  # of the context window: whatever Relay would forward uncompacted fits one summary request


@dataclass(frozen=True)
class Exchange:
    """A prepared request: what to forward, and where to record its usage."""

    body: Body
    partition: bytes
    keys: list[bytes]  # canonical identities of the request's wire items
    state: dict[str, Any]  # stored rewrite applied to this request
    compacted: bool = False  # this request produced a new rewrite
    estimate: int = 0  # Relay's own estimate of the forwarded prompt, if usage is never reported
    depth: int = 0  # leading items that matched a stored prefix
    diverged: int | None = None  # where a request that found no compaction left its conversation's last one


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
        # Each conversation's last stored compaction (covered items, digests of their identities),
        # to tell a request that should have found it but did not, and where it diverged.
        self._compacted: OrderedDict[bytes, tuple[int, list[bytes]]] = OrderedDict()

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
        # The harness summarizing its own history: it summarizes what the model has been seeing
        # (the stored compaction still applies), but is never compacted anew or measured.
        compacting = harness.compacting(codec, raw)
        keys = harness.identity(codec, raw)
        fingerprint = json.dumps(self.strategy.fingerprint(), sort_keys=True)
        partition = self.store.partition(tenant, codec.name, harness.name, fingerprint)
        depth, state = self.store.match(partition, keys) or (0, {})
        named = [key for key in keys if key not in (b"\0system", b"\0context")][:3]
        thread = partition + hashlib.sha256(b"\0".join(named)).digest()  # names the conversation
        diverged = None if state.get("covered") or compacting else self._diverged(thread, keys, depth)

        def view_item(index: int) -> Item:
            return replace(harness.refine(codec.classify(raw[index])), ref=index)

        def materialize(state: dict[str, Any]) -> tuple[list[Item], list[WireItem]]:
            head = [view_item(h["ref"]) if "ref" in h else Item(Kind(h["kind"]), h["text"], wire=h.get("wire"))
                    for h in state.get("head", [])]
            items = head + [view_item(i) for i in range(state.get("covered", 0), len(raw))]
            wire = [raw[i.ref] if i.ref is not None else json.loads(i.wire) if i.wire else codec.user_message(i.text)
                    for i in items]
            return items, wire

        items, wire = materialize(state)
        per_token = bytes_per_token(body.get("model"))  # the model's tokenizer, roughly

        def size(items: list[Item]) -> int:  # the request's own preamble included
            return approx_tokens(codec.preamble(body), per_token) + sum(item_tokens(i, per_token) for i in items)

        volatile_items = [codec.classify(item) for item in volatile]
        if isinstance(state.get("tokens"), int):  # the upstream's count for the stored prefix, then new items
            tokens = state["tokens"] + sum(item_tokens(view_item(i), per_token) for i in range(depth, len(raw)))
        else:
            tokens = size([*items, *volatile_items])
        # The strategy sees the conversation; what the harness wrote itself is its profile's.
        conversation = [n for n, item in enumerate(items) if item.kind not in CONTEXT_KINDS]
        legal = codec.boundaries(wire)

        def position(cut: int) -> int:  # a cut of the conversation, as a cut of `items`
            return conversation[cut] if cut < len(conversation) else len(items)

        view = View(
            tuple(items[n] for n in conversation),
            frozenset(cut for cut in range(len(conversation) + 1) if position(cut) in legal),
            tokens,
            self.window or context_window(body.get("model")),
            force,
            state.get("base"),
        )
        leading = next((n for n, item in enumerate(items) if item.kind not in CONTEXT_KINDS), len(items))

        compacted = False
        if not compacting and (force or time.monotonic() >= self._retry_at.get(thread, 0.0)):
            started = time.monotonic()
            try:
                # Per-item estimates, scaled to the upstream's count of the request when there is one.
                scale = tokens / max(1, size([*items, *volatile_items]))
                summarizer = _Summarizer(codec, body, lambda cut: wire[: position(cut)], leading, post,
                                         view.window and view.window / scale, per_token)
                rewrite = self.strategy.plan(view, summarizer)
                if rewrite is not None:
                    state = _to_state(codec, harness, raw, items, legal, position(rewrite.cut), rewrite.head)
                    self.store.put(partition, keys, state)
                    self._remember(thread, state["covered"], keys)
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
                            "summary_requests": summarizer.requests,
                        },
                        state,
                    )
            except Exception:
                log.warning("%s failed; forwarding the request unchanged", self.strategy.name, exc_info=True)
                self._retry_at[thread] = time.monotonic() + self.retry_after

        unchanged = not state.get("head") and not state.get("covered")
        forwarded = body if unchanged else codec.with_items(body, [*wire, *volatile])
        estimate = size([*items, *volatile_items])
        return Exchange(forwarded, partition, [] if compacting else keys, state, compacted, estimate, depth, diverged)

    def record(self, exchange: Exchange, input_tokens: int | None) -> None:
        """Anchor future token estimates to the prompt tokens the upstream reported.

        The first size after a rewrite (or ever) starts a new context window. Upstreams that
        report no usage (some gateways send zeros) fall back to Relay's own estimate."""

        if not exchange.keys:
            return  # the harness compacting itself: its usage says nothing about the conversation
        state = {k: v for k, v in exchange.state.items() if k != "tokens"}
        if "base" not in state or (input_tokens and state.get("estimated")):
            state["base"], state["estimated"] = input_tokens or exchange.estimate, not input_tokens
        if input_tokens:
            state["tokens"] = input_tokens
        self.store.put(exchange.partition, exchange.keys, state)

    def _remember(self, thread: bytes, covered: int, keys: list[bytes]) -> None:
        self._compacted[thread] = (covered, [hashlib.sha256(key).digest() for key in keys[:covered]])
        self._compacted.move_to_end(thread)
        while len(self._compacted) > 4096:
            self._compacted.popitem(last=False)

    def _diverged(self, thread: bytes, keys: list[bytes], depth: int) -> int | None:
        """Where a request that found no stored compaction left the history its conversation's
        last compaction covered: the harness rewrote that history in a way its profile's
        identity does not see through. (A rewind or a branch finds an earlier compaction.)"""

        covered, digests = self._compacted.get(thread, (0, []))
        if not covered or depth >= covered:
            return None
        index = next((i for i, digest in enumerate(digests) if i >= len(keys) or hashlib.sha256(keys[i]).digest() != digest),
                     len(digests))
        log.warning("a stored compaction no longer applies: the request diverges from it at item %d", index)
        return index

    def _emit(self, event: dict[str, Any], state: dict[str, Any]) -> None:
        log.info("rewrote context: %s", event)
        if self.event_log:
            record = {"time": time.time(), **event, **state}
            try:
                with open(self.event_log, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                log.warning("could not write the event log %s", self.event_log, exc_info=True)


def _to_state(codec: Codec, harness: Harness, raw: list[WireItem], items: list[Item], legal: frozenset[int],
              cut: int, head: tuple[Item, ...]) -> dict[str, Any]:
    """Express a strategy's rewrite of the conversation (`items[:cut]` become `head`) as a stored
    rewrite of the request: the harness's current state placed into the head, the context it
    supersedes right after the cut absorbed, and the protocol's rules applied."""

    if cut not in legal:
        raise ValueError(f"rewrite cut {cut} splits an atomic group")
    current = harness.state(codec, raw)
    while cut < len(items) and items[cut].ref in current.supersedes and cut + 1 in legal:
        cut += 1  # e.g. the context updates that open the pending turn
    if cut < len(items) and items[cut].ref is None:
        raise ValueError("rewrite cut falls inside Relay's own items")
    covered = items[cut].ref if cut < len(items) else len(raw)
    mid_turn = cut == len(items)
    kept = tuple(item for item in current.items if item.ref is None or item.ref < covered)  # the rest is still there
    entries = []
    for item in codec.arrange(harness.place(head, kept, mid_turn), mid_turn):
        if item.ref is None:
            entries.append({"kind": item.kind.value, "text": item.text, **({"wire": item.wire} if item.wire else {})})
        elif item.ref < covered:
            entries.append({"ref": item.ref})
        else:
            raise ValueError("rewrite keeps an item that it does not replace")
    return {"covered": covered, "head": entries}


class _Summarizer:
    """Runs the summary request through the task's own upstream ("native continuation").

    A history too long for one request (Relay joined a long session late, or lost its state,
    while the harness kept every item) is summarized in order, a piece at a time, each piece
    after the summary of what came before it: the compactions that were skipped, caught up.
    Codex instead drops the oldest items until the request fits; nothing is dropped here."""

    def __init__(self, codec: Codec, body: Body, history: Callable[[int], list[WireItem]], keep: int,
                 post: Post, window: float | None, per_token: float) -> None:
        # `history(cut)`: the wire items a conversation cut replaces, the harness's own included;
        # `keep`: the leading ones every piece carries (Codex keeps the initial context);
        # `window`: in the per-item estimates' terms.
        self.codec, self.body, self.history, self.keep, self.post = codec, body, history, keep, post
        self.budget = int(window * SUMMARY_SHARE) if window else None
        self.per_token, self.requests = per_token, 0

    def summarize(self, cut: int, prompt: str) -> str:
        items = self.history(cut)
        lead, rest, budget, summary = items[: self.keep], items[self.keep :], self.budget, ""
        while True:
            fixed = [*lead, self.codec.user_message(f"{SUMMARY_PREFIX}\n{summary}")] if summary else lead
            overhead = approx_tokens(self.codec.preamble(self.body), self.per_token) + sum(map(self._tokens, fixed))
            piece = _piece(self.codec, rest, budget - overhead, self._tokens) if budget else len(rest)
            status, payload = self._request([*fixed, *rest[:piece]], prompt)
            if self.codec.is_overflow(status, payload):
                if piece <= _piece(self.codec, rest, 0, self._tokens):
                    raise RuntimeError("a single turn of the history does not fit in a summary request")
                budget = overhead + sum(map(self._tokens, rest[:piece])) // 2  # the estimate ran short
                continue
            if status >= 300:
                raise RuntimeError(f"summary request failed with HTTP {status}: {payload!r:.300}")
            if not (isinstance(payload, dict) and self.codec.finished(payload)):
                raise RuntimeError(f"the summary response did not complete: {payload!r:.300}")  # never used, like Codex
            if not (summary := self.codec.output_text(payload)):
                raise RuntimeError("the summary response contained no text")
            rest = rest[piece:]
            if not rest:
                return summary

    def _request(self, items: list[WireItem], prompt: str) -> tuple[int, Any]:
        """One summary request, retried while rate limited, overloaded or cut short."""

        attempt = 0
        while True:
            self.requests += 1
            status, payload = self.post(self.codec.summary_request(self.body, items, prompt))
            incomplete = status < 300 and not (isinstance(payload, dict) and self.codec.finished(payload))
            if not (incomplete or status in TRANSIENT) or attempt == RETRIES:
                return status, payload
            attempt += 1
            time.sleep(2**attempt)

    def _tokens(self, item: WireItem) -> int:
        return item_tokens(self.codec.classify(item), self.per_token)


def _piece(codec: Codec, items: list[WireItem], room: int, tokens: Callable[[WireItem], int]) -> int:
    """The longest leading run of `items` that ends at a legal boundary and fits in `room`
    tokens; at least the shortest one."""

    legal = sorted(b for b in codec.boundaries(items) if b > 0) or [len(items)]
    total = [0, *accumulate(map(tokens, items))]
    fitting = [b for b in legal if total[b] <= room]
    return fitting[-1] if fitting else legal[0]
