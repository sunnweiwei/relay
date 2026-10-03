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
from typing import Any

from ..harnesses import Harness
from ..protocols.base import Body, Codec, WireItem
from ..providers import context_window
from ..strategies.base import Strategy
from .ir import CONTEXT_KINDS, Item, Kind, View
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
    depth: int = 0  # leading items that matched a stored prefix
    diverged: int | None = None  # where the request left its conversation's last compaction, if it did


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
        if harness.compacting(codec, raw):  # the harness summarizing its own history: not ours to touch
            return Exchange(body, b"", [], {})
        keys = harness.identity(codec, raw)
        fingerprint = json.dumps(self.strategy.fingerprint(), sort_keys=True)
        partition = self.store.partition(tenant, codec.name, harness.name, fingerprint)
        depth, state = self.store.match(partition, keys) or (0, {})
        named = [key for key in keys if key not in (b"\0system", b"\0context")][:3]
        thread = partition + hashlib.sha256(b"\0".join(named)).digest()  # names the conversation
        diverged = self._diverged(thread, keys, depth)

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
        if isinstance(state.get("tokens"), int):
            tokens = state["tokens"] + sum(approx_tokens(view_item(i).text) for i in range(depth, len(raw)))
        else:
            tokens = approx_tokens(codec.preamble(body)) + sum(approx_tokens(i.text) for i in items)
            tokens += sum(approx_tokens(codec.classify(item).text) for item in volatile)
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
        if force or time.monotonic() >= self._retry_at.get(thread, 0.0):
            started = time.monotonic()
            try:
                summarizer = _Summarizer(codec, body, lambda cut: wire[: position(cut)], leading, post)
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
                        },
                        state,
                    )
            except Exception:
                log.warning("%s failed; forwarding the request unchanged", self.strategy.name, exc_info=True)
                self._retry_at[thread] = time.monotonic() + self.retry_after

        unchanged = not state.get("head") and not state.get("covered")
        forwarded = body if unchanged else codec.with_items(body, [*wire, *volatile])
        estimate = approx_tokens(codec.preamble(body)) + sum(approx_tokens(i.text) for i in items)
        estimate += sum(approx_tokens(codec.classify(item).text) for item in volatile)
        return Exchange(forwarded, partition, keys, state, compacted, estimate, depth, diverged)

    def record(self, exchange: Exchange, input_tokens: int | None) -> None:
        """Anchor future token estimates to the prompt tokens the upstream reported.

        The first size after a rewrite (or ever) starts a new context window. Upstreams that
        report no usage (some gateways send zeros) fall back to Relay's own estimate."""

        if not exchange.keys:
            return  # passed through untouched
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
        """Where a request left the history its conversation's last compaction covered, when it
        no longer reaches it: the harness rewrote that history in a way its profile's identity
        does not see through. None when the compaction applies (or there is none)."""

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
    """Runs the summary request through the task's own upstream ("native continuation")."""

    def __init__(self, codec: Codec, body: Body, history: Callable[[int], list[WireItem]], keep: int,
                 post: Post) -> None:
        # `history(cut)`: the wire items a conversation cut replaces, the harness's own included;
        # `keep`: the leading ones an overflow must not drop (Codex keeps the initial context).
        self.codec, self.body, self.history, self.keep, self.post = codec, body, history, keep, post

    def summarize(self, cut: int, prompt: str) -> str:
        items, keep = self.history(cut), self.keep
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
