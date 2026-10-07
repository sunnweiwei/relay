"""Running a strategy on every request: through the proxy (`prepare`) or a harness hook (`answer`).

For each request the engine restores the context stored for the longest known prefix (compared
by the harness's `identity`): what the model saw last time, followed by what the harness sent
since. The strategy gets the conversation (`Request`) and answers with the context the model sees
from now on (`Context`). The engine puts the harness's own items into it (where they were, or
where the harness puts them after its own compaction when the answer has a summary), tells as a
note each item the API would reject where it stands (a call whose result is gone), and remembers
it, and later the prompt tokens the upstream reported,
under the request's prefix. Strategy failures never reach the harness: the request is forwarded
with the last good context instead.

A harness hook hands over its whole conversation before a request instead (`answer`): the
strategy decides on it, and the harness keeps the context, so only the strategy's state is
stored here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import accumulate
from typing import Any

from ..harnesses import Harness
from ..protocols.base import Body, Codec, WireItem
from ..prompts import SUMMARY_PREFIX
from ..providers import context_window
from ..strategies.base import Strategy, Summarizer
from .ir import CONTEXT_KINDS, LABELS, Context, Item, Kind, Media, Request, note
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
    state: dict[str, Any]  # stored context applied to this request
    compacted: bool = False  # the strategy changed the context on this request
    estimate: int = 0  # Relay's own estimate of the forwarded prompt, if usage is never reported
    depth: int = 0  # leading items that matched a stored prefix
    diverged: int | None = None  # where a request that found no stored context left its conversation's last one


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
        self.store = PrefixStore() if store is None else store  # a store is falsy while it is empty
        self.window = window
        self.event_log = event_log
        self.retry_after = retry_after
        self._retry_at: dict[bytes, float] = {}  # per conversation, after a failure
        # Each conversation's last stored context (items covered, digests of their identities),
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
        # forwarded as they are, but kept out of prefix matching and the strategy's view.
        end = len(raw)
        while end and harness.volatile(harness.refine(codec.classify(raw[end - 1]))):
            end -= 1
        raw, volatile = raw[:end], raw[end:]
        # The harness summarizing its own history: it summarizes what the model has been seeing
        # (the stored context still applies), but the strategy is not asked, nor the usage kept.
        compacting = harness.compacting(codec, raw)
        keys = harness.identity(codec, raw)
        fingerprint = json.dumps(self.strategy.fingerprint(), sort_keys=True)
        partition = self.store.partition(tenant, codec.name, harness.name, fingerprint)
        depth, state = self.store.match(partition, keys) or (0, {})
        named = [key for key in keys if key not in (b"\0system", b"\0context")][:3]
        thread = partition + hashlib.sha256(b"\0".join(named)).digest()  # names the conversation
        diverged = None if state.get("covered") or compacting else self._diverged(thread, keys, depth)
        originals = [replace(harness.refine(codec.classify(item)), ref=index) for index, item in enumerate(raw)]

        def materialize(state: dict[str, Any]) -> tuple[list[Item], list[WireItem]]:
            items = [_item(h, originals) for h in state.get("head", [])] + originals[state.get("covered", 0):]
            return items, [_wire(codec, raw, originals, item) for item in items]

        items, wire = materialize(state)
        per_token = bytes_per_token(body.get("model"))  # the model's tokenizer, roughly

        def size(items: list[Item]) -> int:  # the request's own preamble included
            return approx_tokens(codec.preamble(body), per_token) + sum(item_tokens(i, per_token) for i in items)

        volatile_items = [codec.classify(item) for item in volatile]
        if isinstance(state.get("tokens"), int):  # the upstream's count for the stored prefix, then new items
            tokens = state["tokens"] + sum(item_tokens(originals[i], per_token) for i in range(depth, len(raw)))
        else:
            tokens = size([*items, *volatile_items])
        # The strategy sees the conversation; what the harness wrote itself is its profile's.
        conversation = [n for n, item in enumerate(items) if item.kind not in CONTEXT_KINDS]
        legal = codec.boundaries(wire)

        def position(cut: int) -> int:  # a cut of the conversation, as a cut of `items`
            return conversation[cut] if cut < len(conversation) else len(items)

        current = tuple(items[n] for n in conversation)
        accepted = {raw_index for raw_index in codec.orphans(raw)}  # the harness's own history, as the API took it

        def sendable(answer: tuple[Item, ...]) -> tuple[Item, ...]:
            return tuple(item for item in _sendable(codec, harness, raw, originals, volatile, accepted, answer)
                         if item.kind not in CONTEXT_KINDS or _instruction(item))

        request = Request(
            tuple(item for item in originals if item.kind not in CONTEXT_KINDS),
            current,
            frozenset(cut for cut in range(len(current) + 1) if position(cut) in legal),
            tokens,
            self.window or context_window(body.get("model")),
            force,
            state.get("base"),
            state.get("strategy"),
            hashlib.sha256(thread).hexdigest()[:16],
            codec.offers_tools(body),
            sendable,
        )
        leading = next((n for n, item in enumerate(items) if item.kind not in CONTEXT_KINDS), len(items))

        changed, notes, instructions = False, (), list(state.get("instructions", []))
        if not compacting and (force or time.monotonic() >= self._retry_at.get(thread, 0.0)):
            started = time.monotonic()
            try:
                # Per-item estimates, scaled to the upstream's count of the request when there is one.
                scale = tokens / max(1, size([*items, *volatile_items]))
                summarizer = _Summarizer(codec, body, lambda cut: wire[: position(cut)], leading, post,
                                         request.window and request.window / scale, per_token)
                if (context := self.strategy.plan(request, summarizer)) is not None:
                    answer = sendable(tuple(item for item in context.items if not _instruction(item)))
                    answered = [item.text for item in context.items if _instruction(item)]
                    stored = dict(state)
                    if answer != current:
                        placed = _place(codec, harness, raw, originals, answer)
                        stored = {"covered": len(raw), "head": [_entry(item, originals) for item in placed]}
                        new_items, new_wire = materialize(stored)
                        if lonely := {n for n in codec.orphans([*new_wire, *volatile]) if n >= len(new_items)
                                      or new_items[n].ref not in accepted}:
                            raise ValueError(f"the context would be rejected: items {sorted(lonely)} where they stand")
                        self._remember(thread, len(raw), keys)
                        items, wire, changed = new_items, new_wire, True
                        self.emit(
                            {
                                "strategy": self.strategy.name,
                                "protocol": codec.name,
                                "harness": harness.name,
                                "model": body.get("model"),
                                "forced": force,
                                "tokens_before": tokens,
                                "items_before": len(current),
                                "items_after": len(answer),
                                "seconds": round(time.monotonic() - started, 3),
                                "summary_requests": summarizer.requests,
                            },
                            stored,
                        )
                    if changed or context.state != request.state or answered != instructions:
                        stored = {k: v for k, v in stored.items() if k not in ("strategy", "instructions")}
                        stored.update({"strategy": context.state} if context.state is not None else {})
                        stored.update({"instructions": answered} if answered else {})
                        self.store.put(partition, keys, stored)
                        state = stored
                    notes, instructions = context.notes, answered
            except Exception:
                log.warning("%s failed; forwarding the last context", self.strategy.name, exc_info=True)
                self._retry_at[thread] = time.monotonic() + self.retry_after

        # Notes for this request alone at the end; the strategy's instructions join the system
        # prompt (the same on every request, so the prompt cache still holds).
        sent = [*wire, *volatile]
        if notes:
            sent = codec.note(sent, "\n\n".join(notes))
        unchanged = not state.get("head") and not state.get("covered") and not notes
        forwarded = body if unchanged else codec.with_items(body, sent)
        if instructions:
            forwarded = codec.with_instructions(forwarded, "\n\n".join(instructions))
        estimate = size([*items, *volatile_items]) + approx_tokens(" ".join([*notes, *instructions]), per_token)
        return Exchange(forwarded, partition, [] if compacting else keys, state, changed, estimate, depth, diverged)

    def answer(self, request: Request, summarizer: Summarizer) -> Context | None:
        """The strategy's answer on a conversation a harness hook hands over before a request; the
        harness keeps the context, the strategy's state is kept here by conversation."""

        partition = self.store.partition("hook", json.dumps(self.strategy.fingerprint(), sort_keys=True))
        key = [request.conversation.encode()]
        _, state = self.store.match(partition, key) or (0, {})
        request = replace(request, state=state.get("strategy"))
        context = self.strategy.plan(request, summarizer)
        if context is not None and context.state != request.state:
            self.store.put(partition, key, {"strategy": context.state})
        return context

    def record(self, exchange: Exchange, input_tokens: int | None) -> None:
        """Anchor future token estimates to the prompt tokens the upstream reported.

        The first size after a change of context (or ever) starts a new context window. Upstreams
        that report no usage (some gateways send zeros) fall back to Relay's own estimate."""

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
        """Where a request that found no stored context left the history its conversation's last
        one covered: the harness rewrote that history in a way its profile's identity does not
        see through. (A rewind or a branch finds an earlier context.)"""

        covered, digests = self._compacted.get(thread, (0, []))
        if not covered or depth >= covered:
            return None
        index = next((i for i, digest in enumerate(digests) if i >= len(keys) or hashlib.sha256(keys[i]).digest() != digest),
                     len(digests))
        log.warning("a stored context no longer applies: the request diverges from it at item %d", index)
        return index

    def emit(self, event: dict[str, Any], state: dict[str, Any]) -> None:
        log.info("changed the context: %s", event)
        if self.event_log:
            record = {"time": time.time(), **event, **state}
            try:
                with open(self.event_log, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                log.warning("could not write the event log %s", self.event_log, exc_info=True)


def _instruction(item: Item) -> bool:
    """A SYSTEM item the strategy wrote: it joins the system prompt."""

    return item.kind is Kind.SYSTEM and item.ref is None


def _sendable(codec: Codec, harness: Harness, raw: list[WireItem], originals: list[Item], volatile: list[WireItem],
              accepted: set[int], answer: tuple[Item, ...]) -> list[Item]:
    """The answer as it can be sent, its harness items placed: each item the protocol cannot keep
    where it stands (a call whose result is gone, reasoning away from the item it preceded) is told
    as a note (one with no text goes), the rest stays as the strategy gave it. What the harness's
    own history already had (`accepted`) is the API's to judge."""

    def told(item: Item) -> Item | None:
        return note(LABELS.get(item.kind, item.kind.value), item.text) if item.text.strip() else None

    def keeps(item: Item) -> bool:  # new content for a call, say, cannot change on its own
        try:
            _wire(codec, raw, originals, item)
            return True
        except ValueError:
            return False

    placed = _place(codec, harness, raw, originals, tuple(i for item in answer if (i := item if keeps(item) else told(item))))
    while True:
        wire = [_wire(codec, raw, originals, item) for item in placed]
        lonely = {n for n in codec.orphans([*wire, *volatile])
                  if n < len(placed) and placed[n].kind not in CONTEXT_KINDS and placed[n].ref not in accepted}
        lonely |= {n for n, item in enumerate(placed)  # reasoning, only right before what it preceded
                   if item.kind is Kind.REASONING and item.ref is not None
                   and (n + 1 == len(placed) or placed[n + 1].ref != item.ref + 1)}
        if not lonely:
            return placed
        placed = [i for n, item in enumerate(placed) if (i := told(item) if n in lonely else item)]


def _place(codec: Codec, harness: Harness, raw: list[WireItem], originals: list[Item],
           answer: tuple[Item, ...]) -> list[Item]:
    """The answer with the harness's own items put in. They stay where the harness sent them,
    those before the conversation in front; but what a summary stands in for is replaced by the
    harness's current state, placed as the harness places it after its own compaction."""

    own = [item for item in originals if item.kind in CONTEXT_KINDS]
    summary = max((n for n, item in enumerate(answer) if item.kind is Kind.SUMMARY), default=None)
    if summary is None:
        first = next((item.ref for item in originals if item.kind not in CONTEXT_KINDS), len(raw))
        return [*(item for item in own if item.ref < first), *_merge(answer, [i for i in own if i.ref >= first])]
    head, tail = answer[: summary + 1], answer[summary + 1 :]
    cut = min((item.ref for item in tail if item.ref is not None), default=len(raw))
    state = tuple(item for item in harness.state(codec, raw).items if item.ref is None or item.ref < cut)
    mid_turn = cut == len(raw)
    return [*codec.arrange(harness.place(head, state, mid_turn), mid_turn), *_merge(tail, [i for i in own if i.ref >= cut])]


def _merge(items: tuple[Item, ...], own: list[Item]) -> list[Item]:
    """The harness's items back among the answer's, each before the first answer item from
    later in the request; those after every one at the end."""

    merged, pending = [], list(own)
    for item in items:
        while item.ref is not None and pending and pending[0].ref < item.ref:
            merged.append(pending.pop(0))
        merged.append(item)
    return merged + pending


def _entry(item: Item, originals: list[Item]) -> dict[str, Any]:
    """How a context item is stored: a request item by index (with new content if it has any),
    or an item Relay writes."""

    media = {"media": [asdict(m) for m in item.media]}
    if item.ref is not None:
        original = originals[item.ref]
        return {"ref": item.ref} if (item.text, item.media) == (original.text, original.media) else {
            "ref": item.ref, "text": item.text, **media}
    return {"kind": item.kind.value, "text": item.text, **(media if item.media else {}),
            **({"wire": item.wire} if item.wire else {})}


def _item(entry: dict[str, Any], originals: list[Item]) -> Item:
    media = tuple(Media(**m) for m in entry.get("media", []))
    if "ref" in entry:
        original = originals[entry["ref"]]
        return replace(original, text=entry["text"], media=media) if "text" in entry else original
    return Item(Kind(entry["kind"]), entry["text"], media=media, wire=entry.get("wire"))


def _wire(codec: Codec, raw: list[WireItem], originals: list[Item], item: Item) -> WireItem:
    if item.ref is not None:
        original = originals[item.ref]
        unchanged = (item.text, item.media) == (original.text, original.media)
        return raw[item.ref] if unchanged else codec.edit(raw[item.ref], item.text, item.media)
    return json.loads(item.wire) if item.wire else codec.write(item)


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
            status, payload = self._post(self.codec.summary_request(self.body, [*fixed, *rest[:piece]], prompt))
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

    def complete(self, items: Sequence[Item]) -> str:
        system = "\n\n".join(item.text for item in items if item.kind is Kind.SYSTEM)
        wire = [self.codec.write(item) for item in items if item.kind is not Kind.SYSTEM]
        status, payload = self._post(self.codec.request(self.body, system, wire))
        if status >= 300 or not (isinstance(payload, dict) and self.codec.finished(payload)):
            raise RuntimeError(f"the model request failed with HTTP {status}: {payload!r:.300}")
        return self.codec.output_text(payload)

    def _post(self, request: Body) -> tuple[int, Any]:
        """One request of Relay's own, retried while rate limited, overloaded or cut short."""

        attempt = 0
        while True:
            self.requests += 1
            status, payload = self.post(request)
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
