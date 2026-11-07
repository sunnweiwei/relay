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
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import accumulate
from typing import Any

from .. import private
from ..harnesses import Harness
from ..protocols.base import Body, Codec, WireItem
from ..prompts import SUMMARY_PREFIX
from ..providers import context_window, window_of
from ..strategies.base import Strategy, Summarizer
from .ir import CONTEXT_KINDS, LABELS, Context, Item, Kind, Media, Request, note
from .media import removed
from .store import PrefixStore
from .tokens import approx_tokens, bytes_per_token, counting_with, item_tokens
from .budget import BudgetExceeded, InputBudget
from .truncation import truncate_request

log = logging.getLogger("relay")
RATE = (0.5, 4.0)  # how far the upstream's count may scale Relay's estimate of new items

Post = Callable[[Body], tuple[int, Any]]  # send a body upstream -> (status, JSON payload)
TRANSIENT = {429, 500, 502, 503, 504}  # summary request errors worth retrying
RETRIES = 5  # Codex's default stream_max_retries
SENDABLE = 0.95  # of the context window: the most a context Relay adds to may reach (Codex's effective window)
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
    conversation: str = ""  # the conversation's name, as strategies see it
    title: str = ""  # what its user first typed
    whole: int = 0  # the harness's own history as it sent it, estimated (tokens)
    compacting: bool = False  # the harness compacting its own history
    direct: bool = False  # forwarded as the harness sent it, as going direct would (`direct_within`)
    lasting: str = ""  # the conversation's name whatever the strategy (`conversation` changes with it)


class Engine:
    def __init__(
        self,
        strategy: Strategy,
        store: PrefixStore | None = None,
        *,
        window: int | None = None,
        event_log: str | None = None,
        retry_after: float = 60.0,
        input_budget: InputBudget | None = None,
        harness_profile: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.store = PrefixStore() if store is None else store  # a store is falsy while it is empty
        self.window = window
        self.event_log = event_log
        self.retry_after = retry_after
        self.input_budget = input_budget
        self.harness_profile = harness_profile
        self._retry_at: dict[bytes, float] = {}  # per conversation, after a failure
        # Each conversation's last stored context (items covered, digests of their identities),
        # to tell a request that should have found it but did not, and where it diverged.
        self._compacted: OrderedDict[bytes, tuple[int, list[bytes]]] = OrderedDict()

    def prepare(
        self, codec: Codec, harness: Harness, body: Body, *, tenant: str,
        post: Post, force: bool = False, direct_within: int | Callable[[str], int | None] | None = None,
        fallback_limit: int | None = None,
        on_fallback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Exchange:
        """`input_budget` (an opt-in bounded evaluation): every count uses its declared counter,
        and the request that leaves must fit it. `harness_profile`: an evaluation-only rewrite of
        known harness boilerplate (relay/harnesses/browse_profile.py), applied before matching."""
        if self.harness_profile:
            from ..harnesses.browse_profile import apply_profile

            body = apply_profile(body, self.harness_profile)
        with counting_with(self.input_budget.text_tokens if self.input_budget else None,
                           self.input_budget.opaque_tokens if self.input_budget else None,
                           self.input_budget.media_item_tokens
                           if self.input_budget and self.input_budget.name in self.input_budget.MEDIA_COUNTER_NAMES
                           else None):
            exchange = self._prepare(codec, harness, body, tenant=tenant, post=post,
                                     force=force, direct_within=direct_within,
                                     fallback_limit=fallback_limit, on_fallback=on_fallback)
            if self.input_budget:
                self.input_budget.check(harness.codec(codec), exchange.body, "main")
            return exchange

    def _prepare(
        self,
        codec: Codec,
        harness: Harness,
        body: Body,
        *,
        tenant: str,
        post: Post,
        force: bool = False,
        direct_within: int | Callable[[str], int | None] | None = None,
        fallback_limit: int | None = None,
        on_fallback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Exchange:
        """`direct_within`: a harness leaving Relay. A request whose own history (as the harness sent
        it) is estimated within this many tokens goes as it came, without the stored context or the
        strategy: what going direct would send, and it fits. A function of the conversation's lasting
        name gives each conversation's own (less where the upstream refused it going direct)."""

        codec = harness.codec(codec)
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
        fingerprint_data = self.strategy.fingerprint()
        if self.harness_profile:
            fingerprint_data = {**fingerprint_data, "harness_profile": self.harness_profile}
        if self.input_budget:
            fingerprint_data = {**fingerprint_data, "input_budget": self.input_budget.fingerprint()}
        fingerprint = json.dumps(fingerprint_data, sort_keys=True)
        partition = self.store.partition(tenant, codec.name, harness.name, fingerprint)
        depth, state = self.store.match(partition, keys) or (0, {})
        named = [key for key in keys if key not in (b"\0system", b"\0context")][:3]
        thread = partition + hashlib.sha256(b"\0".join(named)).digest()  # names the conversation
        lasting = hashlib.sha256(self.store.partition(tenant, codec.name, harness.name)
                                 + hashlib.sha256(b"\0".join(named)).digest()).hexdigest()[:16]
        latest = self.store.partition("latest", tenant, codec.name, harness.name)  # lasting name -> its last context's partition
        if compacting and not state.get("covered"):  # (the strategy changed since: the context the model last saw)
            pointer = (self.store.match(latest, [lasting.encode()]) or (0, {}))[1].get("partition")
            if pointer and bytes.fromhex(pointer) != partition:
                depth, state = self.store.match(bytes.fromhex(pointer), keys) or (depth, state)
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
        name = hashlib.sha256(thread).hexdigest()[:16]
        title = next((typed.text[:80] for item in originals if item.kind is Kind.USER
                      and (typed := harness.typed(item)) is not None and typed.text.strip()), "")
        whole = (self.input_budget.count(codec, body) if self.input_budget else
                 math.ceil(size([*originals, *volatile_items]) * state.get("rate", 1.0)))
        if callable(direct_within):
            direct_within = direct_within(lasting)
        if direct_within is not None and not compacting and whole <= direct_within:
            return Exchange(body, partition, [], state, estimate=whole, conversation=name, title=title, whole=whole,
                            direct=True, lasting=lasting)
        if self.input_budget:
            counted_body = codec.with_items(body, [*wire, *volatile])
            if state.get("instructions"):
                counted_body = codec.with_instructions(counted_body, "\n\n".join(state["instructions"]))
            tokens = self.input_budget.count(codec, counted_body)
        elif isinstance(state.get("tokens"), int):  # the upstream's count for the stored prefix, then new items,
            # estimated at the rate the upstream's count of this conversation has shown (the text's own
            # density: code, another language, numbers)
            new = sum(item_tokens(originals[i], per_token) for i in range(depth, len(raw)))
            tokens = state["tokens"] + math.ceil(new * state.get("rate", 1.0))
        else:
            tokens = size([*items, *volatile_items])
        # The strategy sees the conversation; what the harness wrote itself is its profile's.
        conversation = [n for n, item in enumerate(items) if item.kind not in CONTEXT_KINDS]
        legal = codec.boundaries(wire)

        def position(cut: int) -> int:  # a cut of the conversation, as a cut of `items`
            return conversation[cut] if cut < len(conversation) else len(items)

        current = tuple(items[n] for n in conversation)
        # The strategy sees what the user typed of a user message (one a strategy changed before
        # included); the harness's blocks in it are its state (`place`), forwarded as they came
        # unless the strategy changes the message.
        shown: list[tuple[Item, Item]] = []  # a user message as the strategy sees it, and as it is
        typed: dict[int, tuple[Item, Item]] = {}  # the last of them of each wire item

        def seen(item: Item) -> Item:
            if item.kind is not Kind.USER or (view := harness.typed(item)) in (item, None):
                return item
            shown.append((view, item))
            if item.ref is not None:
                typed[item.ref] = (view, item)
            return view

        def unseen(item: Item) -> Item:
            if item.kind is not Kind.USER:
                return item
            if (under := next((under for view, under in shown if view == item), None)) is not None:
                return under  # (as the strategy saw it)
            if item.ref not in typed:
                return item
            view, under = typed[item.ref]
            text = (under.text.replace(view.text, item.text, 1) if view.text in under.text
                    else "\n\n".join([*harness.blocks(under.text), item.text]))  # (its blocks kept)
            return replace(item, text=text)
        accepted = {raw_index for raw_index in codec.orphans(raw)}  # the harness's own history, as the API took it

        def sendable(answer: tuple[Item, ...]) -> tuple[Item, ...]:
            answer = tuple(map(unseen, answer))
            return tuple(item for item in _sendable(codec, harness, raw, originals, volatile, accepted, answer)
                         if item.kind not in CONTEXT_KINDS or _instruction(item))

        request = Request(
            tuple(seen(item) for item in originals if item.kind not in CONTEXT_KINDS),
            tuple(map(seen, current)),
            frozenset(cut for cut in range(len(current) + 1) if position(cut) in legal),
            tokens,
            (self.input_budget.input_limit if self.input_budget else
             window_of(body.get("model"), self.window or harness.window(body.get("model"))
                       or context_window(body.get("model")))),
            force,
            state.get("base"),
            state.get("strategy"),
            name,
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
                # A window the upstream has taken a larger request of is no measure of what fits:
                # then the history goes in one summary request, and an overflow splits it.
                window = None if request.window and state.get("tokens", 0) > request.window else request.window
                summarizer = _Summarizer(codec, body, lambda cut: wire[: position(cut)], leading, post,
                                         window and window / scale, per_token, "\n\n".join(instructions),
                                         input_budget=self.input_budget, on_fallback=on_fallback)
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
                                "summary_usage": summarizer.usage,
                            },
                            stored,
                        )
                    if changed or context.state != request.state or answered != instructions:
                        stored = {k: v for k, v in stored.items() if k not in ("strategy", "instructions")}
                        stored.update({"strategy": context.state} if context.state is not None else {})
                        stored.update({"instructions": answered} if answered else {})
                        self.store.put(partition, keys, stored)
                        self.store.put(latest, [lasting.encode()], {"partition": partition.hex()})
                        state = stored
                    notes, instructions = context.notes, answered
            except BudgetExceeded as error:
                # Local rejection must reach the same recovery path as an
                # upstream context error. Never forward the old context here.
                if not (force and self.input_budget and self.input_budget.overflow_fallback == "truncate"):
                    raise
                if on_fallback:
                    on_fallback({"scope": error.scope, "action": "method_recovery_exhausted",
                                 "input_tokens": error.tokens, "input_limit": error.limit})
            except Exception:
                if self.input_budget:
                    raise  # A bounded evaluation cannot silently bypass a failed manager.
                log.warning("%s failed; forwarding the last context", self.strategy.name, exc_info=True)
                self._retry_at[thread] = time.monotonic() + self.retry_after

        # What the harness's tools answered from its own history, where the context no longer has
        # it, answered again: on every request, as the context is (not stored).
        recalled = harness.recalled(items, originals, raw)
        # (never past what fits, nor on a retry after the upstream found the request too long: the
        # pointer the harness wrote stays, as the model would have seen it anyway)
        fits = request.window is None or size([*recalled, *volatile_items]) * state.get("rate", 1.0) < SENDABLE * request.window
        answers_again = recalled is not items and fits and not force
        if answers_again:
            wire = [w if new is old else _wire(codec, raw, originals, new) for w, old, new in zip(wire, items, recalled)]
            items = recalled
        # Notes for this request alone at the end; the strategy's instructions join the system
        # prompt (the same on every request, so the prompt cache still holds).
        sent = [*wire, *volatile]
        if notes:
            sent = codec.note(sent, "\n\n".join(notes))
        unchanged = not state.get("head") and not state.get("covered") and not notes and not answers_again
        forwarded = body if unchanged else codec.with_items(body, sent)
        if instructions:
            forwarded = codec.with_instructions(forwarded, "\n\n".join(instructions))
        estimate = size([*items, *volatile_items]) + approx_tokens(" ".join([*notes, *instructions]), per_token)
        if self.input_budget:
            estimate = self.input_budget.count(codec, forwarded)
            target = self.input_budget.limit_for(forwarded)
            if fallback_limit is not None:
                target = min(target, fallback_limit)
            if force and self.input_budget.overflow_fallback == "truncate" and estimate > target:
                # Preserve rendered instructions, volatile harness context, and user tasks.
                # Persist only the history portion: per-request notes/instructions are rendered
                # afresh, so storing them in head would duplicate them on the next request.
                positions = {id(value): n for n, value in enumerate(wire)}
                outgoing = codec.items(forwarded)
                protected = {n for n, value in enumerate(outgoing)
                             if id(value) not in positions or items[positions[id(value)]].kind in
                             (*CONTEXT_KINDS, Kind.USER)}
                trimmed, report = truncate_request(codec, forwarded, self.input_budget,
                    scope="main", protected=protected, target_limit=target)
                retained = []
                for source_index, value in zip(report["kept_indices"], codec.items(trimmed)):
                    if (index := positions.get(id(outgoing[source_index]))) is not None:
                        original = items[index]
                        typed_item = codec.classify(value)
                        retained.append(replace(original, text=typed_item.text, media=typed_item.media))
                stored = {k: v for k, v in state.items()
                          if k not in {"head", "covered", "tokens", "budget_tokens", "provider_tokens", "base"}}
                stored.update(covered=len(raw), head=[_entry(item, originals) for item in retained])
                self.store.put(partition, keys, stored)
                self.store.put(latest, [lasting.encode()], {"partition": partition.hex()})
                self._remember(thread, len(raw), keys)
                state, forwarded, estimate, changed = stored, trimmed, report["after_tokens"], True
                if on_fallback:
                    on_fallback({"scope": "main", **report})
        return Exchange(forwarded, partition, [] if compacting else keys, state, changed, estimate, depth, diverged,
                        name, title, whole, compacting, lasting=lasting)

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
        if self.input_budget:
            # Keep costs, but never feed provider counts back into local-budget
            # planning or change the declared tokenizer using an observed ratio.
            state.setdefault("base", exchange.estimate)
            state["budget_tokens"] = exchange.estimate
            state["provider_tokens"] = input_tokens
            self.store.put(exchange.partition, exchange.keys, state)
            return
        if "base" not in state or (input_tokens and state.get("estimated")):
            state["base"], state["estimated"] = input_tokens or exchange.estimate, not input_tokens
        if input_tokens:
            state["tokens"] = input_tokens
            if exchange.estimate:  # (bounded: a count is the upstream's, the estimate stays a guess)
                state["rate"] = min(max(input_tokens / exchange.estimate, RATE[0]), RATE[1])
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
                private.append(self.event_log, json.dumps(record, ensure_ascii=False) + "\n")
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
    return [*codec.arrange(harness.place(head, state), cut == len(raw)), *_merge(tail, [i for i in own if i.ref >= cut])]


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
    if item.ref is not None:  # (a block of a user message is context of its own)
        original = originals[item.ref]
        return {"ref": item.ref} if (item.kind, item.text, item.media) == (original.kind, original.text, original.media) else {
            "ref": item.ref, "kind": item.kind.value, "text": item.text, **media}
    return {"kind": item.kind.value, "text": item.text, **(media if item.media else {}),
            **({"wire": item.wire} if item.wire else {})}


def _item(entry: dict[str, Any], originals: list[Item]) -> Item:
    media = tuple(Media(**m) for m in entry.get("media", []))
    if "ref" in entry:
        original = originals[entry["ref"]]
        kind = Kind(entry.get("kind", original.kind.value))
        return replace(original, kind=kind, text=entry["text"], media=media) if "text" in entry else original
    return Item(Kind(entry["kind"]), entry["text"], media=media, wire=entry.get("wire"))


def _wire(codec: Codec, raw: list[WireItem], originals: list[Item], item: Item) -> WireItem:
    if item.ref is not None:
        original = originals[item.ref]
        if (item.text, item.media) == (original.text, original.media):
            return raw[item.ref]
        gone = [m for m in original.media if m not in item.media]
        silent = gone and item.text.strip() == original.text.strip()  # (new text speaks for itself)
        return codec.edit(raw[item.ref], f"{item.text}\n{removed(gone)}".strip() if silent else item.text, item.media)
    return json.loads(item.wire) if item.wire else codec.write(item)


class _Summarizer:
    """Runs the summary request through the task's own upstream ("native continuation").

    A history too long for one request (Relay joined a long session late, or lost its state,
    while the harness kept every item) is summarized in order, a piece at a time, each piece
    after the summary of what came before it: the compactions that were skipped, caught up.
    The opt-in overflow fallback can discard old groups or shorten a large output
    when splitting alone cannot fit it; every such change is reported separately."""

    def __init__(self, codec: Codec, body: Body, history: Callable[[int], list[WireItem]], keep: int,
                 post: Post, window: float | None, per_token: float, instructions: str = "", *,
                 input_budget: InputBudget | None = None,
                 on_fallback: Callable[[dict[str, Any]], None] | None = None) -> None:
        # `history(cut)`: the wire items a conversation cut replaces, the harness's own included;
        # `keep`: the leading ones every piece carries (Codex keeps the initial context);
        # `window`: in the per-item estimates' terms; `instructions`: the strategy's, as the
        # requests the model saw carry them (a summary request is then their prefix, cached).
        self.codec, self.body, self.history, self.keep, self.post = codec, body, history, keep, post
        self.instructions = instructions
        self.input_budget = input_budget
        self.on_fallback = on_fallback
        self.budget = int(window * SUMMARY_SHARE) if window else None
        self.per_token, self.requests, self.usage = per_token, 0, []  # (each request's usage, as the upstream reported it)

    def summarize(self, cut: int, prompt: str) -> str:
        if self.input_budget:
            return self._summarize_bounded(cut, prompt)
        items = self.history(cut)
        lead, rest, budget, summary = items[: self.keep], items[self.keep :], self.budget, ""
        while True:
            fixed = [*lead, self.codec.user_message(f"{SUMMARY_PREFIX}\n{summary}")] if summary else lead
            overhead = approx_tokens(self.codec.preamble(self.body), self.per_token) + sum(map(self._tokens, fixed))
            piece = _piece(self.codec, rest, budget - overhead, self._tokens) if budget else len(rest)
            request = self.codec.summary_request(self.body, [*fixed, *rest[:piece]], prompt)
            if self.instructions:
                request = self.codec.with_instructions(request, self.instructions)
            status, payload = self._post(request)
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
        request = self.codec.request(self.body, system, wire)
        # The completion's first user prompt defines the RLM task; later iteration
        # outputs are expendable history. Never shorten that initial instruction.
        first_user = next((n for n, value in enumerate(wire) if self.codec.classify(value).kind is Kind.USER), None)
        protected = {first_user} if first_user is not None else set()
        status, payload = self._post(request, scope="completion", protected=protected,
                                     provider_fallback=True)
        if status >= 300 or not (isinstance(payload, dict) and self.codec.finished(payload)):
            raise RuntimeError(f"the model request failed with HTTP {status}: {payload!r:.300}")
        return self.codec.output_text(payload)

    def _summarize_bounded(self, cut: int, prompt: str) -> str:
        """Fit complete requests, including prompts, instructions and carryover.

        Legal call/result groups stay intact. The optional common fallback trims
        an oversized group only after the normal splitting path cannot fit it.
        """
        assert self.input_budget is not None
        items = self.history(cut)
        truncating = self.input_budget.overflow_fallback == "truncate"
        keep = self.keep
        if truncating:
            first_user = next((n for n, value in enumerate(items) if self.codec.classify(value).kind is Kind.USER), None)
            if first_user is not None:
                keep = max(keep, first_user + 1)
        lead, rest, summary = items[:keep], items[keep:], ""

        def request_for(selected: list[WireItem]) -> Body:
            request = self.codec.summary_request(self.body, selected, prompt)
            return self.codec.with_instructions(request, self.instructions) if self.instructions else request

        # Do not spend on summarizing an easy prefix when a later indivisible
        # group is already known not to fit, even without a carried summary.
        boundaries = sorted(self.codec.boundaries(rest))
        if rest and (not boundaries or boundaries[-1] != len(rest)):
            raise BudgetExceeded(self.input_budget.count(self.codec, request_for(items)),
                                 self.input_budget.limit_for(self.body), "summary",
                                 "The history has no complete final tool group.")
        for start, end in zip(boundaries, boundaries[1:]):
            minimal = request_for([*lead, *rest[start:end]])
            try:
                self.input_budget.check(self.codec, minimal, "summary")
            except BudgetExceeded as exc:
                if truncating:
                    continue  # The smallest group will be trimmed just before dispatch.
                raise BudgetExceeded(exc.tokens, exc.limit, "summary",
                                     "The smallest legal history group or fixed prompt cannot fit.") from exc
        maximum: int | None = None
        while True:
            fixed = [*lead, self.codec.user_message(f"{SUMMARY_PREFIX}\n{summary}")] if summary else lead

            def build(piece: int) -> Body:
                return request_for([*fixed, *rest[:piece]])

            legal = sorted(n for n in self.codec.boundaries(rest)
                           if n > 0 and (maximum is None or n <= maximum)) if rest else [0]
            low, high, chosen = 0, len(legal) - 1, None
            while low <= high:
                middle = (low + high) // 2
                candidate = build(legal[middle])
                try:
                    self.input_budget.check(self.codec, candidate, "summary")
                except BudgetExceeded:
                    high = middle - 1
                else:
                    chosen, low = legal[middle], middle + 1
            if chosen is None:
                if truncating and legal:
                    chosen = legal[0]
                else:
                    smallest = build(legal[0] if legal else len(rest))
                    raise BudgetExceeded(self.input_budget.count(self.codec, smallest),
                                         self.input_budget.limit_for(smallest), "summary",
                                         "The smallest legal history group or fixed prompt cannot fit.")
            selected = build(chosen)
            # with_instructions may insert developer items. Pin by identity, plus
            # the final summarization prompt; carryover summaries remain editable.
            lead_ids = {id(value) for value in lead}
            selected_items = self.codec.items(selected)
            protected = {n for n, value in enumerate(selected_items) if id(value) in lead_ids}
            protected.add(len(selected_items) - 1)
            status, payload = self._post(selected, protected=protected,
                                         provider_fallback=chosen == legal[0])
            if self.codec.is_overflow(status, payload):
                # The provider can still disagree with the local convention.
                # Retry a strictly shorter prefix, without repeating inference
                # on the same input and without falling back to another method.
                if chosen == legal[0]:
                    raise BudgetExceeded(self.input_budget.count(self.codec, build(chosen)),
                                         self.input_budget.limit_for(build(chosen)), "summary",
                                         "The provider rejected the smallest legal history group.")
                maximum = chosen - 1
                continue
            if status >= 300 or not (isinstance(payload, dict) and self.codec.finished(payload)):
                raise RuntimeError(f"the summary response failed: HTTP {status}: {payload!r:.300}")
            summary = self.codec.output_text(payload)
            if not summary:
                raise RuntimeError("the summary response contained no text")
            rest, maximum = rest[chosen:], None
            if not rest:
                return summary

    def _post(self, request: Body, *, scope: str = "summary", protected: set[int] | None = None,
              provider_fallback: bool = False) -> tuple[int, Any]:
        """One request of Relay's own, retried while rate limited, overloaded, cut short or empty
        (a model may end its turn at once, without a word)."""

        attempt = 0
        provider_trimmed = False
        while True:
            if self.input_budget:
                try:
                    self.input_budget.check(self.codec, request, scope)
                except BudgetExceeded:
                    if self.input_budget.overflow_fallback != "truncate":
                        raise
                    request, report = truncate_request(self.codec, request, self.input_budget,
                        scope=scope, protected=protected)
                    if protected is not None:
                        protected = {n for n, old in enumerate(report["kept_indices"]) if old in protected}
                    if self.on_fallback:
                        self.on_fallback({"scope": scope, **report})
            self.requests += 1
            status, payload = self.post(request)
            if isinstance(payload, dict) and (usage := payload.get("usage") or payload.get("usageMetadata")):
                self.usage.append(usage)
            if (self.input_budget and self.input_budget.overflow_fallback == "truncate"
                    and provider_fallback and not provider_trimmed and self.codec.is_overflow(status, payload)):
                request, report = truncate_request(self.codec, request, self.input_budget,
                    scope=scope, protected=protected,
                    target_limit=int(self.input_budget.count(self.codec, request) * 0.9))
                if protected is not None:
                    protected = {n for n, old in enumerate(report["kept_indices"]) if old in protected}
                provider_trimmed = True
                if self.on_fallback:
                    self.on_fallback({"scope": scope, "provider_rejected": True, **report})
                continue
            answered = isinstance(payload, dict) and self.codec.finished(payload) and self.codec.output_text(payload)
            incomplete = status < 300 and not answered
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
