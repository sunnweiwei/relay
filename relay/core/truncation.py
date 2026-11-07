"""Deterministically trim old editable text from a request that exceeds its input budget."""

from __future__ import annotations

from typing import Any

from ..protocols.base import Body, Codec, WireItem
from .budget import InputBudget
from .ir import Kind
from .media import removed

MARKER = "[harness truncated {removed} characters]"


def truncate_request(
    codec: Codec,
    body: Body,
    budget: InputBudget,
    *,
    scope: str,
    protected: set[int] | None = None,
    target_limit: int | None = None,
) -> tuple[Body, dict[str, Any]]:
    """Return a budget-fitting request and source-index metadata, or raise ``BudgetExceeded``.

    System/developer items and tool schemas are fixed by construction. Tool call/result groups
    are atomic. Media go first, oldest first, a pinned item's too: of an item only its media, and
    the newest stay, what the model is looking at. ``kept_indices`` maps each returned input item
    to its original input index.
    """
    from .budget import BudgetExceeded

    source = codec.items(body)
    limit = budget.limit_for(body)
    if target_limit is not None:
        limit = min(limit, target_limit)
    if limit < 0:
        raise ValueError("target_limit must be nonnegative")
    if codec.orphans(source):
        raise ValueError("cannot truncate a request with unpaired tool calls or results")
    legal = sorted(codec.boundaries(source))
    if not legal or legal[0] != 0 or legal[-1] != len(source):
        raise ValueError("request does not end at a complete protocol boundary")

    pins = set(protected or ())
    # Protocol-level instructions and inline schema items are never history to discard.
    pins.update(i for i, wire in enumerate(source) if codec.classify(wire).kind is Kind.SYSTEM)
    if any(index < 0 or index >= len(source) for index in pins):
        raise IndexError("protected item index is outside the request")
    if protected is None:
        users = [i for i, wire in enumerate(source) if codec.classify(wire).kind is Kind.USER]
        if users:
            pins.add(users[0])
            if len(users) > 1:
                pins.add(users[-1])

    groups = [(left, right) for left, right in zip(legal, legal[1:])]
    kept = list(range(len(source)))

    def make_body(indices: list[int], replacements: dict[int, WireItem] | None = None) -> Body:
        replacements = replacements or {}
        return codec.with_items(body, [replacements.get(index, source[index]) for index in indices])

    def count(indices: list[int], replacements: dict[int, WireItem] | None = None) -> int:
        return budget.count(codec, make_body(indices, replacements))

    before = budget.count(codec, body)
    if before <= limit:
        return body, {"before_tokens": before, "after_tokens": before, "dropped_groups": 0,
                      "dropped_items": 0, "truncated_text_items": 0, "stripped_media_items": 0, "kept_indices": kept}

    edits: dict[int, WireItem] = {}
    with_media = [i for i, wire in enumerate(source)
                  if (item := codec.classify(wire)).media and item.kind in (Kind.USER, Kind.TOOL_RESULT)]
    for index in with_media[:-1]:
        if count(kept, edits) <= limit:
            break
        item = codec.classify(source[index])
        try:
            edits[index] = codec.edit(source[index], f"{item.text}\n{removed(item.media)}".strip(), ())
        except (TypeError, ValueError):
            continue
    stripped = set(edits)

    dropped_groups = dropped_items = 0
    # Remove oldest removable groups, retaining the newest one while any other removable group
    # remains. Pinned groups are never removed, even when they contain uneditable tool calls.
    active_groups = list(groups)
    while count(kept, edits) > limit:
        removable = [g for g in active_groups if not any(i in pins for i in range(*g))]
        if len(removable) <= 1:
            break
        left, right = removable[0]
        group = set(range(left, right))
        kept = [i for i in kept if i not in group]
        active_groups.remove((left, right))
        dropped_groups += 1
        dropped_items += right - left

    # Shrink plain message/output text in oldest-to-newest order. Binary search retains the
    # largest prefix/suffix payload that makes the complete request fit.
    for original_index in kept:
        if count(kept, edits) <= limit:
            break
        wire = source[original_index]
        item = codec.classify(wire)
        if original_index in pins or item.kind not in (Kind.USER, Kind.ASSISTANT, Kind.TOOL_RESULT):
            continue
        typ = wire.get("type", "message")
        if typ != "message" and not typ.endswith("_output"):
            continue
        raw_text = wire.get("content") if typ == "message" else wire.get("output")
        if isinstance(raw_text, list):
            # codec.edit rebuilds content parts, so only touch a single plain text part whose
            # structure has no additional fields to preserve.
            if (len(raw_text) != 1 or not isinstance(raw_text[0], dict)
                    or set(raw_text[0]) != {"type", "text"}
                    or raw_text[0].get("type") not in {"input_text", "output_text", "text"}):
                continue
        elif not isinstance(raw_text, str):
            continue
        # Require the protocol's text extraction to account for the entire editable payload;
        # media and mixed opaque content remain untouched.
        if item.media or not item.text:
            continue
        current = item.text
        try:
            codec.edit(wire, current, item.media)  # confirms this item supports text editing
        except (TypeError, ValueError):
            continue

        def candidate(retained: int) -> WireItem:
            marker = MARKER.format(removed=max(1, len(current) - retained))
            left = retained // 2
            right = retained - left
            text = current[:left] + marker + (current[-right:] if right else "")
            return codec.edit(wire, text, item.media)

        # Search from marker-only to the full text; fitting is monotonic enough for plain text
        # under the declared tokenizer, and every chosen candidate is counted again below.
        low, high, best = 0, len(current), None
        while low <= high:
            middle = (low + high) // 2
            value = candidate(middle)
            trial = {**edits, original_index: value}
            if count(kept, trial) <= limit:
                best, low = value, middle + 1
            else:
                high = middle - 1
        if best is not None:
            edits[original_index] = best
        else:
            # Marker-only may still be too large. Continue shrinking other old text first.
            marker_only = codec.edit(wire, MARKER.format(removed=len(current)), item.media)
            edits[original_index] = marker_only

    # If output text reductions were insufficient, whole unpinned groups may be dropped, but
    # never the only remaining group. Recount after each mutation.
    while count(kept, edits) > limit:
        removable = [g for g in active_groups if not any(i in pins for i in range(*g))]
        if not removable:
            break
        left, right = removable[0]
        if len(removable) == 1:
            outside = [i for i in kept if i not in range(left, right)]
            other_task = any(codec.classify(source[i]).kind is Kind.USER for i in outside)
            if not other_task:
                break
        group = set(range(left, right))
        kept = [i for i in kept if i not in group]
        edits = {i: value for i, value in edits.items() if i in kept}
        active_groups.remove((left, right))
        dropped_groups += 1
        dropped_items += right - left

    result = make_body(kept, edits)
    after = budget.count(codec, result)
    if after > limit:
        raise BudgetExceeded(after, limit, scope, "Pinned content and protocol overhead cannot fit.")
    if codec.orphans(codec.items(result)):
        raise ValueError("truncation produced unpaired tool calls or results")
    return result, {"before_tokens": before, "after_tokens": after, "dropped_groups": dropped_groups,
                    "dropped_items": dropped_items, "truncated_text_items": len(set(edits) - stripped),
                    "stripped_media_items": len(stripped & set(kept)),
                    "kept_indices": kept}
