"""CLM's edited context, kept on disk so that a Relay restart does not undo the model's edits.

Relay keeps a conversation's context in memory (the prefix store), and the harness resends its
original history on every request. For compaction a lost context is recomputed from that
history; a CLM context is not: it holds decisions the model made over many requests. So after
every request CLM writes a checkpoint next to its mirror files: the context the model now sees,
as references into the harness's history (`ref`, with new text where the model rewrote a turn)
and the items the model added, plus CLM's own state. Only the strategy's view is saved; the
engine's store is untouched.

A checkpoint is found again by the conversation's opening (the harness's history up to and
including the task), which is the same across restarts, and is used only when the history it
was written from is exactly the start of the history the harness sends now. A branch that left
that history, or a history the harness rewrote, finds no checkpoint and starts from its own
history, as before.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.ir import Item, Kind, Request

KEEP = 16  # checkpoints per conversation opening (branches of one session share an opening)
FOLDER = ".checkpoints"


def _identity(item: Item) -> str:
    return hashlib.sha256(f"{item.kind.value}\0{item.ref}\0{item.text}".encode()).hexdigest()


def _digest(items: list[Item] | tuple[Item, ...]) -> str:
    return hashlib.sha256("\n".join(map(_identity, items)).encode()).hexdigest()


def _file(directory: str, history: tuple[Item, ...]) -> Path | None:
    task = next((n + 1 for n, item in enumerate(history) if item.kind is Kind.USER), 0)
    if not task:
        return None
    return Path(directory) / FOLDER / f"{_digest(history[:task])[:24]}.json"


def _load(path: Path) -> list[dict[str, Any]]:
    try:
        checkpoints = json.loads(path.read_text(encoding="utf-8"))
        return checkpoints if isinstance(checkpoints, list) else []
    except (OSError, ValueError):
        return []


def save(directory: str, request: Request, items: list[Item], state: dict[str, Any]) -> None:
    """Remember the context the model now sees (`items`, without the strategy's guidance)."""

    path = _file(directory, request.history)
    if path is None:
        return
    originals = {item.ref: item for item in request.history}
    entries = []
    for item in items:
        if item.ref is not None and item.ref in originals:
            entry: dict[str, Any] = {"ref": item.ref}
            if item.text != originals[item.ref].text:
                entry["text"] = item.text
        elif item.ref is None and not item.media:
            entry = {"kind": item.kind.value, "text": item.text, **({"wire": item.wire} if item.wire else {})}
        else:
            return  # something this file cannot carry: keep no checkpoint rather than a wrong one
        entries.append(entry)
    checkpoint = {"length": len(request.history), "basis": _digest(request.history), "items": entries,
                  "state": state, "conversation": request.conversation}
    kept = [c for c in _load(path) if c.get("basis") != checkpoint["basis"]][: KEEP - 1]
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not (ignore := path.parent.parent / ".gitignore").exists():
        ignore.write_text("*\n", encoding="utf-8")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps([checkpoint, *kept]), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def restore(directory: str, request: Request) -> tuple[Request, dict[str, Any]] | None:
    """The request as the model saw it before Relay lost its state, and CLM's state then; None
    when no checkpoint was written from the start of this history."""

    path = _file(directory, request.history)
    if path is None or not path.exists():
        return None
    history = request.history
    matching = [c for c in _load(path)
                if isinstance(c.get("length"), int) and c["length"] <= len(history)
                and _digest(history[: c["length"]]) == c.get("basis")]
    if not matching:
        return None
    checkpoint = max(matching, key=lambda c: c["length"])
    originals = {item.ref: item for item in history}
    items: list[Item] = []
    for entry in checkpoint["items"]:
        if "ref" in entry:
            if entry["ref"] not in originals:
                return None
            item = originals[entry["ref"]]
            items.append(replace(item, text=entry["text"]) if "text" in entry else item)
        else:
            items.append(Item(Kind(entry["kind"]), entry["text"], wire=entry.get("wire")))
    items += history[checkpoint["length"]:]  # what the harness sent since
    from .clm import _size  # (the same measure CLM's receipts use)

    tokens = max(0, request.tokens - _size(request.current) + _size(items))
    boundaries = frozenset(n for n in range(len(items) + 1) if n == len(items) or not (
        items[n].kind is Kind.TOOL_RESULT or (n and items[n - 1].kind is Kind.TOOL_CALL)))
    # The conversation keeps the name its file's headers carry (Relay names it anew after a restart).
    conversation = checkpoint.get("conversation") or request.conversation
    return (replace(request, current=tuple(items), boundaries=boundaries, tokens=tokens, conversation=conversation),
            checkpoint["state"])
