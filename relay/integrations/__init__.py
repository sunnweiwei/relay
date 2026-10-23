"""Integration paths: how a strategy runs inside a harness.

A strategy is written once, against a view of the conversation, and decides on every request.
Every harness can run it through the proxy (`relay install` points its model endpoint at Relay,
which rewrites each request); some also offer a native hook, which runs the same strategy inside
the harness and leaves its endpoint alone. The paths differ in where a rewrite lives (in Relay,
or in the harness's own history) and in what they can see and write, never in what the strategy
decides. Each harness lists its paths in the order the adapter prefers them; `relay install
--via auto` takes the first that carries the strategy and runs with the harness's release, and
`relay strategy` moves a harness set up that way to the path its new strategy needs.
"""

from __future__ import annotations

import json
from typing import Any

from ..harnesses import Harness
from ..settings import home
from .base import Installation, Integration, Proxy, Supplied, SummaryNeeded
from .claude_code import ClaudeCodeHook

# Claude Code: the hook first, as it keeps Claude Code's endpoint, so Remote Control and every
# login are untouched; the proxy for a strategy the hook cannot carry.
HOOKS: dict[str, Integration] = {"claude_code": ClaudeCodeHook()}

# What each strategy's contexts hold besides the conversation's own messages, which its path must
# carry: instructions (SYSTEM items) the same all through a conversation, or changing within it;
# notes for one request; new text for items it keeps (a tool's result); a new context on every
# request; a completion of its own (`summarizer.complete`); and the whole history, past what the
# harness keeps.
NEEDS: dict[str, frozenset[str]] = {
    "compaction": frozenset(), "selfcompact": frozenset(),
    "prolong": frozenset({"instructions", "history"}), "mem1": frozenset({"instructions", "every request"}),
    "autocompact": frozenset({"instructions", "rewrites"}), "folding": frozenset({"instructions", "rewrites"}),
    "acm": frozenset({"instructions", "rewrites", "notes"}), "clm": frozenset({"instructions", "rewrites", "notes"}),
    "ace": frozenset({"changing instructions", "completions"}),
    "rlm": frozenset({"completions", "history"}), "rlm_persistent": frozenset({"completions", "history"}),
}


def paths(harness: Harness) -> list[Integration]:
    """The harness's integration paths, the preferred first."""

    return [*([HOOKS[harness.name]] if harness.name in HOOKS else []), Proxy(harness)]


def uncarried(path: Integration, strategy: str) -> str | None:
    """What of `strategy`'s contexts `path` cannot carry, or None."""

    if missing := sorted(NEEDS.get(strategy, path.carries) - path.carries):
        return f"{strategy} needs {', '.join(missing)}, which the {path.name} cannot carry"
    return None


def unfit(path: Integration, strategy: str) -> str | None:
    """Why `path` cannot run `strategy` here, or None."""

    return uncarried(path, strategy) or path.unavailable()


def choose(harness: Harness, via: str = "auto", strategy: str = "compaction") -> Integration:
    """The path `relay install --via` names, or (`auto`) the preferred one that runs `strategy`."""

    offered = [path for path in paths(harness) if via in ("auto", path.name)]
    if not offered:
        names = ", ".join(path.name for path in paths(harness))
        raise ValueError(f"{harness.name} has no {via} path (it has: {names})")
    if via != "auto" and (why := unfit(offered[0], strategy)):
        raise ValueError(f"{why}; `--via auto` chooses the path that can")
    return next(path for path in offered if via != "auto" or unfit(path, strategy) is None)


def remember(name: str, via: str, path: str) -> None:
    """How a harness was installed: the path asked for (`auto`, or one named) and the one taken."""

    _save({**_load(), name: {"via": via, "path": path}})


def recorded(name: str) -> dict[str, Any] | None:
    return _load().get(name)


def forget(name: str) -> None:
    if name in (known := _load()):
        known.pop(name)
        _save(known)


def _load() -> dict[str, Any]:
    try:
        return json.loads((home() / "paths.json").read_text())
    except (OSError, ValueError):
        return {}


def _save(known: dict[str, Any]) -> None:
    (home() / "paths.json").parent.mkdir(parents=True, exist_ok=True)
    (home() / "paths.json").write_text(json.dumps(known, indent=2) + "\n")


__all__ = ["HOOKS", "NEEDS", "ClaudeCodeHook", "Installation", "Integration", "Proxy", "Supplied", "SummaryNeeded",
           "choose", "forget", "paths", "recorded", "remember", "uncarried", "unfit"]
