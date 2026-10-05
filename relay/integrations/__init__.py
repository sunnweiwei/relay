"""Integration paths: how a strategy runs inside a harness.

A strategy is written once, against a view of the conversation, and decides on every request.
Every harness can run it through the proxy (`relay install` points its model endpoint at Relay,
which rewrites each request); some also offer a native hook, which runs the same strategy inside
the harness and leaves its endpoint alone. The paths differ in where a rewrite lives (in Relay,
or in the harness's own history) and in what they can see and write, never in what the strategy
decides. Each harness lists its paths in the order the adapter prefers them; `relay install
--via auto` takes the first.
"""

from __future__ import annotations

from ..harnesses import Harness
from .base import Installation, Integration, Proxy, Supplied, SummaryNeeded
from .claude_code import ClaudeCodeHook

# Claude Code: the proxy first, as it follows Codex's timing exactly and continues each
# sub-agent's own conversation for its summary; the hook (early access) keeps Claude Code's
# endpoint, so Remote Control and every login are untouched.
HOOKS: dict[str, Integration] = {"claude_code": ClaudeCodeHook()}


def paths(harness: Harness) -> list[Integration]:
    """The harness's integration paths, the preferred first."""

    return [Proxy(harness), *([HOOKS[harness.name]] if harness.name in HOOKS else [])]


def choose(harness: Harness, via: str = "auto") -> Integration:
    """The path `relay install --via` names (`auto`: the harness's preferred one)."""

    offered = [path for path in paths(harness) if via in ("auto", path.name)]
    if not offered:
        names = ", ".join(path.name for path in paths(harness))
        raise ValueError(f"{harness.name} has no {via} path (it has: {names})")
    return offered[0]


__all__ = ["HOOKS", "ClaudeCodeHook", "Installation", "Integration", "Proxy", "Supplied", "SummaryNeeded",
           "choose", "paths"]
