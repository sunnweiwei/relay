from collections.abc import Mapping

from .base import Harness
from .claude_code import ClaudeCode
from .codex import Codex
from .crush import Crush
from .deepseek_harness import DeepSeekHarness
from .gemini_cli import GeminiCli
from .goose import Goose
from .hermes import Hermes
from .kimi_code import KimiCode
from .mini_swe import MiniSwe
from .nanobot import Nanobot
from .opencode import Kilo, OpenCode
from .openclaw import OpenClaw
from .pi import Pi
from .workbuddy import CodeBuddy, WorkBuddy

HARNESSES: dict[str, Harness] = {
    h.name: h
    for h in (
        Codex(), ClaudeCode(), GeminiCli(), Pi(), OpenCode(), Kilo(), Crush(), OpenClaw(),
        KimiCode(), Goose(), Hermes(), Nanobot(), DeepSeekHarness(), WorkBuddy(),
        CodeBuddy(), MiniSwe(), Harness(),
    )
}


def detect(headers: Mapping[str, str], forced: str | None = None, path: str = "") -> Harness:
    """The harness named by `forced`, else the one `relay install` mounted at the request
    path (`/up/<harness>-<n>/...`), else the first profile matching the headers."""

    if forced:
        if forced not in HARNESSES:
            raise ValueError(f"unknown harness {forced!r}; choose from {sorted(HARNESSES)}")
        return HARNESSES[forced]
    if path.startswith("/up/") and (name := path.split("/")[2].rpartition("-")[0]) in HARNESSES:
        return HARNESSES[name]
    return next((h for h in HARNESSES.values() if h.matches(headers)), HARNESSES["generic"])


__all__ = [
    "HARNESSES", "ClaudeCode", "CodeBuddy", "Codex", "Crush", "DeepSeekHarness", "GeminiCli", "Goose", "Harness", "Hermes",
    "Kilo", "KimiCode", "MiniSwe", "Nanobot", "OpenClaw",
    "OpenCode", "Pi",
    "detect",
]
