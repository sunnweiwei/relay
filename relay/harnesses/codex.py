"""OpenAI Codex CLI (Responses API).

Codex resends its history verbatim, so no canonicalization is needed. It injects
environment and instruction context as user messages; these markers mirror
`CONTEXTUAL_USER_FRAGMENT_MATCHERS` in codex-rs/core/src/context.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from ..core.ir import Item, Kind
from ..install import Setting
from .base import Harness

CONTEXT_MARKERS = (
    "<environment_context>",
    "<user_instructions>",
    "# AGENTS.md instructions",
    "<user_shell_command>",
    "<turn_aborted>",
    "<subagent_notification>",
    "<agent_message_board_notification>",
    "<codex_internal_context",
    "<goal_context>",
)


class Codex(Harness):
    name = "codex"

    def matches(self, headers: Mapping[str, str]) -> bool:
        return headers.get("originator", "").startswith("codex") or headers.get(
            "user-agent", ""
        ).startswith("codex")

    def refine(self, item: Item) -> Item:
        if item.kind is Kind.USER and item.text.lstrip().startswith(CONTEXT_MARKERS):
            return replace(item, kind=Kind.CONTEXT)
        return super().refine(item)

    def settings(self) -> list[Setting]:
        """Re-point the built-in OpenAI provider, so Codex keeps its own login and its
        ChatGPT-backend features (the wrapped URL keeps the `/backend-api/codex` path)."""

        home = Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser()
        auth = home / "auth.json"
        chatgpt = auth.exists() and json.loads(auth.read_text()).get("auth_mode") == "chatgpt"
        default = "https://chatgpt.com/backend-api/codex" if chatgpt else "https://api.openai.com/v1"
        return [Setting(home / "config.toml", ("openai_base_url",), endpoint=default)]

    def launch(self, relay_url: str, args: list[str]) -> tuple[list[str], dict[str, str]]:
        provider = (
            'model_providers.relay={name="Relay",'
            f'base_url="{relay_url}/v1",env_key="OPENAI_API_KEY",'
            'wire_api="responses",supports_websockets=false}'
        )
        return ["codex", "-c", 'model_provider="relay"', "-c", provider, *args], {}
