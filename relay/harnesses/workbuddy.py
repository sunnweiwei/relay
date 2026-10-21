"""Tencent's WorkBuddy (desktop) and CodeBuddy Code (CLI), which share one engine.

Custom models are entries of models.json whose `url` is the full chat/completions
endpoint; built-in models go through Tencent's own service and cannot be wrapped.
A project-level .workbuddy/.codebuddy models.json takes precedence over the user's.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..core.ir import Item
from ..install import Setting
from .base import REMINDER, Harness, command_hook


class WorkBuddy(Harness):
    name = "workbuddy"
    verified = "2.164.0"
    agent = re.compile(r"\bCodeBuddy/(\d[\w.-]*)")
    # Memory and rules files, re-rendered in the first user message around its <user_query>.
    injected = (REMINDER, re.compile(r"<always_applied_workspace_rules>.*?</always_applied_workspace_rules>", re.S))
    config_env, config_dir = "WORKBUDDY_CONFIG_DIR", "~/.workbuddy"

    def state_key(self, item: Item) -> str | None:
        """Its memory and rules, wherever they come (after compacting itself too)."""

        text = item.text.lstrip()
        if text.startswith('<system-reminder data-role="memory">'):
            return "memory"
        return "rules" if text.startswith("<always_applied_workspace_rules>") else None

    def starter(self, command: str) -> list[Setting]:
        user = Path(os.getenv(self.config_env) or self.config_dir).expanduser() / "settings.json"
        return [command_hook(user, "SessionStart", command)]

    def settings(self) -> list[Setting]:
        config = Path(os.getenv(self.config_env) or self.config_dir).expanduser() / "models.json"
        models = json.loads(config.read_text()).get("models", []) if config.exists() else []
        settings = [
            Setting(config, ("models", f"[id={model['id']}]", "url"), endpoint="")
            for model in models
            if model.get("id") and str(model.get("url", "")).startswith("http")  # not a ${VAR} reference
        ]
        if not settings:
            raise ValueError(f"add a custom model with a `url` to {config} first")
        # Automatic compaction off, and the forced one (at 92% of a model's contextWindow or
        # maxInputTokens, the only one this build runs) moved to the window's edge.
        user = config.with_name("settings.json")
        return [*settings, Setting(user, ("autoCompactEnabled",), False),
                Setting(user, ("env", "CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE"), "100")]


class CodeBuddy(WorkBuddy):
    name = "codebuddy"
    verified = "2.164.0"
    config_env, config_dir = "CODEBUDDY_CONFIG_DIR", "~/.codebuddy"
