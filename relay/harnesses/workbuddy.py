"""Tencent's WorkBuddy (desktop) and CodeBuddy Code (CLI), which share one engine.

Custom models are entries of models.json whose `url` is the full chat/completions
endpoint; built-in models go through Tencent's own service and cannot be wrapped.
A project-level .workbuddy/.codebuddy models.json takes precedence over the user's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..install import Setting
from .base import Harness


class WorkBuddy(Harness):
    name = "workbuddy"
    config_env, config_dir = "WORKBUDDY_CONFIG_DIR", "~/.workbuddy"

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
        return settings


class CodeBuddy(WorkBuddy):
    name = "codebuddy"
    config_env, config_dir = "CODEBUDDY_CONFIG_DIR", "~/.codebuddy"
