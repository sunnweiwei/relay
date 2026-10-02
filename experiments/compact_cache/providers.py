"""Model API configuration shared by all live Harness adapters."""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    base_url: str
    api_key: str


def _secret(path: Path) -> str:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError(f"{path.name} must be owned by you and mode 600")
    value = path.read_text().rstrip("\r\n")
    if not value:
        raise ValueError(f"{path.name} is empty")
    return value


def route_for(model: str) -> ModelRoute:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", model):
        raise ValueError("invalid model identifier")
    if model.startswith("gpt-"):
        return ModelRoute("openai", model, "https://api.openai.com/v1",
                          _secret(Path("/private/tmp/relay-openai-api-key")))
    raise ValueError(f"{model}: no verified live provider route")
