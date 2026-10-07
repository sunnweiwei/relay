"""Kimi Code through the proxy: its hooks report the session to Relay (`relay.core.local`).

`relay install kimi_code` adds hooks run before each tool call and as each prompt is submitted
(those Kimi Code waits for, so the report reaches Relay before the request that follows):
`plugins/kimi_code/report.py` reads Kimi Code's hook input (its session id) and tells Relay where
the session's event log is and how long it is, which Kimi Code's own compaction points the model
at (a few lines short: the log grows by the call and its result before the next request).
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from ..install import Setting, read
from .base import Installation

EVENTS = ("PreToolUse", "UserPromptSubmit")
REPORT = Path(__file__).parent / "plugins" / "kimi_code" / "report.py"


def home() -> Path:
    return Path(os.getenv("KIMI_CODE_HOME", "~/.kimi-code")).expanduser()


def session_report(relay_url: str) -> Installation:
    """Its hooks, after the user's own (`hooks = [...]`; tables of them are the user's to extend)."""

    config = home() / "config.toml"
    command = f"{shlex.quote(sys.executable)} -I {shlex.quote(str(REPORT))} {shlex.quote(relay_url)}"
    if config.exists() and "[[hooks]]" in config.read_text(encoding="utf-8"):
        return Installation([], (f"{config} lists its hooks as [[hooks]] tables: add `{command}` as a hook for "
                                 f"{' and '.join(EVENTS)} for Kimi Code's summaries to point at its event log",))
    ours = [{"event": event, "command": command, "timeout": 10} for event in EVENTS]
    return Installation([Setting(config, ("hooks",), [*(read(config, ("hooks",)) or []), *ours])])
