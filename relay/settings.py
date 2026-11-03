"""Relay's settings: one JSON file in Relay's directory (`~/.config/relay/settings.json`),
written by `relay setup`, `relay strategy` and `relay set`, read by every part of Relay.

A setting is a name (`strategy`, `clm_budget`, `context_window`, ...) and a JSON value; a name
not set takes its default. A running Relay follows the file: the strategy and its settings
change on the next request (`version` tells when the file changed); where Relay listens, its
upstreams and its store are read when it starts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from . import private

_lock = Lock()
_cache: tuple[tuple[int, int] | None, dict[str, Any]] = (None, {})


def home() -> Path:
    """Relay's directory: its settings, the harnesses it installed, its service's log."""

    return Path(os.getenv("RELAY_HOME", "~/.config/relay")).expanduser()


def path() -> Path:
    return home() / "settings.json"


def version() -> tuple[int, int] | None:
    """The file's identity as it is now (None: no file), to tell when it changed."""

    try:
        stat = path().stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def load() -> dict[str, Any]:
    """Every setting written, as written."""

    global _cache
    stamp = version()
    with _lock:
        if stamp != _cache[0]:
            try:
                data = json.loads(path().read_text(encoding="utf-8")) if stamp else {}
            except (OSError, ValueError):
                data = {}  # (a file being rewritten: the next request reads it again)
                stamp = None
            _cache = (stamp, data if isinstance(data, dict) else {})
        return dict(_cache[1])


def value(name: str, default: str | None = None) -> str | None:
    """A setting as text (how Relay's parts parse it), or `default` when it is not set."""

    found = load().get(name)
    if found is None:
        return default
    return json.dumps(found) if isinstance(found, (dict, list)) else str(found).lower() if isinstance(found, bool) else str(found)


def save(changes: dict[str, Any], removed: tuple[str, ...] = ()) -> dict[str, Any]:
    """Write `changes` (a None value unsets) and unset `removed`; the settings now."""

    settings = load()
    settings.update(changes)
    for name in (*removed, *(name for name, given in changes.items() if given is None)):
        settings.pop(name, None)
    path().parent.mkdir(parents=True, exist_ok=True)
    private.write(path(), json.dumps(settings, indent=2, sort_keys=True) + "\n")  # (API keys may be in it)
    return settings


def parse(assignment: str) -> tuple[str, Any]:
    """`name=value` from the command line: the value as JSON where it is JSON (26000, true,
    null), else as text."""

    name, separator, text = assignment.partition("=")
    if not separator or not name.strip():
        raise ValueError(f"expected name=value, not {assignment!r}")
    try:
        return name.strip(), json.loads(text)
    except ValueError:
        return name.strip(), text
