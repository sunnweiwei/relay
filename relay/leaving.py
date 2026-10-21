"""A harness leaving Relay (`relay uninstall`) without breaking its conversations.

Relay never shortens a harness's own history: a conversation that grew past the model's window
through Relay is sent whole once the harness goes direct, and fails. So a harness with such a
conversation leaves in two steps. First it stays pointed at Relay, which forwards each request
whose history fits the window as the harness sent it (as going direct would) and keeps managing
the others. A conversation the harness compacts itself (its own `/compact`, through Relay: the
compaction gets Relay's context, so it fits) is handed back: its own history is short from then
on. Once none of its conversations needs Relay, or each one that does was last used `GRACE` ago,
the harness's configuration is restored, by the Relay serving it or by the next `relay` command.

Relay remembers the conversations it served (`conversations.json`, in Relay's directory, readable
by its owner only: what each one's user first typed, its size, the model's window, when it was
last used), for as long as `GRACE`.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import install
from .settings import home

GRACE = 7 * 24 * 3600  # a conversation not used for this long no longer holds its harness on Relay
FITS = 0.9  # of the window: a history this size goes direct (the harness's own compaction then takes over)
KEPT = 500  # conversations remembered at most

_lock = threading.Lock()
_noted: dict[Path, dict[str, dict[str, Any]]] = {}  # Relay's directory -> its conversations, as served
_written: dict[Path, float] = {}


def _path(name: str) -> Path:
    return home() / name


def _read(name: str) -> dict[str, Any]:
    try:
        return json.loads(_path(name).read_text())
    except (OSError, ValueError):
        return {}


def _write(name: str, value: dict[str, Any]) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def seen(harness: str, conversation: str, title: str, whole: int, window: int | None, compacting: bool,
         now: float | None = None) -> None:
    """A request of `conversation`: its history's size, and whether it was the harness compacting it."""

    if not conversation or not window:
        return
    now = time.time() if now is None else now
    with _lock:
        _conversations = _noted.setdefault(home(), _read("conversations.json"))
        entry = _conversations.setdefault(conversation, {"harness": harness, "title": title})
        entry.update(harness=harness, last=now, window=window, whole=whole)
        entry["title"] = entry.get("title") or title
        if compacting:
            entry["handed"] = now  # its own history is the summary from here on
        elif whole <= FITS * window:
            entry.pop("handed", None)  # (fits as it is: nothing to hand back)
        for name in [n for n, e in _conversations.items() if now - e.get("last", 0) > GRACE]:
            del _conversations[name]
        for name in sorted(_conversations, key=lambda n: _conversations[n].get("last", 0))[:-KEPT]:
            del _conversations[name]
        if now - _written.get(home(), 0) > 5 or compacting:
            _write("conversations.json", _conversations)
            _written[home()] = now


def needing(harness: str, now: float | None = None) -> list[dict[str, Any]]:
    """The harness's conversations that would not fit direct, used within `GRACE`, newest first."""

    now = time.time() if now is None else now
    with _lock:
        conversations = _noted.get(home()) or _read("conversations.json")
        found = [e for e in conversations.values() if e.get("harness") == harness and now - e.get("last", 0) <= GRACE
                 and e.get("whole", 0) > FITS * e.get("window", 0) and "handed" not in e]
    return sorted(found, key=lambda e: -e["last"])


def leaving() -> dict[str, dict[str, Any]]:
    """The harnesses uninstalled but still pointed at Relay: {harness: {"since": time}}."""

    try:
        stat = _path("leaving.json").stat()
    except OSError:
        return {}
    return dict(_leaving(home(), (stat.st_mtime_ns, stat.st_size)))


@functools.lru_cache(maxsize=4)
def _leaving(directory: Path, stamp: tuple[int, int]) -> dict[str, dict[str, Any]]:  # (read once per change: every request asks)
    return _read("leaving.json")


def leave(harness: str, now: float | None = None) -> None:
    _write("leaving.json", {**leaving(), harness: {"since": time.time() if now is None else now}})
    _leaving.cache_clear()


def stay(harness: str) -> None:
    """Set up again: no longer leaving."""

    if harness in (current := leaving()):
        del current[harness]
        _write("leaving.json", current)
        _leaving.cache_clear()


def settle(now: float | None = None) -> list[str]:
    """Restore each leaving harness none of whose conversations needs Relay any more; which ones."""

    now = time.time() if now is None else now
    restored = []
    for harness in leaving():
        if not needing(harness, now):
            stay(harness)
            if harness in install.installed():
                install.uninstall(harness)
            restored.append(harness)
    if restored:  # (for `relay status` to say so, a day long)
        _write("restored.json", {**{h: t for h, t in _read("restored.json").items() if now - t < 86400},
                                 **{h: now for h in restored}})
    if restored and not install.installed():
        from . import service  # (no harness goes through Relay now: it is no longer started at login)
        service.release()
    return restored


def restored(now: float | None = None) -> dict[str, float]:
    """The harnesses restored by themselves in the last day: {harness: when}."""

    now = time.time() if now is None else now
    return {h: t for h, t in _read("restored.json").items() if now - t < 86400}


def flush() -> None:
    with _lock:
        if (conversations := _noted.get(home())) is not None:
            _write("conversations.json", conversations)
